// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! File-level pruning based on partition values and file-level statistics

use std::sync::Arc;

use arrow::datatypes::{FieldRef, SchemaRef};
use datafusion_common::{Result, internal_datafusion_err, pruning::PrunableStatistics};
use datafusion_datasource::PartitionedFile;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use datafusion_physical_plan::metrics::Count;
use log::debug;

use crate::{MAX_IN_LIST_SIZE, PruningPredicate, PruningPredicateBuilder};

/// Prune based on file-level statistics.
///
/// Note: Partition column pruning is handled earlier via `replace_columns_with_literals`
/// which substitutes partition column references with their literal values before
/// the predicate reaches this pruner.
pub struct FilePruner {
    predicate: Arc<dyn PhysicalExpr>,
    /// Tracks the dynamic filters inside `predicate` so we only rebuild the
    /// pruning predicate when one of them has actually moved.
    tracking: DynamicFilterTracking,
    /// Whether [`Self::should_prune`] has built+evaluated the pruning predicate
    /// at least once. The first check always runs; subsequent checks only run
    /// when a watched dynamic filter changed.
    checked_once: bool,
    /// Schema used for pruning (the logical file schema).
    file_schema: SchemaRef,
    file_stats_pruning: PrunableStatistics,
    predicate_creation_errors: Count,
    /// `IN (...)` rewrite cap, matching the row-group and page pruning
    /// predicates built later for the same file so the first build can be
    /// reused for them (see [`Self::reusable_pruning_predicate`]).
    max_in_list_size: usize,
    /// The result of the last pruning predicate build (valid once
    /// `checked_once` is set), kept so the scan can reuse it instead of
    /// building the same predicate again per file.
    built: Option<Arc<PruningPredicate>>,
}

impl FilePruner {
    #[deprecated(
        since = "52.0.0",
        note = "Use `try_new` instead which returns None if no statistics are available"
    )]
    #[expect(clippy::needless_pass_by_value)]
    pub fn new(
        predicate: Arc<dyn PhysicalExpr>,
        logical_file_schema: &SchemaRef,
        _partition_fields: Vec<FieldRef>,
        partitioned_file: PartitionedFile,
        predicate_creation_errors: Count,
    ) -> Result<Self> {
        Self::try_new(
            predicate,
            logical_file_schema,
            &partitioned_file,
            predicate_creation_errors,
        )
        .ok_or_else(|| {
            internal_datafusion_err!(
                "FilePruner::new called on a file without statistics: {:?}",
                partitioned_file
            )
        })
    }

    /// Create a file pruner for this file, or `None` when pruning it cannot
    /// help.
    ///
    /// Returns `None` when the file has no statistics struct to evaluate a
    /// pruning predicate against, or when the predicate is purely static and the
    /// file has no usable column statistics — in that case planning already did
    /// everything such a pruner could. A predicate carrying a dynamic filter is
    /// always accepted (given a statistics struct), since it may prune via
    /// partition-value folding even without column statistics.
    pub fn try_new(
        predicate: Arc<dyn PhysicalExpr>,
        file_schema: &SchemaRef,
        partitioned_file: &PartitionedFile,
        predicate_creation_errors: Count,
    ) -> Option<Self> {
        // A pruning predicate is evaluated against a statistics struct, so one
        // must exist (its columns may all be `Absent`).
        let file_stats = partitioned_file.statistics.as_ref()?;
        let tracking = DynamicFilterTracking::classify(&predicate);
        // Only build a pruner when it could prune something planning didn't
        // already: the file has real column statistics, or the predicate carries
        // a dynamic filter (whose value, or folded partition columns, can prune
        // even without column statistics). For a purely static predicate with no
        // usable stats there is nothing to gain.
        if !partitioned_file.has_statistics() && !tracking.contains_dynamic_filter() {
            return None;
        }
        let file_stats_pruning =
            PrunableStatistics::new(vec![file_stats.clone()], Arc::clone(file_schema));
        Some(Self {
            predicate,
            tracking,
            checked_once: false,
            file_schema: Arc::clone(file_schema),
            file_stats_pruning,
            predicate_creation_errors,
            max_in_list_size: MAX_IN_LIST_SIZE,
            built: None,
        })
    }

    /// Sets the `IN (...)` rewrite cap used to build the pruning predicate.
    /// Pass the cap the scan uses for row-group pruning so the build can be
    /// shared with it.
    pub fn with_max_in_list_size(mut self, max_in_list_size: usize) -> Self {
        self.max_in_list_size = max_in_list_size;
        self
    }

    /// Returns the pruning predicate the last [`Self::should_prune`] built, when
    /// building one from `predicate` against `schema` would produce the same
    /// thing: `predicate` is the very expression this pruner holds, `schema`
    /// equals its file schema, and no dynamic filter inside it can still move.
    ///
    /// The outer `None` means "not reusable, build it yourself"; the inner
    /// `None` means the build found nothing to prune with.
    pub fn reusable_pruning_predicate(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        schema: &SchemaRef,
    ) -> Option<Option<Arc<PruningPredicate>>> {
        let reusable = self.checked_once
            && !self.is_watching()
            && Arc::ptr_eq(&self.predicate, predicate)
            && (Arc::ptr_eq(&self.file_schema, schema) || self.file_schema == *schema);
        reusable.then(|| self.built.clone())
    }

    /// Returns `true` if this pruner watches a dynamic filter that can still
    /// change, meaning [`Self::should_prune`] is worth re-checking as the scan
    /// progresses. When `false`, the predicate is effectively static for the
    /// remainder of the scan and the caller can avoid wrapping the stream in a
    /// per-batch re-pruning adapter.
    pub fn is_watching(&self) -> bool {
        matches!(self.tracking, DynamicFilterTracking::Watching(_))
    }

    pub fn should_prune(&mut self) -> Result<bool> {
        // Building the pruning predicate is expensive (it involves expression
        // analysis), so we only do it on the first check and whenever a dynamic
        // filter inside the predicate has actually moved.
        //
        // Dynamic filter expressions can change their values during query
        // execution; `DynamicFilterTracking` watches the still-incomplete
        // filters and reports a change at most once per update. A purely static
        // predicate (or one whose dynamic filters have all completed) is checked
        // exactly once.
        let should_build = if self.checked_once {
            self.tracking.watcher().is_some_and(|w| w.changed())
        } else {
            self.checked_once = true;
            true
        };
        if !should_build {
            return Ok(false);
        }
        let pruning_predicate = PruningPredicateBuilder::new()
            .with_file_schema(Arc::clone(&self.file_schema))
            .with_error_counter(&self.predicate_creation_errors)
            .with_max_in_list_size(self.max_in_list_size)
            .build(Arc::clone(&self.predicate));
        self.built.clone_from(&pruning_predicate);
        let Some(pruning_predicate) = pruning_predicate else {
            return Ok(false);
        };
        match pruning_predicate.prune(&self.file_stats_pruning) {
            Ok(values) => {
                assert_eq!(values.len(), 1);
                // We expect a single container -> if all containers are false skip this file
                if values.into_iter().all(|v| !v) {
                    return Ok(true);
                }
            }
            // Stats filter array could not be built, so we can't prune
            Err(e) => {
                debug!("Ignoring error building pruning predicate for file: {e}");
                self.predicate_creation_errors.add(1);
            }
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::ScalarValue;
    use datafusion_common::stats::{ColumnStatistics, Precision, Statistics};
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{binary, col, lit};

    /// A file whose single `id` column spans 1..=10, and the pruner for
    /// `id = 5` over it.
    fn pruner() -> (FilePruner, Arc<dyn PhysicalExpr>, SchemaRef) {
        let schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let predicate = binary(
            col("id", &schema).unwrap(),
            Operator::Eq,
            lit(5i32),
            &schema,
        )
        .unwrap();
        let stats = Statistics {
            num_rows: Precision::Exact(10),
            total_byte_size: Precision::Absent,
            column_statistics: vec![
                ColumnStatistics::new_unknown()
                    .with_min_value(Precision::Exact(ScalarValue::Int32(Some(1))))
                    .with_max_value(Precision::Exact(ScalarValue::Int32(Some(10)))),
            ],
        };
        let file = PartitionedFile::new("f.parquet", 1).with_statistics(Arc::new(stats));
        let pruner =
            FilePruner::try_new(Arc::clone(&predicate), &schema, &file, Count::new())
                .unwrap();
        (pruner, predicate, schema)
    }

    #[test]
    fn reuses_the_build_only_for_the_same_predicate_and_schema() {
        let (mut pruner, predicate, schema) = pruner();
        assert!(
            pruner
                .reusable_pruning_predicate(&predicate, &schema)
                .is_none()
        );

        assert!(!pruner.should_prune().unwrap());
        let reused = pruner.reusable_pruning_predicate(&predicate, &schema);
        assert!(matches!(reused, Some(Some(_))));

        let equal_schema = Arc::new(schema.as_ref().clone());
        assert!(
            pruner
                .reusable_pruning_predicate(&predicate, &equal_schema)
                .is_some()
        );

        let other = binary(
            col("id", &schema).unwrap(),
            Operator::Eq,
            lit(5i32),
            &schema,
        )
        .unwrap();
        assert!(pruner.reusable_pruning_predicate(&other, &schema).is_none());

        let other_schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        assert!(
            pruner
                .reusable_pruning_predicate(&predicate, &other_schema)
                .is_none()
        );
    }
}
