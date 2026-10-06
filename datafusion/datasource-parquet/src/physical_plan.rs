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

//! Per-file Parquet read planning for readers that drive their own decoder.
//!
//! [`PerFileParquetReadPlanner`] runs the same per-file steps `ParquetSource`
//! runs after footer discovery — expression adaptation to the file schema,
//! nested projection, decoder row-filter construction, and row-group/page
//! pruning predicates — and returns them as one owned
//! [`PerFileParquetReadPlan`]. It opens no reader and performs no IO, so a
//! reader outside `ParquetSource` can apply exactly the plan `DataFusion`
//! would.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::Result;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr::utils::reassign_expr_columns;
use datafusion_physical_expr_adapter::PhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use datafusion_pruning::PruningPredicate;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::RowFilter;
use parquet::file::metadata::ParquetMetaData;

use crate::ParquetFileMetrics;
use crate::opener::{build_page_pruning_predicate, build_pruning_predicates};
use crate::page_filter::PagePruningAccessPlanFilter;
use crate::projection_read_plan::build_projection_read_plan;
use crate::row_filter::build_row_filter;

/// Everything [`PerFileParquetReadPlanner::plan`] needs for one file, known
/// once its footer has been read.
pub struct PerFileParquetReadInput {
    /// Output expressions over `logical_file_schema`.
    pub projection: ProjectionExprs,
    /// Optional filter over `logical_file_schema`.
    pub filter: Option<Arc<dyn PhysicalExpr>>,
    /// The schema the projection and filter were planned against.
    pub logical_file_schema: SchemaRef,
    /// The Arrow schema of this file.
    pub physical_file_schema: SchemaRef,
    /// The file's Parquet metadata, including its schema descriptor.
    pub metadata: Arc<ParquetMetaData>,
    /// Adapts expressions from the logical to the physical file schema.
    pub expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory>,
    /// Maximum `IN` list size the pruning predicates expand.
    pub max_in_list_size: usize,
    /// Metrics the decoder row filter and pruning predicates record into.
    pub file_metrics: ParquetFileMetrics,
}

/// Why a [`PerFileParquetReadPlan`] reads more than its expressions name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerFileReadFallback {
    /// A function declared nested input fields this file does not carry, so
    /// its inputs are decoded whole and no statistics pruning is applied.
    InvalidInputRequirements,
}

/// The owned read plan for one Parquet file.
pub struct PerFileParquetReadPlan {
    /// The Parquet leaves the decoder reads.
    pub projection_mask: ProjectionMask,
    /// The Arrow schema of the batches the decoder produces under
    /// `projection_mask`; nested types are pruned to the selected leaves.
    pub projected_schema: SchemaRef,
    /// The adapted projection, rebased onto `projected_schema`.
    pub projection: ProjectionExprs,
    /// Filter evaluated while decoding, if any conjunct can be.
    pub row_filter: Option<RowFilter>,
    /// Row-group statistics predicate.
    pub row_group_predicate: Option<Arc<PruningPredicate>>,
    /// Page-index predicate.
    pub page_predicate: Option<Arc<PagePruningAccessPlanFilter>>,
    /// Set when the plan conservatively reads more or prunes less.
    pub fallback: Option<PerFileReadFallback>,
}

impl std::fmt::Debug for PerFileParquetReadPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerFileParquetReadPlan")
            .field("projection_mask", &self.projection_mask)
            .field("projected_schema", &self.projected_schema)
            .field("projection", &self.projection)
            .field("row_filter", &self.row_filter.is_some())
            .field("row_group_predicate", &self.row_group_predicate)
            .field("page_predicate", &self.page_predicate.is_some())
            .field("fallback", &self.fallback)
            .finish()
    }
}

/// Plans how to read one Parquet file. See the module documentation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PerFileParquetReadPlanner;

impl PerFileParquetReadPlanner {
    /// Adapt `input`'s expressions to the file and derive its read plan.
    ///
    /// Invalid nested input requirements are not an error: the affected roots
    /// are decoded whole, both pruning predicates are omitted, and
    /// [`PerFileReadFallback::InvalidInputRequirements`] is reported.
    ///
    /// # Errors
    ///
    /// Returns the adapter, simplifier, or row-filter error raised when an
    /// expression cannot be evaluated against the file schema.
    pub fn plan(input: PerFileParquetReadInput) -> Result<PerFileParquetReadPlan> {
        let PerFileParquetReadInput {
            projection,
            filter,
            logical_file_schema,
            physical_file_schema,
            metadata,
            expr_adapter_factory,
            max_in_list_size,
            file_metrics,
        } = input;
        let rewriter = expr_adapter_factory
            .create(logical_file_schema, Arc::clone(&physical_file_schema))?;
        let simplifier = PhysicalExprSimplifier::new(&physical_file_schema);
        let filter = filter
            .map(|filter| simplifier.simplify(rewriter.rewrite(filter)?))
            .transpose()?;
        let projection = projection
            .try_map_exprs(|expr| simplifier.simplify(rewriter.rewrite(expr)?))?;

        let fallback = projection
            .expr_iter()
            .chain(filter.iter().cloned())
            .any(|expr| has_invalid_requirements(&expr, &physical_file_schema))
            .then_some(PerFileReadFallback::InvalidInputRequirements);

        let read_plan = build_projection_read_plan(
            projection.expr_iter(),
            &physical_file_schema,
            metadata.file_metadata().schema_descr(),
        );
        let projection = projection.try_map_exprs(|expr| {
            reassign_expr_columns(expr, &read_plan.projected_schema)
        })?;
        let row_filter = filter
            .as_ref()
            .map(|filter| {
                build_row_filter(
                    filter,
                    &physical_file_schema,
                    &metadata,
                    true,
                    &file_metrics,
                )
            })
            .transpose()?
            .flatten();
        let pruning_filter = filter.as_ref().filter(|_| fallback.is_none());
        let row_group_predicate = build_pruning_predicates(
            pruning_filter,
            &physical_file_schema,
            &file_metrics.predicate_evaluation_errors,
            max_in_list_size,
        );
        let page_predicate = pruning_filter
            .map(|filter| build_page_pruning_predicate(filter, &physical_file_schema))
            .filter(|predicate| predicate.filter_number() > 0);
        Ok(PerFileParquetReadPlan {
            projection_mask: read_plan.projection_mask,
            projected_schema: read_plan.projected_schema,
            projection,
            row_filter,
            row_group_predicate,
            page_predicate,
            fallback,
        })
    }
}

/// Reports whether any function in `expr` declares nested input fields that
/// do not validate against `schema`.
fn has_invalid_requirements(expr: &Arc<dyn PhysicalExpr>, schema: &SchemaRef) -> bool {
    let mut invalid = false;
    let _ = expr.apply(|node| {
        if let Some(function) = node.downcast_ref::<ScalarFunctionExpr>()
            && declares_requirements(function, schema)
            && function.required_input_fields(schema).is_none()
        {
            invalid = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    invalid
}

/// Reports whether `function` declares any nested input requirement against
/// `schema`, valid or not.
fn declares_requirements(function: &ScalarFunctionExpr, schema: &SchemaRef) -> bool {
    let Ok(fields) = function
        .args()
        .iter()
        .map(|arg| arg.return_field(schema))
        .collect::<Result<Vec<_>>>()
    else {
        return true;
    };
    let literals = function
        .args()
        .iter()
        .map(|arg| {
            arg.downcast_ref::<datafusion_physical_expr::expressions::Literal>()
                .map(|literal| literal.value())
        })
        .collect::<Vec<_>>();
    function
        .fun()
        .required_input_fields(datafusion_expr::ReturnFieldArgs {
            arg_fields: &fields,
            scalar_arguments: &literals,
        })
        .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray, StructArray};
    use arrow::datatypes::{DataType, Field, Fields, Schema};
    use bytes::Bytes;
    use datafusion_common::ScalarValue;
    use datafusion_expr::{
        ColumnarValue, InputFieldRequirement, ReturnFieldArgs, ScalarFunctionArgs,
        ScalarUDF, ScalarUDFImpl, Signature, Volatility,
    };
    use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal};
    use datafusion_physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
    use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::ArrowReaderMetadata;
    use parquet::file::properties::{EnabledStatistics, WriterProperties};

    /// `required_path(s)`: `s IS NULL`, declaring one nested path of `s`.
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct RequiredPath {
        path: Vec<String>,
        signature: Signature,
    }

    impl ScalarUDFImpl for RequiredPath {
        fn name(&self) -> &str {
            "required_path"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, _: &[DataType]) -> Result<DataType> {
            Ok(DataType::Boolean)
        }

        fn required_input_fields(
            &self,
            _: ReturnFieldArgs,
        ) -> Option<Vec<InputFieldRequirement>> {
            Some(vec![InputFieldRequirement {
                arg_index: 0,
                field_paths: vec![self.path.clone()],
            }])
        }

        fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            Ok(ColumnarValue::Array(Arc::new(arrow::compute::is_null(
                args.args[0].to_array(args.number_rows)?.as_ref(),
            )?)))
        }
    }

    /// The `s: {a: Int64, b: Utf8}`, `n: Int64` schema of [`file`].
    fn schema() -> SchemaRef {
        let s = Fields::from(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Utf8, true),
        ]);
        Arc::new(Schema::new(vec![
            Field::new("s", DataType::Struct(s), true),
            Field::new("n", DataType::Int64, true),
        ]))
    }

    /// Two row groups of ten rows, `s.a` and `n` 0..10 then 10..20, with
    /// statistics and a page index.
    fn file() -> Arc<ParquetMetaData> {
        let schema = schema();
        let DataType::Struct(fields) = schema.field(0).data_type().clone() else {
            unreachable!()
        };
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(10))
            .set_statistics_enabled(EnabledStatistics::Page)
            .build();
        let mut buffer = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut buffer, Arc::clone(&schema), Some(properties))
                .unwrap();
        let values = Arc::new(Int64Array::from_iter_values(0..20)) as ArrayRef;
        let labels = Arc::new(StringArray::from_iter_values(
            (0..20).map(|i| format!("label-{i}")),
        )) as ArrayRef;
        let s = StructArray::new(fields, vec![Arc::clone(&values), labels], None);
        writer
            .write(&RecordBatch::try_new(schema, vec![Arc::new(s), values]).unwrap())
            .unwrap();
        writer.close().unwrap();
        let options = parquet::arrow::arrow_reader::ArrowReaderOptions::new()
            .with_page_index_policy(parquet::file::metadata::PageIndexPolicy::Required);
        Arc::clone(
            ArrowReaderMetadata::load(&Bytes::from(buffer), options)
                .unwrap()
                .metadata(),
        )
    }

    /// `get_field(s, name)` over the logical schema.
    fn field(name: &str) -> Arc<dyn PhysicalExpr> {
        let schema = schema();
        Arc::new(
            ScalarFunctionExpr::try_new(
                datafusion_functions::core::get_field(),
                vec![
                    Arc::new(Column::new("s", 0)),
                    Arc::new(Literal::new(ScalarValue::Utf8(Some(name.into())))),
                ],
                &schema,
                Arc::new(datafusion_common::config::ConfigOptions::default()),
            )
            .unwrap(),
        )
    }

    /// `expr > value`.
    fn greater(expr: Arc<dyn PhysicalExpr>, value: i64) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            expr,
            datafusion_expr::Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int64(Some(value)))),
        ))
    }

    /// Plans `projection` and `filter` against [`file`] with the default
    /// adapter.
    fn plan(
        projection: Vec<Arc<dyn PhysicalExpr>>,
        filter: Option<Arc<dyn PhysicalExpr>>,
    ) -> Result<PerFileParquetReadPlan> {
        let metadata = file();
        let physical = Arc::new(
            parquet::arrow::parquet_to_arrow_schema(
                metadata.file_metadata().schema_descr(),
                None,
            )
            .unwrap(),
        );
        PerFileParquetReadPlanner::plan(PerFileParquetReadInput {
            projection: ProjectionExprs::new(projection.into_iter().enumerate().map(
                |(index, expr)| {
                    datafusion_physical_expr::projection::ProjectionExpr::new(
                        expr,
                        format!("c{index}"),
                    )
                },
            )),
            filter,
            logical_file_schema: schema(),
            physical_file_schema: physical,
            metadata,
            expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
            max_in_list_size: 20,
            file_metrics: ParquetFileMetrics::new(
                0,
                "file",
                &ExecutionPlanMetricsSet::new(),
            ),
        })
    }

    /// Row groups the plan's statistics predicate keeps.
    fn kept_row_groups(plan: &PerFileParquetReadPlan) -> Vec<usize> {
        let metadata = file();
        let mut groups = crate::RowGroupAccessPlanFilter::new(
            crate::ParquetAccessPlan::new_all(metadata.num_row_groups()),
        );
        if let Some(predicate) = &plan.row_group_predicate {
            groups.prune_by_statistics(
                &plan_schema(&metadata),
                metadata.file_metadata().schema_descr(),
                metadata.row_groups(),
                predicate,
                &ParquetFileMetrics::new(0, "file", &ExecutionPlanMetricsSet::new()),
            );
        }
        groups.build().row_group_indexes()
    }

    /// The Arrow schema of [`file`].
    fn plan_schema(metadata: &ParquetMetaData) -> SchemaRef {
        Arc::new(
            parquet::arrow::parquet_to_arrow_schema(
                metadata.file_metadata().schema_descr(),
                None,
            )
            .unwrap(),
        )
    }

    #[test]
    fn per_file_plan_covers_projection_filter_and_pruning() {
        let plan = plan(
            vec![field("a")],
            Some(Arc::new(BinaryExpr::new(
                greater(field("a"), 14),
                datafusion_expr::Operator::And,
                greater(Arc::new(Column::new("n", 1)), 14),
            ))),
        )
        .unwrap();
        let metadata = file();
        let leaves = metadata.file_metadata().schema_descr();
        // Projection decodes only `s.a`; the filter's `n` is decoded by the
        // row filter, never by the projection.
        assert_eq!(plan.projection_mask, ProjectionMask::leaves(leaves, [0]));
        assert_eq!(
            plan.projected_schema.field(0).data_type(),
            &DataType::Struct(vec![Field::new("a", DataType::Int64, true)].into())
        );
        assert_eq!(plan.projection.as_ref().len(), 1);
        assert!(plan.row_filter.is_some());
        assert_eq!(kept_row_groups(&plan), vec![1]);
        assert!(plan.page_predicate.is_some());
        assert_eq!(plan.fallback, None);

        // A projection over a column the logical schema does not have is a
        // typed planning error, not a fallback.
        let missing = plan_with_missing_column();
        assert!(missing.is_err());
    }

    /// Plans a projection whose column index is outside the logical schema.
    fn plan_with_missing_column() -> Result<PerFileParquetReadPlan> {
        plan(vec![Arc::new(Column::new("absent", 7))], None)
    }

    #[test]
    fn invalid_requirements_fall_back_without_pruning() {
        let required = |path: &[&str]| -> Arc<dyn PhysicalExpr> {
            let schema = schema();
            Arc::new(
                ScalarFunctionExpr::try_new(
                    Arc::new(ScalarUDF::new_from_impl(RequiredPath {
                        path: path.iter().map(|name| (*name).to_owned()).collect(),
                        signature: Signature::any(1, Volatility::Immutable),
                    })),
                    vec![Arc::new(Column::new("s", 0))],
                    &schema,
                    Arc::new(datafusion_common::config::ConfigOptions::default()),
                )
                .unwrap(),
            )
        };
        let metadata = file();
        let leaves = metadata.file_metadata().schema_descr();
        let filter = || Some(greater(field("a"), 14));

        // A valid declaration decodes only the declared leaf and prunes.
        let valid = plan(vec![required(&["b"])], filter()).unwrap();
        assert_eq!(valid.projection_mask, ProjectionMask::leaves(leaves, [1]));
        assert_eq!(valid.fallback, None);
        assert_eq!(kept_row_groups(&valid), vec![1]);

        // A path the file does not carry decodes the whole root, still
        // filters while decoding, and never prunes.
        let invalid = plan(vec![required(&["missing"])], filter()).unwrap();
        assert_eq!(
            invalid.projection_mask,
            ProjectionMask::leaves(leaves, [0, 1])
        );
        assert_eq!(
            invalid.fallback,
            Some(PerFileReadFallback::InvalidInputRequirements)
        );
        assert!(invalid.row_filter.is_some());
        assert!(invalid.row_group_predicate.is_none());
        assert!(invalid.page_predicate.is_none());
        assert_eq!(kept_row_groups(&invalid), vec![0, 1]);
    }
}
