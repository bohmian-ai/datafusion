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

//! Converting a stored Parquet Variant to a table's Variant type.
//!
//! A table declares a Variant column with one storage type, normally the
//! unshredded `Struct<metadata, value>`. A Parquet file may store the same
//! column shredded (`Struct<metadata, value, typed_value>`). A name-based
//! struct cast would drop `typed_value` and with it every shredded value, so
//! [`cast_column_to_field`](crate::nested_struct::cast_column_to_field) sends
//! a cast whose target field is a Variant here instead.
//!
//! Every byte buffer the conversion reads is validated first: arrow-rs
//! decoding panics on malformed bytes, while a malformed file must be an
//! error.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BinaryViewArray, GenericListArray, OffsetSizeTrait,
    StructArray,
};
use arrow::compute::{CastOptions, cast};
use arrow::datatypes::{DataType, Field};
use parquet_variant::{Variant, VariantMetadata};
use parquet_variant_compute::{VariantArray, unshred_variant};

use crate::error::{_exec_datafusion_err, Result};
use crate::nested_struct::cast_column;

/// Convert a stored Variant array to the storage type of `target`.
///
/// The source is validated, then unshredded with arrow-rs
/// [`unshred_variant`], then each child is cast to the target's child type
/// (for example `BinaryView` to `Binary`).
///
/// # Errors
///
/// Returns an error when the source is not a Variant layout, when any
/// present value's bytes are not a valid Variant, or when the target is not
/// a Variant storage struct.
pub fn cast_to_variant_field(
    source: &ArrayRef,
    target: &Field,
    cast_options: &CastOptions,
) -> Result<ArrayRef> {
    let DataType::Struct(target_fields) = target.data_type() else {
        return Err(_exec_datafusion_err!(
            "Variant field '{}' must be a struct, found {}",
            target.name(),
            target.data_type()
        ));
    };
    let stored = VariantArray::try_new(source.as_ref())?;
    validate(&stored)?;
    let canonical = unshred_variant(&stored)?.into_inner();
    let children = target_fields
        .iter()
        .map(|field| {
            let child = canonical.column_by_name(field.name()).ok_or_else(|| {
                _exec_datafusion_err!(
                    "Variant field '{}' has no child '{}'",
                    target.name(),
                    field.name()
                )
            })?;
            cast_column(child, field.data_type(), cast_options)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(StructArray::try_new(
        target_fields.clone(),
        children,
        canonical.nulls().cloned(),
    )?))
}

/// Validate every metadata and value buffer `stored` will be decoded from.
///
/// Each present row's metadata must parse, and every present residual
/// `value`, at the top level and under `typed_value`, must be a fully valid
/// Variant against its row's metadata. arrow-rs full validation bounds its
/// own recursion depth. Call it before any arrow-rs kernel that decodes
/// values, since those panic on malformed bytes.
///
/// # Errors
///
/// Returns an error naming the first row whose metadata or value bytes are
/// not a valid Variant.
pub fn validate(stored: &VariantArray) -> Result<()> {
    let metadata = cast(stored.metadata_column(), &DataType::BinaryView)?;
    let metadata = metadata.as_binary_view();
    let owners = (0..stored.len())
        .map(|row| stored.is_valid(row).then_some(row))
        .collect::<Vec<_>>();
    for row in owners.iter().flatten() {
        VariantMetadata::try_new(metadata.value(*row))?;
    }
    let residuals = Residuals { metadata };
    residuals.validate_values(stored.value_column(), &owners)?;
    match stored.typed_value_column() {
        Some(typed) => residuals.validate_typed(typed.as_ref(), &owners),
        None => Ok(()),
    }
}

/// The residual `value` buffers of one Variant column, each decoded with the
/// metadata of the row that owns it.
struct Residuals<'a> {
    /// The column's metadata, one per row.
    metadata: &'a BinaryViewArray,
}

impl Residuals<'_> {
    /// Validate each present `value` against its owner row's metadata.
    ///
    /// `owners[i]` is the row that element `i` belongs to, or `None` when it
    /// is unreachable (a null row, or under a null parent).
    fn validate_values(&self, value: &ArrayRef, owners: &[Option<usize>]) -> Result<()> {
        let value = cast(value, &DataType::BinaryView)?;
        let value = value.as_binary_view();
        for (element, owner) in owners.iter().enumerate() {
            if let Some(owner) = *owner
                && value.is_valid(element)
            {
                Variant::try_new(self.metadata.value(owner), value.value(element))
                    .map_err(|error| {
                        _exec_datafusion_err!(
                            "row {owner} is not a valid Variant: {error}"
                        )
                    })?;
            }
        }
        Ok(())
    }

    /// Validate the residuals below one `typed_value`: an object descends
    /// into each shredded field, an array into its elements, and a primitive
    /// holds no residual bytes.
    fn validate_typed(&self, typed: &dyn Array, owners: &[Option<usize>]) -> Result<()> {
        let owners = present(typed, owners);
        match typed.data_type() {
            DataType::Struct(_) => typed
                .as_struct()
                .columns()
                .iter()
                .try_for_each(|field| self.validate_shredded(field.as_ref(), &owners)),
            DataType::List(_) => self.validate_elements(typed.as_list::<i32>(), &owners),
            DataType::LargeList(_) => {
                self.validate_elements(typed.as_list::<i64>(), &owners)
            }
            _ => Ok(()),
        }
    }

    /// Validate one shredded node, `Struct<value?, typed_value?>`.
    fn validate_shredded(
        &self,
        node: &dyn Array,
        owners: &[Option<usize>],
    ) -> Result<()> {
        let Some(node) = node.as_struct_opt() else {
            return Err(_exec_datafusion_err!(
                "shredded Variant field is {}, not a struct",
                node.data_type()
            ));
        };
        let owners = present(node, owners);
        if let Some(value) = node.column_by_name("value") {
            self.validate_values(value, &owners)?;
        }
        match node.column_by_name("typed_value") {
            Some(typed) => self.validate_typed(typed.as_ref(), &owners),
            None => Ok(()),
        }
    }

    /// Validate a shredded array's elements, each owned by its list's row.
    fn validate_elements<O: OffsetSizeTrait>(
        &self,
        list: &GenericListArray<O>,
        owners: &[Option<usize>],
    ) -> Result<()> {
        let mut elements = vec![None; list.values().len()];
        let offsets = list.value_offsets();
        for (row, owner) in owners.iter().enumerate() {
            if owner.is_some() {
                elements[offsets[row].as_usize()..offsets[row + 1].as_usize()]
                    .fill(*owner);
            }
        }
        self.validate_shredded(list.values().as_ref(), &elements)
    }
}

/// `owners` with every element `array` holds null cleared.
fn present(array: &dyn Array, owners: &[Option<usize>]) -> Vec<Option<usize>> {
    owners
        .iter()
        .enumerate()
        .map(|(element, owner)| owner.filter(|_| array.is_valid(element)))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow::array::{BinaryViewArray, Int64Array, ListArray, StringArray};
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::Fields;
    use parquet_variant_compute::{json_to_variant, shred_variant, variant_to_json};

    use super::*;
    use crate::nested_struct::cast_column_to_field;

    /// The canonical (unshredded) Variant field a table declares.
    fn variant_field(name: &str) -> Field {
        Field::new(
            name,
            DataType::Struct(Fields::from(vec![
                Field::new("metadata", DataType::Binary, false),
                Field::new("value", DataType::Binary, false),
            ])),
            true,
        )
        .with_metadata(HashMap::from([
            (
                "ARROW:extension:name".to_owned(),
                "arrow.parquet.variant".to_owned(),
            ),
            ("ARROW:extension:metadata".to_owned(), String::new()),
        ]))
    }

    /// `json` rows encoded as Variants, shredded with `typed_value: as_type`.
    fn shredded(json: Vec<Option<&str>>, as_type: &DataType) -> ArrayRef {
        let text: ArrayRef = Arc::new(StringArray::from(json));
        let variants = json_to_variant(&text).unwrap();
        ArrayRef::from(shred_variant(&variants, as_type).unwrap())
    }

    /// The JSON text of each row of a Variant array.
    fn json(array: &ArrayRef) -> Vec<Option<String>> {
        variant_to_json(array)
            .unwrap()
            .iter()
            .map(|row| row.map(str::to_owned))
            .collect()
    }

    fn cast(source: &ArrayRef, target: &Field) -> Result<ArrayRef> {
        cast_column_to_field(source, target, &CastOptions::default())
    }

    #[test]
    fn shredded_top_level_reads_as_canonical() {
        let rows = vec![Some(r#"{"a":1,"b":"x"}"#), Some(r#"{"a":"s"}"#), None];
        let object =
            DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int64, true)]));
        let source = shredded(rows, &object);
        let target = variant_field("v");

        let read = cast(&source, &target).unwrap();

        assert_eq!(read.data_type(), target.data_type());
        assert_eq!(
            json(&read),
            vec![
                Some(r#"{"a":1,"b":"x"}"#.to_owned()),
                Some(r#"{"a":"s"}"#.to_owned()),
                None
            ]
        );
    }

    #[test]
    fn variant_under_a_null_struct_parent_is_null() {
        let source = shredded(vec![Some("1"), Some("2"), Some("3")], &DataType::Int64);
        let (fields, mut columns, nulls) = source.as_struct().clone().into_parts();
        // A reader may leave an empty placeholder under a null parent.
        columns[0] = Arc::new(BinaryViewArray::from(vec![
            &[1u8, 0, 0][..],
            &[][..],
            &[1u8, 0, 0][..],
        ]));
        let child: ArrayRef =
            Arc::new(StructArray::try_new(fields, columns, nulls).unwrap());
        let child_field = Field::new("v", child.data_type().clone(), true);
        let parent: ArrayRef = Arc::new(StructArray::new(
            Fields::from(vec![child_field]),
            vec![child],
            Some(NullBuffer::from(vec![true, false, true])),
        ));
        let target = Field::new(
            "s",
            DataType::Struct(Fields::from(vec![variant_field("v")])),
            true,
        );

        let read = cast(&parent, &target).unwrap();

        let v = Arc::clone(read.as_struct().column(0));
        assert_eq!(
            json(&v),
            vec![Some("1".to_owned()), None, Some("3".to_owned())]
        );
    }

    #[test]
    fn shredded_list_elements_read_as_canonical() {
        let elements =
            shredded(vec![Some("1"), Some(r#""x""#), Some("3")], &DataType::Int64);
        let list: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("element", elements.data_type().clone(), true)),
            OffsetBuffer::<i32>::from_lengths([2, 1]),
            elements,
            None,
        ));
        let target = Field::new(
            "l",
            DataType::List(Arc::new(variant_field("element"))),
            true,
        );

        let read = cast(&list, &target).unwrap();

        let values = Arc::clone(read.as_list::<i32>().values());
        assert_eq!(
            json(&values),
            vec![
                Some("1".to_owned()),
                Some(r#""x""#.to_owned()),
                Some("3".to_owned())
            ]
        );
    }

    #[test]
    fn malformed_residual_is_an_error_not_a_panic() {
        let empty_metadata = &[1u8, 0, 0][..];
        let source: ArrayRef = Arc::new(StructArray::new(
            Fields::from(vec![
                Field::new("metadata", DataType::BinaryView, false),
                Field::new("value", DataType::BinaryView, true),
                Field::new("typed_value", DataType::Int64, true),
            ]),
            vec![
                Arc::new(BinaryViewArray::from(vec![empty_metadata, empty_metadata])),
                Arc::new(BinaryViewArray::from(vec![
                    Some(&[0xFFu8, 0xFF, 0xFF][..]),
                    None,
                ])),
                Arc::new(Int64Array::from(vec![None, Some(5)])),
            ],
            None,
        ));

        let error = cast(&source, &variant_field("v")).unwrap_err();

        assert!(
            error.to_string().contains("row 0 is not a valid Variant"),
            "{error}"
        );
    }
}
