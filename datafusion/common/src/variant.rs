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
//! arrow-rs validates every value as it decodes it and returns malformed
//! bytes as an error, so the conversion adds no validation pass of its own.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StructArray};
use arrow::compute::CastOptions;
use arrow::datatypes::{DataType, Field};
use parquet_variant_compute::{VariantArray, unshred_variant};

use crate::error::{_exec_datafusion_err, Result};
use crate::nested_struct::cast_column;

/// Convert a stored Variant array to the storage type of `target`.
///
/// The source is unshredded with arrow-rs [`unshred_variant`], which
/// validates each value it decodes, then each child is cast to the target's
/// child type (for example `BinaryView` to `Binary`).
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow::array::{AsArray, BinaryViewArray, Int64Array, ListArray, StringArray};
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
            matches!(error.find_root(), crate::DataFusionError::ArrowError(..)),
            "{error:?}"
        );
    }
}
