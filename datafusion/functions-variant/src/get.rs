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

//! `variant_get(v, path...)` and `variant_get_text(v, path...)`.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StringBuilder, make_array};
use arrow::buffer::NullBuffer;
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, FieldRef, Fields};
use datafusion_common::nested_struct::is_variant;
use datafusion_common::{Result, ScalarValue, plan_err};
use datafusion_expr::{
    ColumnarValue, ExpressionPlacement, InputFieldRequirement, ReturnFieldArgs,
    ScalarFunctionArgs, ScalarUDFImpl, Signature, StoredValue, Volatility,
};
use parquet_variant::{Variant, VariantPath, VariantPathElement};
use parquet_variant_compute::{
    GetOptions, VariantArray, VariantArrayBuilder, unshred_variant, variant_get,
};
use parquet_variant_json::VariantToJson;

use crate::{canonical, decode, variant_field};

/// The value at a path of object keys and array indexes.
///
/// `variant_get` returns it as a Variant; `variant_get_text` returns it as
/// text: a string as itself, any other value as its JSON text, and SQL null
/// when the path is absent or holds a Variant null.
///
/// A fully literal path is one arrow-rs [`variant_get`] call over the
/// column; a path with a non-literal element is resolved row by row.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct VariantGet {
    /// Variadic: one Variant followed by one or more path elements.
    signature: Signature,
    /// Whether the result is text (`variant_get_text`).
    text: bool,
}

impl VariantGet {
    /// `variant_get` when `text` is false, `variant_get_text` otherwise.
    pub fn new(text: bool) -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
            text,
        }
    }
}

impl ScalarUDFImpl for VariantGet {
    fn name(&self) -> &str {
        if self.text {
            "variant_get_text"
        } else {
            "variant_get"
        }
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    /// A literal path over a column moves toward the scan, as `get_field`
    /// does for a Struct field, so the scan can read only that path.
    fn placement(&self, args: &[ExpressionPlacement]) -> ExpressionPlacement {
        match args.split_first() {
            Some((
                ExpressionPlacement::Column | ExpressionPlacement::MoveTowardsLeafNodes,
                path,
            )) if !path.is_empty()
                && path
                    .iter()
                    .all(|step| *step == ExpressionPlacement::Literal) =>
            {
                ExpressionPlacement::MoveTowardsLeafNodes
            }
            _ => ExpressionPlacement::KeepInPlace,
        }
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(if self.text {
            DataType::Utf8
        } else {
            variant_field("", true).data_type().clone()
        })
    }

    /// Requires a Variant root and string-key or integer-index path elements.
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [root, path @ ..] = args.arg_fields else {
            return plan_err!("{} requires a Variant and a path", self.name());
        };
        if path.is_empty() || !is_variant(root) {
            return plan_err!("{} requires a Variant and a path", self.name());
        }
        if let Some(bad) = path.iter().find(|field| !is_path_type(field.data_type())) {
            return plan_err!(
                "{} path elements are string keys or integer indexes, not {}",
                self.name(),
                bad.data_type()
            );
        }
        Ok(Arc::new(if self.text {
            Field::new(self.name(), DataType::Utf8, true)
        } else {
            variant_field(self.name(), true)
        }))
    }

    /// Accepts the root in any Variant layout and declares the leaves it
    /// reads.
    ///
    /// Against a root that shreds an object, a literal key path needs the
    /// root `metadata` and, following each key through `typed_value.<key>`,
    /// either the whole shredded subtree of the last key or the residual
    /// `value` of the first level that does not shred the next key: the
    /// standard layout never repeats a shredded field in its residual. An
    /// unshredded root, an array index, or a non-literal key reads the whole
    /// root.
    fn required_input_fields(
        &self,
        args: ReturnFieldArgs,
    ) -> Option<Vec<InputFieldRequirement>> {
        let root = args.arg_fields.first()?;
        let leaf = args
            .scalar_arguments
            .get(1..)?
            .iter()
            .map(|key| key.and_then(|key| key.try_as_str().flatten()))
            .collect::<Option<Vec<_>>>()
            .and_then(|keys| shredded_leaf(root.data_type(), &keys));
        Some(vec![InputFieldRequirement {
            arg_index: 0,
            field_paths: match leaf {
                Some(leaf) => vec![vec!["metadata".to_owned()], leaf],
                None => vec![vec![]],
            },
            accepts_any_layout: true,
        }])
    }

    /// `variant_get_text` over a root that shreds every key of a literal key
    /// path is stored in the last key's `typed_value` wherever its residual
    /// `value` is null, when that field is a string, integer, or Boolean:
    /// casting those to text renders them exactly as this function does.
    fn stored_value(&self, args: ReturnFieldArgs) -> Option<StoredValue> {
        if !self.text {
            return None;
        }
        let keys = args
            .scalar_arguments
            .get(1..)?
            .iter()
            .map(|key| key.and_then(|key| key.try_as_str().flatten()))
            .collect::<Option<Vec<_>>>()?;
        let (last, parents) = keys.split_last()?;
        let mut fields = object_fields(args.arg_fields.first()?.data_type())?;
        let mut path = Vec::new();
        for key in parents {
            let child = fields.find(key)?.1;
            path.extend(["typed_value".to_owned(), (*key).to_owned()]);
            fields = object_fields(child.data_type())?;
        }
        let DataType::Struct(leaf) = fields.find(last)?.1.data_type() else {
            return None;
        };
        leaf.find("value")?;
        let typed = leaf.find("typed_value")?.1.data_type();
        if !typed.is_integer()
            && !matches!(
                typed,
                DataType::Boolean
                    | DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Utf8View
            )
        {
            return None;
        }
        path.extend(["typed_value".to_owned(), (*last).to_owned()]);
        Some(StoredValue {
            arg_index: 0,
            typed_path: [path.as_slice(), &["typed_value".to_owned()]].concat(),
            residual_path: [path.as_slice(), &["value".to_owned()]].concat(),
        })
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let rows = args.number_rows;
        let (root, path) = args
            .args
            .split_first()
            .expect("return_field_from_args requires a root and a path");
        let root = decode(root, rows)?;
        let literal = path
            .iter()
            .map(|step| match step {
                ColumnarValue::Scalar(scalar) => Some(scalar_path_element(scalar)),
                ColumnarValue::Array(_) => None,
            })
            .collect::<Option<Vec<_>>>();
        let found = match literal {
            // A literal null key matches nothing.
            Some(path) if path.iter().any(Option::is_none) => {
                let mut builder = VariantArrayBuilder::new(rows);
                builder.append_nulls(rows);
                builder.build()
            }
            Some(path) => {
                let path = VariantPath::new(path.into_iter().flatten().collect());
                // No requested type keeps a shredded leaf shredded, so a
                // string column renders without row-wise decoding. A key
                // arrow-rs proves missing comes back as an untyped
                // `NullArray`, which is a null Variant in every row.
                let root: ArrayRef = root.into();
                let found = variant_get(&root, GetOptions::new_with_path(path))?;
                if found.data_type() == &DataType::Null {
                    let mut builder = VariantArrayBuilder::new(rows);
                    builder.append_nulls(rows);
                    builder.build()
                } else {
                    VariantArray::try_new(found.as_ref())?
                }
            }
            // Row-wise decoding needs objects unshredded.
            None => per_row_get(&unshred_variant(&root)?, path, rows)?,
        };
        Ok(ColumnarValue::Array(if self.text {
            as_text(&found)?
        } else {
            canonical(&found)?
        }))
    }
}

/// Render each row: a string as itself, any other value as JSON text, and
/// SQL null for a null row or a Variant null. Shredded objects are
/// unshredded first, since row-wise decoding cannot read them.
///
/// # Errors
///
/// Returns the arrow error raised while decoding a row or rendering JSON.
fn as_text(variant: &VariantArray) -> Result<ArrayRef> {
    match shredded_text(variant)? {
        Some(text) => Ok(text),
        None => decoded_text(variant),
    }
}

/// [`as_text`] row by row over the unshredded array.
///
/// # Errors
///
/// Returns the arrow error raised while decoding a row or rendering JSON.
fn decoded_text(variant: &VariantArray) -> Result<ArrayRef> {
    let variant = unshred_variant(variant)?;
    let mut text = StringBuilder::with_capacity(variant.len(), 0);
    let mut json = Vec::new();
    for row in 0..variant.len() {
        if variant.is_null(row) {
            text.append_null();
            continue;
        }
        match variant.try_value(row)? {
            Variant::Null => text.append_null(),
            value => match value.as_string() {
                Some(string) => text.append_value(string),
                None => {
                    json.clear();
                    value.to_json(&mut json)?;
                    text.append_value(String::from_utf8_lossy(&json));
                }
            },
        }
    }
    Ok(Arc::new(text.finish()))
}

/// The text of a Variant whose every value is shredded as a string: its
/// `typed_value` column, null wherever the row or that column is null.
///
/// A string renders as itself, and a row with neither a residual `value` nor
/// a `typed_value` is a missing path, which renders as SQL null, so the
/// column already is the answer [`as_text`] would build row by row. `None`
/// when the array shreds no string or any row keeps a residual `value`,
/// which then needs row-wise decoding.
///
/// # Errors
///
/// Returns the arrow error raised while casting the string column to `Utf8`.
fn shredded_text(variant: &VariantArray) -> Result<Option<ArrayRef>> {
    let state = variant.shredding_state();
    let Some(typed) = state.typed_value_column() else {
        return Ok(None);
    };
    let residual = state.value_column();
    if residual.logical_null_count() != residual.len()
        || !matches!(
            typed.data_type(),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        )
    {
        return Ok(None);
    }
    let text = cast(typed, &DataType::Utf8)?;
    let nulls = NullBuffer::union(variant.inner().nulls(), text.nulls());
    let data = text.to_data().into_builder().nulls(nulls).build()?;
    Ok(Some(make_array(data)))
}

/// Resolve a path with at least one non-literal element, row by row: each
/// row reads its own path from its root, and a null key, negative index, or
/// missing step yields null.
///
/// # Errors
///
/// Returns the arrow error raised while decoding a root or a path column.
fn per_row_get(
    root: &VariantArray,
    path: &[ColumnarValue],
    rows: usize,
) -> Result<VariantArray> {
    let columns = path
        .iter()
        .map(|step| step.to_array(rows))
        .collect::<Result<Vec<_>>>()?;
    let mut builder = VariantArrayBuilder::new(rows);
    for row in 0..rows {
        let steps = columns
            .iter()
            .map(|column| {
                ScalarValue::try_from_array(column, row)
                    .ok()
                    .and_then(|step| scalar_path_element(&step))
            })
            .collect::<Option<Vec<_>>>();
        let root_value = match steps {
            Some(steps) if root.is_valid(row) => {
                Some((root.try_value(row)?, VariantPath::new(steps)))
            }
            _ => None,
        };
        let value = root_value
            .as_ref()
            .and_then(|(value, path)| value.get_path(path));
        match value {
            Some(found) => builder.append_variant(found),
            None => builder.append_null(),
        }
    }
    Ok(builder.build())
}

/// Whether a path element type is a string key or an integer index.
fn is_path_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View | DataType::Null
    ) || data_type.is_integer()
}

/// One literal path element; `None` for SQL null or a negative index.
fn scalar_path_element(scalar: &ScalarValue) -> Option<VariantPathElement<'static>> {
    if scalar.is_null() {
        return None;
    }
    match scalar {
        ScalarValue::Utf8(Some(key))
        | ScalarValue::LargeUtf8(Some(key))
        | ScalarValue::Utf8View(Some(key)) => {
            Some(VariantPathElement::field(key.clone()))
        }
        other => match other.cast_to(&DataType::Int64).ok()? {
            ScalarValue::Int64(Some(index)) => {
                usize::try_from(index).ok().map(VariantPathElement::index)
            }
            _ => None,
        },
    }
}

/// The physical path, inside a shredded Variant of type `storage`, of the
/// subtree that holds the value at object-key path `keys`; `None` when
/// `storage` shreds no object.
///
/// Each key present in its level's shredded object descends into
/// `typed_value.<key>`; the last such key selects its whole subtree. A key
/// that level does not shred can only live in that level's residual `value`.
fn shredded_leaf(storage: &DataType, keys: &[&str]) -> Option<Vec<String>> {
    let mut fields = object_fields(storage)?;
    let mut path = Vec::new();
    for (depth, key) in keys.iter().enumerate() {
        let Some(child) = fields.iter().find(|field| field.name() == *key) else {
            path.push("value".to_owned());
            return Some(path);
        };
        path.extend(["typed_value".to_owned(), (*key).to_owned()]);
        if depth + 1 == keys.len() {
            break;
        }
        let Some(next) = object_fields(child.data_type()) else {
            path.push("value".to_owned());
            return Some(path);
        };
        fields = next;
    }
    Some(path)
}

/// The fields of `storage`'s `typed_value` when it shreds an object.
fn object_fields(storage: &DataType) -> Option<Fields> {
    let DataType::Struct(children) = storage else {
        return None;
    };
    match children.find("typed_value")?.1.data_type() {
        DataType::Struct(fields) => Some(fields.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use parquet_variant_compute::{json_to_variant, shred_variant};

    /// `rows` as JSON objects shredded as `{a: Utf8}`, read at key `a`.
    fn key_a(rows: Vec<Option<&str>>) -> VariantArray {
        let json: ArrayRef = Arc::new(StringArray::from(rows));
        let object = DataType::Struct(vec![Field::new("a", DataType::Utf8, true)].into());
        let shredded = shred_variant(&json_to_variant(&json).unwrap(), &object).unwrap();
        let root: ArrayRef = ArrayRef::from(shredded);
        let options =
            GetOptions::new_with_path(VariantPath::new(vec![VariantPathElement::field(
                "a".to_owned(),
            )]));
        VariantArray::try_new(variant_get(&root, options).unwrap().as_ref()).unwrap()
    }

    /// Every value of a shredded string key comes straight from its column,
    /// with a missing key or null row null, exactly as row-wise rendering.
    #[test]
    fn shredded_strings_render_as_row_wise_text() {
        let found = key_a(vec![
            Some(r#"{"a":"x"}"#),
            Some(r#"{"b":1}"#),
            None,
            Some(r#"{"a":""}"#),
        ]);
        assert!(shredded_text(&found).unwrap().is_some());
        let text = as_text(&found).unwrap();
        assert_eq!(text.as_ref(), decoded_text(&found).unwrap().as_ref());
        let text = text.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            text.iter().collect::<Vec<_>>(),
            vec![Some("x"), None, None, Some("")]
        );
    }

    /// `variant_get` keeps each value's type: an int beside a shredded
    /// string key comes back as a Variant int, not text.
    #[test]
    fn residual_ints_stay_ints() {
        let found = key_a(vec![Some(r#"{"a":"x"}"#), Some(r#"{"a":7}"#)]);
        let out = VariantArray::try_new(canonical(&found).unwrap().as_ref()).unwrap();
        assert_eq!(out.value(0).as_string(), Some("x"));
        assert_eq!(out.value(1).as_int64(), Some(7));
    }

    /// A value kept in the residual, a number or a Variant null, needs
    /// row-wise rendering, which the fast path declines.
    #[test]
    fn residual_values_render_row_wise() {
        let found = key_a(vec![
            Some(r#"{"a":"x"}"#),
            Some(r#"{"a":7}"#),
            Some(r#"{"a":null}"#),
        ]);
        assert!(shredded_text(&found).unwrap().is_none());
        let text = as_text(&found).unwrap();
        let text = text.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            text.iter().collect::<Vec<_>>(),
            vec![Some("x"), Some("7"), None]
        );
    }
}
