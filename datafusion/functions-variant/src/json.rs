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

//! `parse_json(text)`, `try_parse_json(text)`, and `to_json(v)`.

use std::sync::Arc;

use arrow::array::{Array, AsArray, StringBuilder};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::{Result, exec_err};
use datafusion_expr::{
    ColumnarValue, InputFieldRequirement, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDFImpl, Signature, Volatility,
};
use parquet_variant_compute::{VariantArrayBuilder, unshred_variant};
use parquet_variant_json::{VariantToJson, append_json};

use crate::{canonical, decode, require_variant, variant_field};

/// `parse_json(text)` and `try_parse_json(text)`: a Variant from JSON text.
///
/// `parse_json` raises an error on invalid JSON; `try_parse_json` returns
/// null instead. JSON is parsed with `serde_json`, whose recursion limit
/// bounds nesting.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ParseJson {
    /// One string argument, coerced to `Utf8`.
    signature: Signature,
    /// Whether invalid JSON yields null instead of an error.
    lenient: bool,
}

impl ParseJson {
    /// `try_parse_json` when `lenient`, `parse_json` otherwise.
    pub fn new(lenient: bool) -> Self {
        Self {
            signature: Signature::uniform(1, vec![DataType::Utf8], Volatility::Immutable),
            lenient,
        }
    }
}

impl ScalarUDFImpl for ParseJson {
    fn name(&self) -> &str {
        if self.lenient {
            "try_parse_json"
        } else {
            "parse_json"
        }
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(variant_field("", true).data_type().clone())
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(Arc::new(variant_field(self.name(), true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let array = args.args[0].to_array(args.number_rows)?;
        let texts = array.as_string::<i32>();
        let mut builder = VariantArrayBuilder::new(texts.len());
        for (row, text) in texts.iter().enumerate() {
            let Some(text) = text else {
                builder.append_null();
                continue;
            };
            // Parse before building, so a refused row leaves nothing built.
            match serde_json::from_str(text) {
                Ok(json) => append_json(&json, &mut builder)?,
                Err(_) if self.lenient => builder.append_null(),
                Err(error) => {
                    return exec_err!(
                        "{}: row {row} is not valid JSON: {error}",
                        self.name()
                    );
                }
            }
        }
        Ok(ColumnarValue::Array(canonical(&builder.build())?))
    }
}

/// `to_json(v)`: the JSON text of each Variant; a Variant null renders as
/// `null` and a null row as SQL null.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ToJson {
    /// One Variant argument.
    signature: Signature,
}

impl ToJson {
    /// The one-argument signature; the argument is checked when the return
    /// field is resolved.
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl Default for ToJson {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for ToJson {
    fn name(&self) -> &str {
        "to_json"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        require_variant(self.name(), args.arg_fields)?;
        Ok(Arc::new(Field::new(self.name(), DataType::Utf8, true)))
    }

    /// Reads the whole Variant, in any layout.
    fn required_input_fields(
        &self,
        _args: ReturnFieldArgs,
    ) -> Option<Vec<InputFieldRequirement>> {
        Some(vec![InputFieldRequirement {
            arg_index: 0,
            field_paths: vec![vec![]],
            accepts_any_layout: true,
        }])
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        // Row-wise decoding needs objects unshredded.
        let variant = unshred_variant(&decode(&args.args[0], args.number_rows)?)?;
        let mut text = StringBuilder::with_capacity(variant.len(), 0);
        let mut json = Vec::new();
        for row in 0..variant.len() {
            if variant.is_null(row) {
                text.append_null();
                continue;
            }
            json.clear();
            variant.try_value(row)?.to_json(&mut json)?;
            text.append_value(String::from_utf8_lossy(&json));
        }
        Ok(ColumnarValue::Array(Arc::new(text.finish())))
    }
}
