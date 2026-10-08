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

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/apache/datafusion/19fe44cf2f30cbdd63d4a4f52c74055163c6cc38/docs/logos/standalone_logo/logo_original.svg",
    html_favicon_url = "https://raw.githubusercontent.com/apache/datafusion/19fe44cf2f30cbdd63d4a4f52c74055163c6cc38/docs/logos/standalone_logo/logo_original.svg"
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// Make sure fast / cheap clones on Arc are explicit:
// https://github.com/apache/datafusion/issues/11143
#![deny(clippy::clone_on_ref_ptr)]
#![cfg_attr(test, allow(clippy::needless_pass_by_value))]

//! Parquet Variant functions for [DataFusion].
//!
//! A Variant is the Arrow extension type `arrow.parquet.variant`: a struct of
//! `metadata` and `value` binaries, optionally with a shredded
//! `typed_value`. This crate provides:
//!
//! - `variant_get(v, path...)`: the Variant at a path of object keys and
//!   array indexes, or null when absent;
//! - `variant_get_text(v, path...)`: the same value as text (a string as
//!   itself, any other value as its JSON text, null when absent or null);
//! - `parse_json(text)` / `try_parse_json(text)`: a Variant from JSON text;
//! - `to_json(v)`: the JSON text of a Variant; and
//! - [`planner::VariantFunctionPlanner`], which plans `v -> k` as
//!   `variant_get` and `v ->> k` as `variant_get_text`.
//!
//! Every function reads any stored Variant layout, shredded or not, and
//! returns the canonical unshredded layout. Malformed bytes are an error,
//! never a panic.
//!
//! [DataFusion]: https://crates.io/crates/datafusion

use std::sync::{Arc, LazyLock};

use arrow::array::{ArrayRef, StructArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, FieldRef, Fields};
use datafusion_common::nested_struct::is_variant;
use datafusion_common::{Result, plan_err};
use datafusion_expr::registry::FunctionRegistry;
use datafusion_expr::{ColumnarValue, ScalarUDF};
use parquet_variant_compute::{VariantArray, VariantType, unshred_variant};

pub mod get;
pub mod json;
#[cfg(feature = "sql")]
pub mod planner;

/// `variant_get`: the Variant at a path.
pub static VARIANT_GET: LazyLock<Arc<ScalarUDF>> =
    LazyLock::new(|| Arc::new(ScalarUDF::new_from_impl(get::VariantGet::new(false))));
/// `variant_get_text`: the text at a path.
pub static VARIANT_GET_TEXT: LazyLock<Arc<ScalarUDF>> =
    LazyLock::new(|| Arc::new(ScalarUDF::new_from_impl(get::VariantGet::new(true))));
/// `parse_json`: a Variant from JSON text, raising on invalid JSON.
pub static PARSE_JSON: LazyLock<Arc<ScalarUDF>> =
    LazyLock::new(|| Arc::new(ScalarUDF::new_from_impl(json::ParseJson::new(false))));
/// `try_parse_json`: a Variant from JSON text, null on invalid JSON.
pub static TRY_PARSE_JSON: LazyLock<Arc<ScalarUDF>> =
    LazyLock::new(|| Arc::new(ScalarUDF::new_from_impl(json::ParseJson::new(true))));
/// `to_json`: the JSON text of a Variant.
pub static TO_JSON: LazyLock<Arc<ScalarUDF>> =
    LazyLock::new(|| Arc::new(ScalarUDF::new_from_impl(json::ToJson::new())));

/// Every Variant function, for a session's default function list.
pub fn all_default_variant_functions() -> Vec<Arc<ScalarUDF>> {
    [
        &VARIANT_GET,
        &VARIANT_GET_TEXT,
        &PARSE_JSON,
        &TRY_PARSE_JSON,
        &TO_JSON,
    ]
    .into_iter()
    .map(|udf| Arc::clone(udf))
    .collect()
}

/// Register every Variant function in `registry`.
///
/// # Errors
///
/// Returns the registry's error.
pub fn register_all(registry: &mut dyn FunctionRegistry) -> Result<()> {
    for udf in all_default_variant_functions() {
        registry.register_udf(udf)?;
    }
    Ok(())
}

/// A canonical (unshredded) Variant field: `metadata` and `value` binaries
/// under the `arrow.parquet.variant` extension.
pub fn variant_field(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::Struct(canonical_fields()), nullable)
        .with_extension_type(VariantType)
}

/// The children of the canonical Variant storage struct.
fn canonical_fields() -> Fields {
    Fields::from(vec![
        Field::new("metadata", DataType::Binary, false),
        Field::new("value", DataType::Binary, false),
    ])
}

/// Decode a Variant argument of any stored layout, validated.
///
/// # Errors
///
/// Returns an error when the argument is not a Variant layout or holds
/// malformed bytes.
fn decode(argument: &ColumnarValue, rows: usize) -> Result<VariantArray> {
    let variant = VariantArray::try_new(argument.to_array(rows)?.as_ref())?;
    datafusion_common::variant::validate(&variant)?;
    Ok(variant)
}

/// Rebuild a Variant result in the canonical layout, so a function's
/// declared return field and its output always agree.
///
/// # Errors
///
/// Returns the arrow error raised while unshredding or casting a child.
fn canonical(variant: &VariantArray) -> Result<ArrayRef> {
    let variant = unshred_variant(variant)?;
    let metadata = cast(variant.metadata_column(), &DataType::Binary)?;
    let value = cast(variant.value_column(), &DataType::Binary)?;
    Ok(Arc::new(StructArray::try_new(
        canonical_fields(),
        vec![metadata, value],
        variant.nulls().cloned(),
    )?))
}

/// Require exactly one Variant argument.
///
/// # Errors
///
/// Returns a plan error naming `function` otherwise.
fn require_variant(function: &str, fields: &[FieldRef]) -> Result<()> {
    match fields {
        [field] if is_variant(field) => Ok(()),
        _ => plan_err!("{function} requires one Variant argument"),
    }
}
