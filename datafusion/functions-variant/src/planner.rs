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

//! Plans `v -> k` and `v ->> k` on a Variant operand.

use std::sync::Arc;

use datafusion_common::{DFSchema, Result};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::planner::{ExprPlanner, PlannerResult, RawBinaryExpr};
use datafusion_expr::sqlparser::ast::BinaryOperator;
use datafusion_expr::{Expr, ExprSchemable};

use datafusion_common::nested_struct::is_variant;

use crate::{VARIANT_GET, VARIANT_GET_TEXT};

/// Lowers `->` to `variant_get` and `->>` to `variant_get_text` when the
/// left operand is a Variant.
///
/// A chain `v -> 'a' -> 'b' ->> 'c'` becomes one
/// `variant_get_text(v, 'a', 'b', 'c')`, so the whole literal path is one
/// lookup and one declaration of the leaves it reads. Any other operand is
/// left alone.
#[derive(Debug, Default)]
pub struct VariantFunctionPlanner;

impl ExprPlanner for VariantFunctionPlanner {
    fn plan_binary_op(
        &self,
        expr: RawBinaryExpr,
        schema: &DFSchema,
    ) -> Result<PlannerResult<RawBinaryExpr>> {
        let text = match expr.op {
            BinaryOperator::Arrow => false,
            BinaryOperator::LongArrow => true,
            _ => return Ok(PlannerResult::Original(expr)),
        };
        let (_, field) = expr.left.to_field(schema)?;
        if !is_variant(&field) {
            return Ok(PlannerResult::Original(expr));
        }
        let mut args = match expr.left {
            Expr::ScalarFunction(call) if call.func.name() == VARIANT_GET.name() => {
                call.args
            }
            left => vec![left],
        };
        args.push(expr.right);
        let function = if text {
            &VARIANT_GET_TEXT
        } else {
            &VARIANT_GET
        };
        Ok(PlannerResult::Planned(Expr::ScalarFunction(
            ScalarFunction::new_udf(Arc::clone(function), args),
        )))
    }
}
