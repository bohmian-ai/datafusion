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

//! Declaration of built-in (higher order) functions.
//! This module contains built-in functions' enumeration and metadata.
//!
//! Generally, a function has:
//! * a signature
//! * a return type, that is a function of the incoming argument's types
//! * the computation, that must accept each valid signature
//!
//! * Signature: see `Signature`
//! * Return type: a function `(arg_types) -> return_type`. E.g. for array_transform, ([[f32]], v -> v*2) -> [f32], ([[f32]], v -> v > 3.0) -> [bool].
//!
//! This module also has a set of coercion rules to improve user experience: if an argument i32 is passed
//! to a function that supports f64, it is coerced to f64.

use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::expressions::{LambdaExpr, LambdaVariable, Literal};
use crate::{PhysicalExpr, ScalarFunctionExpr};

use arrow::array::{Array, RecordBatch};
use arrow::datatypes::{DataType, FieldRef, Schema};
use datafusion_common::config::{ConfigEntry, ConfigOptions};
use datafusion_common::datatype::FieldExt;
use datafusion_common::tree_node::{
    Transformed, TreeNode, TreeNodeRecursion, TreeNodeRewriter,
};
use datafusion_common::utils::remove_list_null_values;
use datafusion_common::{
    Result, ScalarValue, exec_err, internal_datafusion_err, internal_err,
    plan_datafusion_err, plan_err,
};
use datafusion_expr::type_coercion::functions::value_fields_with_higher_order_udf;
use datafusion_expr::{
    ColumnarValue, HigherOrderFunctionArgs, HigherOrderReturnFieldArgs, HigherOrderUDF,
    LambdaArgument, LambdaParametersProgress, ValueOrLambda, Volatility, expr_vec_fmt,
};

/// Per-argument classification cached at construction time.
///
/// Walking the wrapped lambda tree and scanning a `Vec<usize>` of lambda
/// positions used to be done on every `evaluate` call. Both costs collapse
/// to a single up-front pass by storing the classification (and the resolved
/// inner [`LambdaExpr`]) here.
enum ArgSlot {
    /// A regular value-producing expression at this position.
    Value,
    /// A lambda position. Stores the inner [`LambdaExpr`] pre-extracted from
    /// any wrapper expressions that may have been introduced via
    /// [`PhysicalExpr::with_new_children`] tree rewrites.
    Lambda(Arc<LambdaExpr>),
}

/// Physical expression of a higher order function
pub struct HigherOrderFunctionExpr {
    /// A shared instance of the higher-order function
    fun: Arc<HigherOrderUDF>,
    /// The name of the higher-order function
    name: String,
    /// List of expressions to feed to the function as arguments
    ///
    /// For example, for `array_transform([2, 3], v -> v != 2)`, this will be:
    ///
    /// ```text
    /// ListExpression [2,3]
    /// LambdaExpression
    ///     parameters: ["v"]
    ///     body:
    ///         BinaryExpression (!=)
    ///             left:
    ///                 LambdaVariableExpression("v", Field::new("", Int32, false))
    ///             right:
    ///                 LiteralExpression(2)
    /// ```
    args: Vec<Arc<dyn PhysicalExpr>>,
    /// Per-arg classification, parallel to `args`. Length always equals
    /// `args.len()`. Lambda variants carry the resolved inner [`LambdaExpr`]
    /// so `evaluate` doesn't walk through wrapper nodes.
    slots: Vec<ArgSlot>,
    /// The output field associated this expression
    ///
    /// For example, for `array_transform([2, 3], v -> v != 2)`, this will be
    /// `Field::new("", DataType::new_list(DataType::Boolean, true), true)`
    return_field: FieldRef,
    /// The config options at execution time
    config_options: Arc<ConfigOptions>,
}

impl Debug for HigherOrderFunctionExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let lambda_positions: Vec<_> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| matches!(slot, ArgSlot::Lambda(_)).then_some(i))
            .collect();
        f.debug_struct("HigherOrderFunctionExpr")
            .field("fun", &"<FUNC>")
            .field("name", &self.name)
            .field("args", &self.args)
            .field("lambda_positions", &lambda_positions)
            .field("return_field", &self.return_field)
            .finish()
    }
}

impl HigherOrderFunctionExpr {
    /// Create a new Higher Order function
    ///
    /// Note that lambda arguments must be present directly in args as [LambdaExpr],
    /// and not as a wrapped child of any arg
    pub fn try_new_with_schema(
        fun: Arc<HigherOrderUDF>,
        args: Vec<Arc<dyn PhysicalExpr>>,
        schema: &Schema,
        config_options: Arc<ConfigOptions>,
    ) -> Result<Self> {
        let name = fun.name().to_string();
        let mut slots = Vec::with_capacity(args.len());
        let arg_fields = args
            .iter()
            .map(|e| match e.downcast_ref::<LambdaExpr>() {
                Some(lambda) => {
                    slots.push(ArgSlot::Lambda(Arc::new(lambda.clone())));
                    Ok(ValueOrLambda::Lambda(lambda.body().return_field(schema)?))
                }
                None => {
                    slots.push(ArgSlot::Value);
                    Ok(ValueOrLambda::Value(e.return_field(schema)?))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // verify that input data types is consistent with function's `HigherOrderTypeSignature`
        value_fields_with_higher_order_udf(&arg_fields, fun.as_ref())?;

        let arguments = args
            .iter()
            .map(|e| e.downcast_ref::<Literal>().map(|literal| literal.value()))
            .collect::<Vec<_>>();

        let ret_args = HigherOrderReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &arguments,
        };

        let return_field = fun.return_field_from_args(ret_args)?;

        Ok(Self {
            fun,
            name,
            args,
            slots,
            return_field,
            config_options,
        })
    }

    /// Get the higher order function implementation
    pub fn fun(&self) -> &HigherOrderUDF {
        self.fun.as_ref()
    }

    /// The name for this expression
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Input arguments
    pub fn args(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.args
    }

    /// Data type produced by this expression
    pub fn return_type(&self) -> &DataType {
        self.return_field.data_type()
    }

    pub fn nullable(&self) -> bool {
        self.return_field.is_nullable()
    }

    pub fn config_options(&self) -> &ConfigOptions {
        &self.config_options
    }

    /// Rebuild this function over `list`, a new expression for the List
    /// argument of its [`ListElementLambda`](datafusion_expr::ListElementLambda),
    /// rebinding that lambda's element parameter to the new element field.
    ///
    /// A [`LambdaVariable`] keeps the field it was planned with, and its
    /// evaluation rejects a batch whose field differs. A reader that decodes
    /// only some fields of the List's elements, or a schema adapter that
    /// moves a conversion of the List into the lambda, changes the element
    /// type the parameter is bound to, so the lambda must be rebound:
    ///
    /// - every reference to the parameter in the lambda body becomes
    ///   `bind(variable)`, where `variable` reads the parameter at the new
    ///   element field; references under a nested lambda that declares a
    ///   parameter of the same name are a different variable and are kept;
    /// - scalar functions with a nested return type are rebuilt so their
    ///   return fields follow their arguments, and nested higher-order
    ///   functions are rebuilt, rebinding their own list element lambdas.
    ///
    /// The body is rewritten in one bottom-up pass in which every node is
    /// visited once and a replacement is never visited again, so `bind` may
    /// return an expression that contains the variable it was given.
    /// `schema` is the schema `list` and the columns the lambda captures are
    /// evaluated against.
    ///
    /// Returns `None` when the function declares no list element lambda.
    ///
    /// # Errors
    ///
    /// Returns an error when `list` is not a List or LargeList, when `bind`
    /// fails, or when a rebuilt function rejects its new argument types.
    pub fn rebind_list_element(
        &self,
        list: Arc<dyn PhysicalExpr>,
        schema: &Schema,
        bind: &dyn Fn(Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>>,
    ) -> Result<Option<Self>> {
        let Some(access) = self.fun.list_element_lambda() else {
            return Ok(None);
        };
        let (DataType::List(element) | DataType::LargeList(element)) =
            list.data_type(schema)?
        else {
            return plan_err!(
                "{} expects a List argument at position {}, got {list}",
                self.name,
                access.list_arg
            );
        };
        let ArgSlot::Lambda(lambda) = &self.slots[access.lambda_arg] else {
            return internal_err!(
                "{} declares a lambda at position {} that is not one",
                self.name,
                access.lambda_arg
            );
        };
        let lambda = match lambda.params().get(access.parameter) {
            Some(name) => {
                let mut rebinder = ParameterRebinder {
                    name,
                    field: element.renamed(name),
                    schema,
                    bind,
                };
                let body = Arc::clone(lambda.body()).rewrite(&mut rebinder)?.data;
                LambdaExpr::try_new(lambda.params().to_vec(), body)?
            }
            // A lambda that declares no element parameter cannot read it.
            None => lambda.as_ref().clone(),
        };

        let mut args = self.args.clone();
        args[access.list_arg] = list;
        args[access.lambda_arg] = Arc::new(lambda);
        Self::try_new_with_schema(
            Arc::clone(&self.fun),
            args,
            schema,
            Arc::clone(&self.config_options),
        )
        .map(Some)
    }

    /// Resolve every lambda's parameter list. Returns an empty `Vec` when
    /// there are no lambdas, avoiding the [`datafusion_expr::HigherOrderUDFImpl::lambda_parameters`]
    /// virtual call entirely.
    fn resolve_lambda_parameters(
        &self,
        fields: &[ValueOrLambda<FieldRef, Option<FieldRef>>],
    ) -> Result<Vec<Vec<FieldRef>>> {
        let num_lambdas = self
            .slots
            .iter()
            .filter(|s| matches!(s, ArgSlot::Lambda(_)))
            .count();
        if num_lambdas == 0 {
            return Ok(Vec::new());
        }
        match self.fun().lambda_parameters(0, fields)? {
            LambdaParametersProgress::Partial(_) => plan_err!(
                "{} lambda_parameters returned a partial result when the return type of all it's lambdas were provided",
                self.name()
            ),
            LambdaParametersProgress::Complete(items) => {
                // functions can support multiple lambdas where some trailing ones are optional,
                // but to simplify the implementor, lambda_parameters returns the parameters of all of them,
                // so we can't do equality check. one example is spark reduce:
                // https://spark.apache.org/docs/latest/api/sql/index.html#reduce
                if items.len() < num_lambdas {
                    return exec_err!(
                        "{} invocation defined {num_lambdas} but lambda_parameters returned only {}",
                        self.name(),
                        items.len()
                    );
                }
                Ok(items)
            }
        }
    }
}

impl fmt::Display for HigherOrderFunctionExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}({})", self.name, expr_vec_fmt!(self.args))
    }
}

impl PartialEq for HigherOrderFunctionExpr {
    fn eq(&self, o: &Self) -> bool {
        if std::ptr::eq(self, o) {
            // The equality implementation is somewhat expensive, so let's short-circuit when possible.
            return true;
        }
        // `slots` is a deterministic function of `fun` and `args`, so it's
        // not part of the comparison.
        let Self {
            fun,
            name,
            args,
            slots: _,
            return_field,
            config_options,
        } = self;
        fun.eq(&o.fun)
            && name.eq(&o.name)
            && args.eq(&o.args)
            && return_field.eq(&o.return_field)
            && (Arc::ptr_eq(config_options, &o.config_options)
                || sorted_config_entries(config_options)
                    == sorted_config_entries(&o.config_options))
    }
}
impl Eq for HigherOrderFunctionExpr {}
impl Hash for HigherOrderFunctionExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let Self {
            fun,
            name,
            args,
            slots: _,
            return_field,
            config_options: _, // expensive to hash, and often equal
        } = self;
        fun.hash(state);
        name.hash(state);
        args.hash(state);
        return_field.hash(state);
    }
}

fn sorted_config_entries(config_options: &ConfigOptions) -> Vec<ConfigEntry> {
    let mut entries = config_options.entries();
    entries.sort_by(|l, r| l.key.cmp(&r.key));
    entries
}

impl PhysicalExpr for HigherOrderFunctionExpr {
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let mut arg_fields = Vec::with_capacity(self.args.len());
        let mut fields = Vec::with_capacity(self.args.len());
        for (arg, slot) in self.args.iter().zip(&self.slots) {
            match slot {
                ArgSlot::Lambda(lambda) => {
                    let field = lambda.body().return_field(batch.schema_ref())?;
                    arg_fields.push(ValueOrLambda::Lambda(Arc::clone(&field)));
                    fields.push(ValueOrLambda::Lambda(Some(field)));
                }
                ArgSlot::Value => {
                    let field = arg.return_field(batch.schema_ref())?;
                    arg_fields.push(ValueOrLambda::Value(Arc::clone(&field)));
                    fields.push(ValueOrLambda::Value(field));
                }
            }
        }

        let mut lambda_parameters = self.resolve_lambda_parameters(&fields)?.into_iter();

        let args = self
            .args
            .iter()
            .zip(&self.slots)
            .map(|(arg, slot)| match slot {
                ArgSlot::Lambda(lambda) => {
                    let lambda_params = lambda_parameters.next().ok_or_else(|| {
                        internal_datafusion_err!(
                            "params len should have been checked above"
                        )
                    })?;

                    if lambda.params().len() > lambda_params.len() {
                        return exec_err!(
                            "lambda defined {} params but higher-order function support only {}",
                            lambda.params().len(),
                            lambda_params.len()
                        );
                    }

                    let params = std::iter::zip(lambda.params(), lambda_params)
                        .map(|(name, param)| param.renamed(name.as_str()))
                        .collect();

                    // lambda.projection may include indexes of nested lambda variables not present on this batch
                    let projection = lambda
                        .projection()
                        .iter()
                        .copied()
                        .filter(|i| *i < batch.num_columns())
                        .collect::<Vec<_>>();

                    Ok(ValueOrLambda::Lambda(LambdaArgument::new(
                        params,
                        Arc::clone(lambda.projected_body()),
                        if projection.is_empty() {
                            None
                        } else {
                            Some(batch.project(&projection)?)
                        },
                        lambda.used_param_indices(),
                    )))
                }
                ArgSlot::Value => {
                    let value = arg.evaluate(batch)?;

                    let value = if self.fun.clear_null_values()
                        && matches!(
                            value.data_type(),
                            DataType::List(_) | DataType::LargeList(_)
                        )
                    {
                        let arr = value.into_array(batch.num_rows())?;
                        if arr.null_count() == 0 {
                            ColumnarValue::Array(arr)
                        } else {
                            ColumnarValue::Array(remove_list_null_values(&arr)?)
                        }
                    } else {
                        value
                    };

                    Ok(ValueOrLambda::Value(value))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let input_empty = args.is_empty();
        let input_all_scalar = args
            .iter()
            .all(|arg| matches!(arg, ValueOrLambda::Value(ColumnarValue::Scalar(_))));

        // evaluate the function
        let output = self.fun.invoke_with_args(HigherOrderFunctionArgs {
            args,
            arg_fields,
            number_rows: batch.num_rows(),
            return_field: Arc::clone(&self.return_field),
            config_options: Arc::clone(&self.config_options),
        })?;

        if let ColumnarValue::Array(array) = &output
            && array.len() != batch.num_rows()
        {
            // If the arguments are a non-empty slice of scalar values, we can assume that
            // returning a one-element array is equivalent to returning a scalar.
            let preserve_scalar = array.len() == 1 && !input_empty && input_all_scalar;
            return if preserve_scalar {
                ScalarValue::try_from_array(array, 0).map(ColumnarValue::Scalar)
            } else {
                internal_err!(
                    "higher-order function {} returned a different number of rows than expected. Expected: {}, Got: {}",
                    self.name,
                    batch.num_rows(),
                    array.len()
                )
            };
        }
        Ok(output)
    }

    fn return_field(&self, _input_schema: &Schema) -> Result<FieldRef> {
        Ok(Arc::clone(&self.return_field))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.args.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if children.len() != self.args.len() {
            return internal_err!(
                "HigherOrderFunctionExpr expects exactly {} child, got {}",
                self.args.len(),
                children.len()
            );
        }

        // Re-derive `slots` for the new children using the original slot kinds
        // as the source of truth for which positions must (still) be lambdas.
        let mut new_slots = Vec::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            match &self.slots[i] {
                ArgSlot::Lambda(_) => {
                    let lambda = wrapped_lambda(child).ok_or_else(|| {
                        plan_datafusion_err!(
                            "{} unable to unwrap lambda from {} at position {i}",
                            &children[i],
                            self.name()
                        )
                    })?;
                    new_slots.push(ArgSlot::Lambda(Arc::new(lambda.clone())));
                }
                ArgSlot::Value => {
                    if child.is::<LambdaExpr>() {
                        return plan_err!(
                            "{} received a lambda via with_new_children at position {i} that wasn't a lambda before",
                            self.name()
                        );
                    }
                    new_slots.push(ArgSlot::Value);
                }
            }
        }

        Ok(Arc::new(HigherOrderFunctionExpr {
            name: self.name.clone(),
            fun: Arc::clone(&self.fun),
            args: children,
            slots: new_slots,
            return_field: Arc::clone(&self.return_field),
            config_options: Arc::clone(&self.config_options),
        }))
    }

    fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.name)?;
        for (i, expr) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            expr.fmt_sql(f)?;
        }
        write!(f, ")")
    }

    fn is_volatile_node(&self) -> bool {
        self.fun.signature().volatility == Volatility::Volatile
    }
}

/// Rewrites a lambda body for [`HigherOrderFunctionExpr::rebind_list_element`].
///
/// `f_down` never changes the tree; it only skips the subtree of a nested
/// lambda that shadows the parameter. `f_up` sees each node once, after its
/// children, and its replacements are not visited again, so the rewrite
/// terminates whatever `bind` returns.
struct ParameterRebinder<'a> {
    /// The parameter's name, which every [`LambdaVariable`] reading it carries.
    name: &'a str,
    /// The parameter's new field, already renamed to `name`.
    field: FieldRef,
    /// The schema captured columns are evaluated against.
    schema: &'a Schema,
    /// Builds the expression that replaces each reference to the parameter.
    bind: &'a dyn Fn(Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>>,
}

impl TreeNodeRewriter for ParameterRebinder<'_> {
    type Node = Arc<dyn PhysicalExpr>;

    /// Skip a nested lambda that declares a parameter named like this one:
    /// references below it read that parameter instead.
    fn f_down(&mut self, node: Self::Node) -> Result<Transformed<Self::Node>> {
        let shadows = node
            .downcast_ref::<LambdaExpr>()
            .is_some_and(|lambda| lambda.params().iter().any(|p| p == self.name));
        Ok(if shadows {
            Transformed::new(node, false, TreeNodeRecursion::Jump)
        } else {
            Transformed::no(node)
        })
    }

    /// Rebind a reference to the parameter, or rebuild a function so its
    /// return field follows its (possibly rebound) arguments.
    fn f_up(&mut self, node: Self::Node) -> Result<Transformed<Self::Node>> {
        if let Some(variable) = node.downcast_ref::<LambdaVariable>()
            && variable.name() == self.name
        {
            let rebound = LambdaVariable::new(variable.index(), Arc::clone(&self.field));
            return (self.bind)(Arc::new(rebound)).map(Transformed::yes);
        }
        if let Some(function) = node.downcast_ref::<ScalarFunctionExpr>()
            && function.return_type().is_nested()
        {
            let rebuilt = ScalarFunctionExpr::try_new(
                Arc::new(function.fun().clone()),
                function.args().to_vec(),
                self.schema,
                Arc::new(function.config_options().clone()),
            )?;
            if rebuilt.return_field(self.schema)? == function.return_field(self.schema)? {
                return Ok(Transformed::no(node));
            }
            return Ok(Transformed::yes(Arc::new(rebuilt)));
        }
        if let Some(function) = node.downcast_ref::<HigherOrderFunctionExpr>() {
            let rebuilt = match function.fun.list_element_lambda() {
                Some(access) => function.rebind_list_element(
                    Arc::clone(&function.args[access.list_arg]),
                    self.schema,
                    &Ok,
                )?,
                None => None,
            };
            let rebuilt = match rebuilt {
                Some(rebuilt) => rebuilt,
                None => HigherOrderFunctionExpr::try_new_with_schema(
                    Arc::clone(&function.fun),
                    function.args.clone(),
                    self.schema,
                    Arc::clone(&function.config_options),
                )?,
            };
            if rebuilt == *function {
                return Ok(Transformed::no(node));
            }
            return Ok(Transformed::yes(Arc::new(rebuilt)));
        }
        Ok(Transformed::no(node))
    }
}

fn wrapped_lambda(expr: &Arc<dyn PhysicalExpr>) -> Option<&LambdaExpr> {
    let mut current = expr;

    loop {
        if let Some(lambda) = current.downcast_ref::<LambdaExpr>() {
            return Some(lambda);
        } else if current.is::<HigherOrderFunctionExpr>() {
            return None;
        }

        match current.children().as_slice() {
            [single_child] => current = *single_child,
            _ => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::HigherOrderFunctionExpr;
    use crate::create_physical_expr;
    use crate::expressions::CastExpr;
    use crate::expressions::Column;
    use crate::expressions::NoOp;
    use crate::expressions::lambda;
    use crate::expressions::not;
    use arrow::array::RecordBatchOptions;
    use arrow::array::{
        ArrayRef, AsArray, Int32Array, ListArray, StringArray, StructArray,
    };
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{DataType, Field, Fields, Schema};
    use datafusion_common::Result;
    use datafusion_common::assert_contains;
    use datafusion_expr::execution_props::ExecutionProps;
    use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
    use datafusion_expr::{
        HigherOrderFunctionArgs, HigherOrderSignature, HigherOrderUDF,
        HigherOrderUDFImpl, ListElementLambda,
    };
    use datafusion_expr_common::columnar_value::ColumnarValue;
    use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
    use datafusion_physical_expr_common::physical_expr::is_volatile;

    /// Test helper to create a mock UDF with a specific volatility
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct MockHigherOrderUDF {
        signature: HigherOrderSignature,
    }

    impl HigherOrderUDFImpl for MockHigherOrderUDF {
        fn name(&self) -> &str {
            "mock_function"
        }

        fn signature(&self) -> &HigherOrderSignature {
            &self.signature
        }

        fn lambda_parameters(
            &self,
            _step: usize,
            _fields: &[ValueOrLambda<FieldRef, Option<FieldRef>>],
        ) -> Result<LambdaParametersProgress> {
            // Offer two params; single-param lambdas just ignore the second.
            Ok(LambdaParametersProgress::Complete(vec![vec![
                Arc::new(Field::new("", DataType::Int32, true)),
                Arc::new(Field::new("", DataType::Int32, true)),
            ]]))
        }

        fn return_field_from_args(
            &self,
            args: HigherOrderReturnFieldArgs,
        ) -> Result<FieldRef> {
            match &args.arg_fields[0] {
                ValueOrLambda::Lambda(field) | ValueOrLambda::Value(field) => {
                    Ok(Arc::clone(field))
                }
            }
        }

        fn invoke_with_args(
            &self,
            args: HigherOrderFunctionArgs,
        ) -> Result<ColumnarValue> {
            match &args.args[0] {
                ValueOrLambda::Lambda(lambda) => lambda.evaluate(
                    &[
                        // Sentinel for the first param, distinct from the second's value.
                        &|| {
                            Ok(Arc::new(Int32Array::from(vec![-1000; args.number_rows]))
                                as ArrayRef)
                        },
                        &|| {
                            Ok(Arc::new(Int32Array::from_iter_values(
                                (0..args.number_rows as i32).map(|i| 10 * (i + 1)),
                            )) as ArrayRef)
                        },
                    ],
                    |arrays| Ok(arrays.to_vec()),
                ),
                ValueOrLambda::Value(value) => Ok(value.clone()),
            }
        }
    }

    #[test]
    fn test_higher_order_function_volatile_node() {
        // Create a volatile UDF
        let volatile_udf = Arc::new(HigherOrderUDF::new_from_impl(MockHigherOrderUDF {
            signature: HigherOrderSignature::variadic_any(Volatility::Volatile),
        }));

        // Create a non-volatile UDF
        let stable_udf = Arc::new(HigherOrderUDF::new_from_impl(MockHigherOrderUDF {
            signature: HigherOrderSignature::variadic_any(Volatility::Stable),
        }));

        let schema = Schema::new(vec![Field::new("a", DataType::Float32, false)]);
        let args = vec![Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>];
        let config_options = Arc::new(ConfigOptions::new());

        // Test volatile function
        let volatile_expr = HigherOrderFunctionExpr::try_new_with_schema(
            volatile_udf,
            args.clone(),
            &schema,
            Arc::clone(&config_options),
        )
        .unwrap();

        assert!(volatile_expr.is_volatile_node());
        let volatile_arc: Arc<dyn PhysicalExpr> = Arc::new(volatile_expr);
        assert!(is_volatile(&volatile_arc));

        // Test non-volatile function
        let stable_expr = HigherOrderFunctionExpr::try_new_with_schema(
            stable_udf,
            args,
            &schema,
            config_options,
        )
        .unwrap();

        assert!(!stable_expr.is_volatile_node());
        let stable_arc: Arc<dyn PhysicalExpr> = Arc::new(stable_expr);
        assert!(!is_volatile(&stable_arc));
    }

    #[test]
    fn test_higher_order_function_wrapped_lambda() {
        let fun = Arc::new(HigherOrderUDF::new_from_impl(MockHigherOrderUDF {
            signature: HigherOrderSignature::variadic_any(Volatility::Stable),
        }));

        let expected = ScalarValue::Int32(Some(42));

        let hof = HigherOrderFunctionExpr::try_new_with_schema(
            fun,
            vec![lambda(["a"], Arc::new(Literal::new(expected.clone()))).unwrap()],
            &Schema::empty(),
            Arc::new(ConfigOptions::new()),
        )
        .unwrap();

        let new_children = vec![not(Arc::clone(&hof.args[0])).unwrap()];
        let wrapped = Arc::new(hof).with_new_children(new_children).unwrap();

        let result = wrapped
            .evaluate(
                &RecordBatch::try_new_with_options(
                    Arc::new(Schema::empty()),
                    vec![],
                    &RecordBatchOptions::new().with_row_count(Some(0)),
                )
                .unwrap(),
            )
            .unwrap();

        let ColumnarValue::Scalar(result) = result else {
            unreachable!()
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn test_higher_order_function_badly_wrapped_lambda() {
        let fun = Arc::new(HigherOrderUDF::new_from_impl(MockHigherOrderUDF {
            signature: HigherOrderSignature::variadic_any(Volatility::Stable),
        }));

        let hof = HigherOrderFunctionExpr::try_new_with_schema(
            fun,
            vec![
                not(
                    lambda(["a"], Arc::new(Literal::new(ScalarValue::Int32(Some(42)))))
                        .unwrap(),
                )
                .unwrap(),
            ],
            &Schema::empty(),
            Arc::new(ConfigOptions::new()),
        )
        .unwrap();

        let result = hof
            .evaluate(
                &RecordBatch::try_new_with_options(
                    Arc::new(Schema::empty()),
                    vec![],
                    &RecordBatchOptions::new().with_row_count(Some(0)),
                )
                .unwrap(),
            )
            .unwrap_err();

        assert_contains!(
            result.to_string(),
            "LambdaExpr::evaluate() should not be called"
        );
    }

    #[test]
    fn test_higher_order_function_unexpected_lambda() {
        let fun = Arc::new(HigherOrderUDF::new_from_impl(MockHigherOrderUDF {
            signature: HigherOrderSignature::variadic_any(Volatility::Stable),
        }));

        let hof = HigherOrderFunctionExpr::try_new_with_schema(
            fun,
            vec![Arc::new(NoOp::new())],
            &Schema::empty(),
            Arc::new(ConfigOptions::new()),
        )
        .unwrap();

        let result = Arc::new(hof)
            .with_new_children(vec![lambda(["a"], Arc::new(NoOp::new())).unwrap()])
            .unwrap_err();

        assert_contains!(
            result.to_string(),
            "mock_function received a lambda via with_new_children at position 0 that wasn't a lambda before"
        );
    }

    /// Exercises the real planner end to end (not hand-picked indices) to
    /// check the "captures before own-params" layout invariant.
    #[test]
    fn test_higher_order_function_two_lambda_params_capture_and_unused_param() {
        use datafusion_common::DFSchema;
        use datafusion_expr::expr::{HigherOrderFunction, LambdaVariable};
        use datafusion_expr::{Expr, col, lambda as logical_lambda};

        let fun = Arc::new(HigherOrderUDF::new_from_impl(MockHigherOrderUDF {
            signature: HigherOrderSignature::variadic_any(Volatility::Stable),
        }));

        // Body uses capture "a" and param "v"; param "k" is left unused.
        let v = Expr::LambdaVariable(LambdaVariable::new(
            "v".to_string(),
            Some(Arc::new(Field::new("v", DataType::Int32, true))),
        ));
        let body = col("a") + v;
        let lambda_expr = logical_lambda(["k", "v"], body);

        let schema = DFSchema::from_unqualified_fields(
            vec![Field::new("a", DataType::Int32, false)].into(),
            std::collections::HashMap::new(),
        )
        .unwrap();

        let physical_expr = create_physical_expr(
            &Expr::HigherOrderFunction(HigherOrderFunction::new(fun, vec![lambda_expr])),
            &schema,
            &ExecutionProps::new(),
            &PhysicalPlanningContext::default(),
        )
        .unwrap();

        let batch = RecordBatch::try_new(
            Arc::clone(schema.inner()),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef],
        )
        .unwrap();

        let result = physical_expr.evaluate(&batch).unwrap();
        let ColumnarValue::Array(result) = result else {
            unreachable!()
        };

        // a + v; k's sentinel (-1000) must not leak into the result.
        let expected = Int32Array::from(vec![11, 22, 33]);
        assert_eq!(result.as_ref(), &expected);
    }

    /// A list function whose lambda maps each element, like
    /// `array_transform`: it declares a [`ListElementLambda`].
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct MockListTransform {
        signature: HigherOrderSignature,
    }

    impl HigherOrderUDFImpl for MockListTransform {
        fn name(&self) -> &str {
            "mock_transform"
        }

        fn signature(&self) -> &HigherOrderSignature {
            &self.signature
        }

        fn lambda_parameters(
            &self,
            _step: usize,
            fields: &[ValueOrLambda<FieldRef, Option<FieldRef>>],
        ) -> Result<LambdaParametersProgress> {
            let ValueOrLambda::Value(list) = &fields[0] else {
                return plan_err!("expected a list");
            };
            let DataType::List(element) = list.data_type() else {
                return plan_err!("expected a list");
            };
            Ok(LambdaParametersProgress::Complete(vec![vec![Arc::clone(
                element,
            )]]))
        }

        fn list_element_lambda(&self) -> Option<ListElementLambda> {
            Some(ListElementLambda {
                list_arg: 0,
                lambda_arg: 1,
                parameter: 0,
            })
        }

        fn return_field_from_args(
            &self,
            args: HigherOrderReturnFieldArgs,
        ) -> Result<FieldRef> {
            let ValueOrLambda::Lambda(lambda) = &args.arg_fields[1] else {
                return plan_err!("expected a lambda");
            };
            Ok(Arc::new(Field::new(
                "",
                DataType::new_list(lambda.data_type().clone(), true),
                true,
            )))
        }

        fn invoke_with_args(
            &self,
            args: HigherOrderFunctionArgs,
        ) -> Result<ColumnarValue> {
            let (ValueOrLambda::Value(list), ValueOrLambda::Lambda(lambda)) =
                (&args.args[0], &args.args[1])
            else {
                return plan_err!("expected a list and a lambda");
            };
            let list = list.to_array(args.number_rows)?;
            let list = list.as_list::<i32>();
            let values = Arc::clone(list.values());
            let mapped = lambda
                .evaluate(&[&|| Ok(Arc::clone(&values))], |arrays| Ok(arrays.to_vec()))?
                .into_array(values.len())?;
            let DataType::List(field) = args.return_field.data_type() else {
                return plan_err!("expected a list");
            };
            Ok(ColumnarValue::Array(Arc::new(ListArray::new(
                Arc::clone(field),
                list.offsets().clone(),
                mapped,
                list.nulls().cloned(),
            ))))
        }
    }

    /// Element fields named by `names`, from `f: Int32`, `g: Utf8` and
    /// `items: List<Struct>` whose item fields, from `a: Int32` and
    /// `b: Utf8`, are named by `item_names`.
    fn element_fields(names: &[&str], item_names: &[&str]) -> Fields {
        names
            .iter()
            .map(|name| match *name {
                "f" | "a" => Field::new(*name, DataType::Int32, true),
                "g" | "b" => Field::new(*name, DataType::Utf8, true),
                "items" => Field::new(
                    "items",
                    DataType::new_list(
                        DataType::Struct(element_fields(item_names, &[])),
                        true,
                    ),
                    true,
                ),
                other => unreachable!("no test field {other}"),
            })
            .collect()
    }

    /// Three Structs with `fields`, each `items` holding one item.
    fn struct_array(fields: &Fields) -> ArrayRef {
        let columns = fields
            .iter()
            .map(|field| -> ArrayRef {
                match field.data_type() {
                    DataType::Int32 if field.name() == "f" => {
                        Arc::new(Int32Array::from(vec![1, 2, 3]))
                    }
                    DataType::Int32 => Arc::new(Int32Array::from(vec![10, 20, 30])),
                    DataType::Utf8 => Arc::new(StringArray::from(vec!["p", "q", "r"])),
                    DataType::List(item) => {
                        let DataType::Struct(item_fields) = item.data_type() else {
                            unreachable!("items hold Structs")
                        };
                        Arc::new(ListArray::new(
                            Arc::clone(item),
                            OffsetBuffer::from_lengths([1, 1, 1]),
                            struct_array(item_fields),
                            None,
                        ))
                    }
                    other => unreachable!("no test type {other}"),
                }
            })
            .collect();
        Arc::new(StructArray::new(fields.clone(), columns, None))
    }

    /// One row whose List `l` holds the three Structs of [`struct_array`].
    fn list_batch(fields: &Fields) -> RecordBatch {
        let element = Arc::new(Field::new_list_field(
            DataType::Struct(fields.clone()),
            true,
        ));
        let l = ListArray::new(
            Arc::clone(&element),
            OffsetBuffer::from_lengths([3]),
            struct_array(fields),
            None,
        );
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "l",
                DataType::List(element),
                true,
            )])),
            vec![Arc::new(l)],
        )
        .unwrap()
    }

    /// `get_field(source, name)` planned over `schema`.
    fn field_of(
        source: Arc<dyn PhysicalExpr>,
        name: &str,
        schema: &Schema,
    ) -> Arc<dyn PhysicalExpr> {
        Arc::new(
            ScalarFunctionExpr::try_new(
                datafusion_functions::core::get_field(),
                vec![source, Arc::new(Literal::new(ScalarValue::from(name)))],
                schema,
                Arc::new(ConfigOptions::new()),
            )
            .unwrap(),
        )
    }

    /// `fun(list, (param) -> body(param))` planned over `schema`, with the
    /// parameter at `index` in the lambda's planning schema.
    fn with_lambda(
        fun: Arc<HigherOrderUDF>,
        list: Arc<dyn PhysicalExpr>,
        params: &[&str],
        index: usize,
        param_field: FieldRef,
        body: impl FnOnce(Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> HigherOrderFunctionExpr {
        let variable =
            Arc::new(LambdaVariable::new(index, param_field.renamed(params[0])));
        let lambda = LambdaExpr::try_new(
            params.iter().map(|p| p.to_string()).collect(),
            body(variable),
        )
        .unwrap();
        let args = if fun.list_element_lambda().is_some() {
            vec![list, Arc::new(lambda) as Arc<dyn PhysicalExpr>]
        } else {
            vec![Arc::new(lambda) as Arc<dyn PhysicalExpr>]
        };
        HigherOrderFunctionExpr::try_new_with_schema(
            fun,
            args,
            schema,
            Arc::new(ConfigOptions::new()),
        )
        .unwrap()
    }

    /// `mock_transform(list, (param) -> body(param))` over `schema`.
    fn transform(
        list: Arc<dyn PhysicalExpr>,
        param: &str,
        index: usize,
        body: impl FnOnce(Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> Arc<dyn PhysicalExpr> {
        let DataType::List(element) = list.data_type(schema).unwrap() else {
            unreachable!("transform takes a List")
        };
        let fun = Arc::new(HigherOrderUDF::new_from_impl(MockListTransform {
            signature: HigherOrderSignature::variadic_any(Volatility::Immutable),
        }));
        Arc::new(with_lambda(
            fun,
            list,
            &[param],
            index,
            element,
            body,
            schema,
        ))
    }

    /// Evaluate `expr` over `batch` to an array.
    fn evaluate(expr: &dyn PhysicalExpr, batch: &RecordBatch) -> Result<ArrayRef> {
        expr.evaluate(batch)?.into_array(batch.num_rows())
    }

    /// Rebind `expr`, a `mock_transform`, over `l` in `batch`.
    fn rebind(
        expr: &Arc<dyn PhysicalExpr>,
        batch: &RecordBatch,
        bind: &dyn Fn(Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>>,
    ) -> Result<HigherOrderFunctionExpr> {
        let function = expr.downcast_ref::<HigherOrderFunctionExpr>().unwrap();
        Ok(function
            .rebind_list_element(Arc::new(Column::new("l", 0)), batch.schema_ref(), bind)?
            .expect("mock_transform declares a list element lambda"))
    }

    /// Over a List narrowed to some element fields, the planned function is
    /// rejected; rebinding rebinds its parameter and a nested list element
    /// lambda's, and evaluates to what the planned function did over the
    /// whole List.
    #[test]
    fn rebind_list_element_follows_a_narrowed_list() -> Result<()> {
        let full = list_batch(&element_fields(&["f", "g", "items"], &["a", "b"]));
        let narrowed = list_batch(&element_fields(&["f", "items"], &["a"]));
        let schema = full.schema();
        // mock_transform(l, x -> mock_transform(x['items'], y -> y['a']))
        let planned = transform(
            Arc::new(Column::new("l", 0)),
            "x",
            1,
            |x| {
                transform(
                    field_of(x, "items", &schema),
                    "y",
                    2,
                    |y| field_of(y, "a", &schema),
                    &schema,
                )
            },
            &schema,
        );

        assert_contains!(
            evaluate(planned.as_ref(), &narrowed)
                .unwrap_err()
                .to_string(),
            "doesn't match batch field"
        );
        let rebound = rebind(&planned, &narrowed, &Ok)?;
        assert_eq!(
            evaluate(&rebound, &narrowed)?.as_ref(),
            evaluate(planned.as_ref(), &full)?.as_ref()
        );
        Ok(())
    }

    /// `bind` may return an expression containing the variable it was given:
    /// each reference is replaced once.
    #[test]
    fn rebind_list_element_binds_each_reference_once() -> Result<()> {
        let full = list_batch(&element_fields(&["f", "g", "items"], &["a", "b"]));
        let narrowed = list_batch(&element_fields(&["f"], &[]));
        let schema = full.schema();
        // mock_transform(l, x -> x['f'])
        let planned = transform(
            Arc::new(Column::new("l", 0)),
            "x",
            1,
            |x| field_of(x, "f", &schema),
            &schema,
        );
        let element_type =
            DataType::Struct(element_fields(&["f", "g", "items"], &["a", "b"]));

        let rebound = rebind(&planned, &narrowed, &|x| {
            Ok(Arc::new(CastExpr::new(x, element_type.clone(), None)))
        })?;

        let ArgSlot::Lambda(lambda) = &rebound.slots[1] else {
            unreachable!("argument 1 is the lambda")
        };
        let mut casts = 0;
        let mut variables = 0;
        lambda.body().apply(|node| {
            casts += usize::from(node.is::<CastExpr>());
            variables += usize::from(node.is::<LambdaVariable>());
            Ok(TreeNodeRecursion::Continue)
        })?;
        assert_eq!((casts, variables), (1, 1));
        assert_eq!(
            evaluate(&rebound, &narrowed)?.as_ref(),
            evaluate(planned.as_ref(), &full)?.as_ref()
        );
        Ok(())
    }

    /// A nested lambda declaring the parameter's name reads its own
    /// parameter, which keeps its field.
    #[test]
    fn rebind_list_element_keeps_shadowing_parameters() -> Result<()> {
        let full = list_batch(&element_fields(&["f", "g"], &[]));
        let narrowed = list_batch(&element_fields(&["f"], &[]));
        let schema = full.schema();
        let inner = Arc::new(HigherOrderUDF::new_from_impl(MockHigherOrderUDF {
            signature: HigherOrderSignature::variadic_any(Volatility::Immutable),
        }));
        let int32 = Arc::new(Field::new("", DataType::Int32, true));
        // mock_transform(l, x -> mock_function((x, v) -> x)): the inner `x`
        // is mock_function's Int32 parameter.
        let planned = transform(
            Arc::new(Column::new("l", 0)),
            "x",
            1,
            |_| {
                Arc::new(with_lambda(
                    inner,
                    Arc::new(Column::new("l", 0)),
                    &["x", "v"],
                    2,
                    int32,
                    |x| x,
                    &schema,
                ))
            },
            &schema,
        );

        let rebound: Arc<dyn PhysicalExpr> = Arc::new(rebind(&planned, &narrowed, &Ok)?);

        let mut fields = Vec::new();
        rebound.apply(|node| {
            if let Some(variable) = node.downcast_ref::<LambdaVariable>() {
                fields.push(variable.field().data_type().clone());
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        assert_eq!(fields, vec![DataType::Int32]);
        assert_eq!(
            evaluate(rebound.as_ref(), &narrowed)?.as_ref(),
            evaluate(planned.as_ref(), &full)?.as_ref()
        );
        Ok(())
    }

    /// A function without a list element lambda is not rebound.
    #[test]
    fn rebind_list_element_needs_a_list_element_lambda() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
        let fun = Arc::new(HigherOrderUDF::new_from_impl(MockHigherOrderUDF {
            signature: HigherOrderSignature::variadic_any(Volatility::Immutable),
        }));
        let function = with_lambda(
            fun,
            Arc::new(Column::new("a", 0)),
            &["x"],
            1,
            Arc::new(Field::new("", DataType::Int32, true)),
            |x| x,
            &schema,
        );
        assert!(
            function
                .rebind_list_element(Arc::new(Column::new("a", 0)), &schema, &Ok)?
                .is_none()
        );
        Ok(())
    }
}
