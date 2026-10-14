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

//! Defines the SINGLE_VALUE aggregation, which gives an uncorrelated scalar
//! subquery its row-count semantics when it is planned as a join.

use std::fmt::Debug;
use std::hash::Hash;
use std::mem::size_of_val;
use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray};
use arrow::datatypes::{DataType, Field, FieldRef, UInt64Type};
use datafusion_common::{Result, ScalarValue, exec_err, not_impl_err};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::{AggregateOrderSensitivity, format_state_name};
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, Signature, Volatility,
};
use datafusion_macros::user_doc;

make_udaf_expr_and_func!(
    SingleValue,
    single_value,
    expression,
    "Returns the only row's value, NULL for no rows, and fails on more than one row",
    single_value_udaf
);

#[user_doc(
    doc_section(label = "General Functions"),
    description = "Returns the value of the only row in a group, or NULL if the group has no rows. Fails if the group has more than one row, even when the rows are equal or NULL. This gives a scalar subquery its row-count semantics when it is planned as a join.",
    syntax_example = "single_value(expression)",
    sql_example = r#"```sql
> SELECT single_value(column_name) FROM table_name WHERE id = 1;
+---------------------------+
| single_value(column_name) |
+---------------------------+
| only_value                |
+---------------------------+
```"#,
    standard_argument(name = "expression",)
)]
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct SingleValue {
    signature: Signature,
}

impl Default for SingleValue {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleValue {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for SingleValue {
    fn name(&self) -> &str {
        "single_value"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        not_impl_err!("Not called because return_field is implemented")
    }

    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
        Ok(Arc::new(
            Field::new(self.name(), arg_fields[0].data_type().clone(), true)
                .with_metadata(arg_fields[0].metadata().clone()),
        ))
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(SingleValueAccumulator {
            value: ScalarValue::try_from(acc_args.return_field.data_type())?,
            rows: 0,
        }))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Field::new(
                format_state_name(args.name, "single_value"),
                args.return_type().clone(),
                true,
            )
            .into(),
            Field::new(
                format_state_name(args.name, "single_value_rows"),
                DataType::UInt64,
                false,
            )
            .into(),
        ])
    }

    fn order_sensitivity(&self) -> AggregateOrderSensitivity {
        AggregateOrderSensitivity::Insensitive
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// Holds the one row seen so far, across every batch and every merged
/// partial state, and fails as soon as a second row arrives.
#[derive(Debug)]
struct SingleValueAccumulator {
    /// The row's value, or a typed NULL while no row has been seen.
    value: ScalarValue,
    /// Rows seen so far: 0 or 1.
    rows: u64,
}

impl SingleValueAccumulator {
    /// Counts `rows` more rows, taking `value` as the row when they make one.
    fn add(
        &mut self,
        rows: u64,
        value: impl FnOnce() -> Result<ScalarValue>,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        if self.rows + rows > 1 {
            return exec_err!("Scalar subquery returned more than one row");
        }
        self.value = value()?;
        self.rows = 1;
        Ok(())
    }
}

impl Accumulator for SingleValueAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = &values[0];
        self.add(values.len() as u64, || {
            ScalarValue::try_from_array(values, 0)
        })
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let (values, rows) = (&states[0], states[1].as_primitive::<UInt64Type>());
        for (index, rows) in rows.values().iter().enumerate() {
            self.add(*rows, || ScalarValue::try_from_array(values, index))?;
        }
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![
            self.value.clone(),
            ScalarValue::UInt64(Some(self.rows)),
        ])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(self.value.clone())
    }

    fn size(&self) -> usize {
        size_of_val(self) - size_of_val(&self.value) + self.value.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, UInt64Array};

    fn accumulator() -> SingleValueAccumulator {
        SingleValueAccumulator {
            value: ScalarValue::Int32(None),
            rows: 0,
        }
    }

    fn ints(values: &[Option<i32>]) -> ArrayRef {
        Arc::new(Int32Array::from(values.to_vec()))
    }

    #[test]
    fn no_rows_is_a_typed_null() -> Result<()> {
        let mut acc = accumulator();
        acc.update_batch(&[ints(&[])])?;
        assert_eq!(acc.evaluate()?, ScalarValue::Int32(None));
        Ok(())
    }

    #[test]
    fn one_row_is_its_value_even_when_null() -> Result<()> {
        let mut acc = accumulator();
        acc.update_batch(&[ints(&[Some(7)])])?;
        assert_eq!(acc.evaluate()?, ScalarValue::Int32(Some(7)));
        let mut acc = accumulator();
        acc.update_batch(&[ints(&[None])])?;
        assert_eq!(acc.evaluate()?, ScalarValue::Int32(None));
        Ok(())
    }

    #[test]
    fn a_second_row_fails_in_a_later_batch_even_when_equal_or_null() {
        for second in [Some(7), None] {
            let mut acc = accumulator();
            acc.update_batch(&[ints(&[Some(7)])]).unwrap();
            let err = acc.update_batch(&[ints(&[second])]).unwrap_err();
            assert!(err.to_string().contains("more than one row"), "{err}");
        }
    }

    #[test]
    fn merged_partial_rows_are_counted_together() -> Result<()> {
        let mut one = accumulator();
        one.update_batch(&[ints(&[Some(1)])])?;
        let mut empty = accumulator();
        let mut merged = accumulator();
        let states = |acc: &mut SingleValueAccumulator| -> Result<Vec<ArrayRef>> {
            acc.state()?.into_iter().map(|s| s.to_array()).collect()
        };
        merged.merge_batch(&states(&mut empty)?)?;
        merged.merge_batch(&states(&mut one)?)?;
        assert_eq!(merged.evaluate()?, ScalarValue::Int32(Some(1)));
        let err = merged.merge_batch(&states(&mut one)?).unwrap_err();
        assert!(err.to_string().contains("more than one row"), "{err}");
        let two: Vec<ArrayRef> =
            vec![ints(&[Some(1)]), Arc::new(UInt64Array::from(vec![2]))];
        let err = accumulator().merge_batch(&two).unwrap_err();
        assert!(err.to_string().contains("more than one row"), "{err}");
        Ok(())
    }
}
