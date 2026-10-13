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

use crate::aggregates::group_values::GroupValues;
use arrow::array::types::{IntervalDayTime, IntervalMonthDayNano};
use arrow::array::{
    ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, NullBufferBuilder, PrimitiveArray,
    cast::AsArray,
};
use arrow::datatypes::{DataType, i256};
use datafusion_common::Result;
use datafusion_common::hash_utils::RandomState;
use datafusion_common::utils::split_vec_min_alloc;
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::{EmitTo, GroupSelection};
use half::f16;
use hashbrown::hash_table::HashTable;
#[cfg(not(feature = "force_hash_collisions"))]
use std::hash::BuildHasher;
use std::mem::size_of;
use std::sync::Arc;

/// A trait to allow hashing of floating point numbers
pub trait HashValue {
    fn hash(&self, state: &RandomState) -> u64;

    /// Return a canonical representative whose bit pattern is identical for
    /// all values that should be grouped together. Default is the identity;
    /// floats override this to fold `-0.0` into `+0.0` so the bit-equal
    /// `is_eq` check used during insertion treats them as the same group.
    /// NaN payload bits are preserved.
    #[inline]
    fn canonicalize(self) -> Self
    where
        Self: Sized,
    {
        self
    }

    /// Return the value as an `i64` when equal values, and only equal
    /// values, map to equal integers, so a group can be addressed directly
    /// by its offset from the smallest key. Default is `None`, which keeps
    /// the hash table.
    #[inline]
    fn dense_key(self) -> Option<i64>
    where
        Self: Sized,
    {
        None
    }
}

macro_rules! hash_integer {
    ($key:ident => $dense:expr; $($t:ty),+) => {
        $(impl HashValue for $t {
            #[cfg(not(feature = "force_hash_collisions"))]
            fn hash(&self, state: &RandomState) -> u64 {
                state.hash_one(self)
            }

            #[cfg(feature = "force_hash_collisions")]
            fn hash(&self, _state: &RandomState) -> u64 {
                0
            }

            #[inline]
            fn dense_key(self) -> Option<i64> {
                let $key = self;
                $dense
            }
        })+
    };
    ($($t:ty),+) => {
        hash_integer!(_key => None; $($t),+);
    };
}
hash_integer!(key => Some(i64::from(key)); i8, i16, i32, u8, u16, u32);
hash_integer!(key => Some(key); i64);
hash_integer!(key => i64::try_from(key).ok(); i128, u64);
hash_integer!(i256);
hash_integer!(IntervalDayTime, IntervalMonthDayNano);

macro_rules! hash_float {
    ($($t:ty),+) => {
        $(impl HashValue for $t {
            #[cfg(not(feature = "force_hash_collisions"))]
            fn hash(&self, state: &RandomState) -> u64 {
                state.hash_one(self.canonicalize().to_bits())
            }

            #[cfg(feature = "force_hash_collisions")]
            fn hash(&self, _state: &RandomState) -> u64 {
                0
            }

            #[inline]
            fn canonicalize(self) -> Self {
                let bits = self.to_bits();
                let bits = if bits << 1 == 0 { 0 } else { bits };
                Self::from_bits(bits)
            }
        })+
    };
}

hash_float!(f16, f32, f64);

/// The most keys [`DenseIndex`] spans before grouping falls back to the
/// hash table: 4 KiB of slots.
const DENSE_MAX_SLOTS: u64 = 1024;

/// Group indexes addressed by `key - base`, for keys that all fall within
/// [`DENSE_MAX_SLOTS`] of each other.
///
/// Low-cardinality integer keys (enums, status codes, small id ranges) are
/// interned by one array lookup instead of a hash and a probe.
#[derive(Debug, Default)]
struct DenseIndex {
    /// The key stored in `slots[0]`.
    base: i64,
    /// The group index of each key from `base`, or [`DenseIndex::EMPTY`].
    slots: Vec<u32>,
}

impl DenseIndex {
    /// A slot whose key has no group yet.
    const EMPTY: u32 = u32::MAX;

    /// Widens the index to cover every non-null key of `array`.
    ///
    /// Returns `false`, leaving the index unchanged, when a key has no
    /// [`HashValue::dense_key`] or the widened range would exceed
    /// [`DENSE_MAX_SLOTS`].
    fn cover<T: ArrowPrimitiveType>(&mut self, array: &PrimitiveArray<T>) -> bool
    where
        T::Native: HashValue,
    {
        let mut range = (!self.slots.is_empty())
            .then(|| (self.base, self.base + self.slots.len() as i64 - 1));
        for key in array.iter().flatten() {
            let Some(key) = key.dense_key() else {
                return false;
            };
            range = Some(range.map_or((key, key), |(lo, hi)| (lo.min(key), hi.max(key))));
        }
        let Some((lo, hi)) = range else {
            return true;
        };
        if hi.abs_diff(lo) >= DENSE_MAX_SLOTS {
            return false;
        }
        let span = hi.abs_diff(lo) as usize + 1;
        if self.slots.is_empty() {
            self.base = lo;
            self.slots.resize(span, Self::EMPTY);
        } else if span > self.slots.len() {
            let shift = self.base.abs_diff(lo) as usize;
            self.slots.resize(span, Self::EMPTY);
            self.slots.rotate_right(shift);
            self.base = lo;
        }
        true
    }

    /// Returns the slot of `key`, which [`DenseIndex::cover`] has covered.
    #[inline]
    fn slot(&mut self, key: i64) -> &mut u32 {
        &mut self.slots[key.abs_diff(self.base) as usize]
    }
}

/// A [`GroupValues`] storing a single column of primitive values
///
/// This specialization is significantly faster than using the more general
/// purpose `Row`s format
pub struct GroupValuesPrimitive<T: ArrowPrimitiveType> {
    /// The data type of the output array
    data_type: DataType,
    /// Stores the `(group_index, hash)` based on the hash of its value
    ///
    /// We also store `hash` is for reducing cost of rehashing. Such cost
    /// is obvious in high cardinality group by situation.
    /// More details can see:
    /// <https://github.com/apache/datafusion/issues/15961>
    map: HashTable<(usize, u64)>,
    /// The group index of the null value if any
    null_group: Option<usize>,
    /// The values for each group index
    values: Vec<T::Native>,
    /// The random state used to generate hashes
    random_state: RandomState,
    /// Direct group indexes while every key fits a [`DenseIndex`], with
    /// `map` empty; `None` once one does not, with `map` holding every group.
    dense: Option<DenseIndex>,
}

impl<T: ArrowPrimitiveType> GroupValuesPrimitive<T> {
    pub fn new(data_type: DataType) -> Self {
        assert!(PrimitiveArray::<T>::is_compatible(&data_type));
        Self {
            data_type,
            map: HashTable::with_capacity(128),
            values: Vec::with_capacity(128),
            null_group: None,
            random_state: crate::aggregates::AGGREGATION_HASH_SEED,
            dense: Some(DenseIndex::default()),
        }
    }

    /// Stops direct addressing and moves every group into the hash table.
    fn leave_dense(&mut self)
    where
        T::Native: HashValue,
    {
        self.dense = None;
        self.map.reserve(self.values.len(), |&(_, h)| h);
        for (group, value) in self.values.iter().enumerate() {
            if Some(group) != self.null_group {
                let hash = value.hash(&self.random_state);
                self.map.insert_unique(hash, (group, hash), |&(_, h)| h);
            }
        }
    }
}

impl<T: ArrowPrimitiveType> GroupValues for GroupValuesPrimitive<T>
where
    T::Native: HashValue,
{
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);
        groups.clear();
        let array = cols[0].as_primitive::<T>();

        if let Some(dense) = &mut self.dense {
            if dense.cover(array) {
                for v in array {
                    let group_id = match v {
                        None => *self.null_group.get_or_insert_with(|| {
                            let group_id = self.values.len();
                            self.values.push(Default::default());
                            group_id
                        }),
                        Some(key) => {
                            let slot = dense
                                .slot(key.dense_key().expect("cover proved a dense key"));
                            if *slot == DenseIndex::EMPTY {
                                *slot = self.values.len() as u32;
                                self.values.push(key);
                            }
                            *slot as usize
                        }
                    };
                    groups.push(group_id)
                }
                return Ok(());
            }
            self.leave_dense();
        }

        for v in array {
            let group_id = match v {
                None => *self.null_group.get_or_insert_with(|| {
                    let group_id = self.values.len();
                    self.values.push(Default::default());
                    group_id
                }),
                Some(key) => {
                    // Fold equivalence-class duplicates (e.g. `-0.0` → `+0.0`)
                    // so the bit-equal `is_eq` matches and the stored value is
                    // the canonical representative.
                    let key = key.canonicalize();
                    let state = &self.random_state;
                    let hash = key.hash(state);
                    let insert = self.map.entry(
                        hash,
                        |&(g, h)| unsafe {
                            hash == h && self.values.get_unchecked(g).is_eq(key)
                        },
                        |&(_, h)| h,
                    );

                    match insert {
                        hashbrown::hash_table::Entry::Occupied(o) => o.get().0,
                        hashbrown::hash_table::Entry::Vacant(v) => {
                            let g = self.values.len();
                            v.insert((g, hash));
                            self.values.push(key);
                            g
                        }
                    }
                }
            };
            groups.push(group_id)
        }
        Ok(())
    }

    fn size(&self) -> usize {
        self.map.capacity() * size_of::<(usize, u64)>()
            + self.values.allocated_size()
            + self
                .dense
                .as_ref()
                .map_or(0, |dense| dense.slots.allocated_size())
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        fn build_primitive<T: ArrowPrimitiveType>(
            values: Vec<T::Native>,
            null_idx: Option<usize>,
        ) -> PrimitiveArray<T> {
            let nulls = null_idx.map(|null_idx| {
                let mut buffer = NullBufferBuilder::new(values.len());
                buffer.append_n_non_nulls(null_idx);
                buffer.append_null();
                buffer.append_n_non_nulls(values.len() - null_idx - 1);
                // NOTE: The inner builder must be constructed as there is at least one null
                buffer.finish().unwrap()
            });
            PrimitiveArray::<T>::new(values.into(), nulls)
        }

        let array: PrimitiveArray<T> = match emit_to {
            EmitTo::All => {
                self.map.clear();
                if let Some(dense) = &mut self.dense {
                    dense.slots.clear();
                }
                build_primitive(std::mem::take(&mut self.values), self.null_group.take())
            }
            EmitTo::First(n) => {
                self.map.retain(|entry| {
                    // Decrement group index by n
                    let group_idx = entry.0;
                    match group_idx.checked_sub(n) {
                        // Group index was >= n, shift value down
                        Some(sub) => {
                            entry.0 = sub;
                            true
                        }
                        // Group index was < n, so remove from table
                        None => false,
                    }
                });
                if let Some(dense) = &mut self.dense {
                    // Same shift as the map: drop groups < n, move the rest down
                    for slot in
                        dense.slots.iter_mut().filter(|s| **s != DenseIndex::EMPTY)
                    {
                        *slot = slot.checked_sub(n as u32).unwrap_or(DenseIndex::EMPTY);
                    }
                }
                let null_group = match &mut self.null_group {
                    Some(v) if *v >= n => {
                        *v -= n;
                        None
                    }
                    Some(_) => self.null_group.take(),
                    None => None,
                };
                build_primitive(split_vec_min_alloc(&mut self.values, n), null_group)
            }
        };

        Ok(vec![Arc::new(array.with_data_type(self.data_type.clone()))])
    }

    fn values_preserving(
        &mut self,
        selection: GroupSelection<'_>,
    ) -> Result<Vec<ArrayRef>> {
        selection.validate_num_groups(self.values.len())?;
        let values: Vec<T::Native> =
            selection.iter().map(|index| self.values[index]).collect();
        let nulls = if let Some(null_group) = self.null_group {
            let mut nulls = NullBufferBuilder::new(values.len());
            for index in selection.iter() {
                if index == null_group {
                    nulls.append_null();
                } else {
                    nulls.append_non_null();
                }
            }
            nulls.finish()
        } else {
            None
        };
        let array = PrimitiveArray::<T>::new(values.into(), nulls)
            .with_data_type(self.data_type.clone());
        Ok(vec![Arc::new(array)])
    }

    fn supports_values_preserving(&self) -> bool {
        true
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.values.clear();
        self.values.shrink_to(num_rows);
        self.map.clear();
        self.map.shrink_to(num_rows, |_| 0); // hasher does not matter since the map is cleared
        self.dense = Some(DenseIndex::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::types::{Int32Type, Int64Type};
    use arrow::array::{ArrayRef, Int32Array, Int64Array};
    use arrow::datatypes::DataType;
    use datafusion_expr::EmitTo;
    use std::sync::Arc;

    /// Mirror of the `EmitTo::take_needed` regression test, applied to the
    /// concrete `GroupValuesPrimitive` accumulator.
    ///
    /// When `n` is small, the old `split_off(n) + swap` pattern used inside
    /// `emit(EmitTo::First(n))` left `self.values` with a small fresh allocation
    /// and returned the emitted prefix carrying the original large backing.
    ///
    /// With `split_vec_min_alloc` and `n * 2 <= len`, the drain branch is taken:
    /// the emitted prefix gets a compact allocation and `self.values` retains the
    /// original large one.
    #[test]
    fn emit_first_small_n_allocates_minimally() -> Result<()> {
        let mut gv = GroupValuesPrimitive::<Int32Type>::new(DataType::Int32);

        // Intern 20 distinct values; `new()` pre-allocates capacity 128 for `values`.
        let arr: ArrayRef = Arc::new(Int32Array::from_iter_values(0..20i32));
        let mut groups = vec![];
        gv.intern(&[arr], &mut groups)?;
        let capacity_before = gv.values.capacity(); // 128

        // n=4, n*2=8 <= len=20 -> drain branch
        let emitted = gv.emit(EmitTo::First(4))?;

        assert_eq!(emitted[0].len(), 4);

        // `self.values` must retain its original large allocation.
        // Old split_off+swap left it with a fresh small allocation (~16).
        assert_eq!(
            gv.values.capacity(),
            capacity_before,
            "self.values capacity {} should equal original {} after small First(n) emit",
            gv.values.capacity(),
            capacity_before,
        );

        Ok(())
    }

    /// Interns `keys` and returns their group indexes.
    fn intern(
        gv: &mut GroupValuesPrimitive<Int64Type>,
        keys: Vec<Option<i64>>,
    ) -> Vec<usize> {
        let mut groups = vec![];
        let array: ArrayRef = Arc::new(Int64Array::from(keys));
        gv.intern(&[array], &mut groups).unwrap();
        groups
    }

    /// Small key ranges are addressed directly, widening downward keeps
    /// existing groups, and a key past `DENSE_MAX_SLOTS` moves every group
    /// into the hash table without renumbering any.
    #[test]
    fn dense_groups_widen_then_fall_back_to_the_hash_table() -> Result<()> {
        let mut gv = GroupValuesPrimitive::<Int64Type>::new(DataType::Int64);

        assert_eq!(
            intern(&mut gv, vec![Some(10), None, Some(12), Some(10)]),
            [0, 1, 2, 0]
        );
        assert_eq!(intern(&mut gv, vec![Some(-5), Some(12), None]), [3, 2, 1]);
        assert!(gv.dense.is_some() && gv.map.is_empty());

        assert_eq!(
            intern(&mut gv, vec![Some(5000), Some(-5), Some(10)]),
            [4, 3, 0]
        );
        assert!(gv.dense.is_none());
        assert_eq!(intern(&mut gv, vec![Some(12), None, Some(7)]), [2, 1, 5]);

        let emitted = gv.emit(EmitTo::All)?;
        let expected: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(10),
            None,
            Some(12),
            Some(-5),
            Some(5000),
            Some(7),
        ]));
        assert_eq!(&emitted[0], &expected);
        Ok(())
    }

    /// Emitting the first groups shifts the direct indexes the same way it
    /// shifts the hash table's.
    #[test]
    fn dense_emit_first_renumbers_remaining_groups() -> Result<()> {
        let mut gv = GroupValuesPrimitive::<Int64Type>::new(DataType::Int64);
        assert_eq!(intern(&mut gv, vec![Some(1), Some(2), Some(3)]), [0, 1, 2]);

        let emitted = gv.emit(EmitTo::First(2))?;
        assert_eq!(emitted[0].len(), 2);

        assert_eq!(intern(&mut gv, vec![Some(3), Some(1)]), [0, 1]);
        Ok(())
    }
}
