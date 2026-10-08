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

//! Variant leaf projection: unshredded vs shredded.
//!
//! The before/after of `parquet_struct_projection`, for Variant. Before is an
//! unshredded Variant: each object is one binary value, so reading one key
//! decodes the whole object. After is the same objects shredded onto their
//! own type: each key is its own Parquet leaf, so reading one key decodes only
//! that leaf. Same SQL, same rows, same build; only the file layout differs.
//!
//! The shapes and cases are `parquet_struct_projection`'s (narrow, wide and
//! nested; `select_struct` is `select_variant` and `select_inner_struct` is
//! `select_inner_object`). String leaves are 16 KiB and differ per row, so
//! nothing collapses to a dictionary. Both tables declare the canonical
//! (unshredded) Variant, as a table does when its files may hold either
//! layout.
//!
//! `{shape}/{layout}/{case}` times planning plus execution, as
//! `parquet_struct_projection` does; `{shape}/plan/{layout}/{case}` times
//! planning alone.
//!
//! Reading the whole Variant (`select_variant`, `select_all`) from shredded
//! files rebuilds each value, so those cases may be slower than unshredded,
//! but by no more than 2x.
//!
//! Runs on mimalloc, as `datafusion-cli` and the `benchmarks` crate do. Under
//! glibc's default allocator every multi-megabyte output buffer is mapped
//! fresh and page-faulted in, which dominates the whole-Variant cases.

use arrow::array::{ArrayRef, Int32Array, RecordBatch, StringArray, StructArray};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{WriterProperties, WriterVersion};
use parquet_variant_compute::{VariantType, cast_to_variant, shred_variant};
use std::fmt::Write as _;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::runtime::Runtime;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const NUM_BATCHES: usize = 2;
const WRITE_RECORD_BATCH_SIZE: usize = 256;
const LARGE_STRING_LEN: usize = 16 * 1024;

/// `(shape, the objects' type, [(case, SQL)])`.
type Shape = (&'static str, Fields, [(&'static str, &'static str); 6]);

fn utf8(name: &str) -> Field {
    Field::new(name, DataType::Utf8, false)
}

fn small_int() -> Field {
    Field::new("small_int", DataType::Int32, false)
}

fn shapes() -> [Shape; 3] {
    let inner = Fields::from(vec![utf8("large_string"), small_int()]);
    [
        (
            "narrow",
            inner.clone(),
            [
                ("select_variant", "SELECT s FROM t"),
                (
                    "select_small_field",
                    "SELECT CAST(s ->> 'small_int' AS INT) FROM t",
                ),
                ("select_large_field", "SELECT s ->> 'large_string' FROM t"),
                ("select_all", "SELECT * FROM t"),
                (
                    "select_id_and_small_field",
                    "SELECT id, CAST(s ->> 'small_int' AS INT) FROM t",
                ),
                (
                    "sum_small_field",
                    "SELECT SUM(CAST(s ->> 'small_int' AS INT)) FROM t",
                ),
            ],
        ),
        (
            "wide",
            Fields::from(vec![
                utf8("str_a"),
                utf8("str_b"),
                utf8("str_c"),
                utf8("str_d"),
                small_int(),
            ]),
            [
                ("select_variant", "SELECT s FROM t"),
                (
                    "select_small_field",
                    "SELECT CAST(s ->> 'small_int' AS INT) FROM t",
                ),
                ("select_one_string_field", "SELECT s ->> 'str_a' FROM t"),
                (
                    "select_two_string_fields",
                    "SELECT s ->> 'str_a', s ->> 'str_b' FROM t",
                ),
                ("select_all", "SELECT * FROM t"),
                (
                    "sum_small_field",
                    "SELECT SUM(CAST(s ->> 'small_int' AS INT)) FROM t",
                ),
            ],
        ),
        (
            "nested",
            Fields::from(vec![
                Field::new("inner", DataType::Struct(inner), false),
                utf8("extra_string"),
            ]),
            [
                ("select_variant", "SELECT s FROM t"),
                ("select_inner_object", "SELECT s -> 'inner' FROM t"),
                (
                    "select_inner_small_field",
                    "SELECT CAST(s -> 'inner' ->> 'small_int' AS INT) FROM t",
                ),
                ("select_extra_string", "SELECT s ->> 'extra_string' FROM t"),
                ("select_all", "SELECT * FROM t"),
                (
                    "sum_inner_small_field",
                    "SELECT SUM(CAST(s -> 'inner' ->> 'small_int' AS INT)) FROM t",
                ),
            ],
        ),
    ]
}

/// `LARGE_STRING_LEN` bytes of xorshift hex seeded by `name` and `row`, so
/// siblings and rows differ.
fn text(name: &str, row: usize) -> String {
    let mut state = name
        .bytes()
        .fold(row as u64 ^ 0x9E37_79B9_7F4A_7C15, |hash, byte| {
            hash.rotate_left(5) ^ u64::from(byte)
        })
        | 1;
    let mut value = String::with_capacity(LARGE_STRING_LEN);
    while value.len() < LARGE_STRING_LEN {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        write!(value, "{state:016x}").unwrap();
    }
    value
}

/// `field` over `rows`: Int32 leaves hold the row number, strings [`text`].
fn column(field: &Field, rows: std::ops::Range<usize>) -> ArrayRef {
    match field.data_type() {
        DataType::Struct(fields) => Arc::new(StructArray::new(
            fields.clone(),
            fields.iter().map(|f| column(f, rows.clone())).collect(),
            None,
        )),
        DataType::Int32 => Arc::new(Int32Array::from_iter_values(rows.map(|r| r as i32))),
        _ => Arc::new(StringArray::from_iter_values(
            rows.map(|r| text(field.name(), r)),
        )),
    }
}

/// The canonical (unshredded) Variant field `s`.
fn canonical_variant() -> Field {
    Field::new(
        "s",
        DataType::Struct(Fields::from(vec![
            Field::new("metadata", DataType::Binary, false),
            Field::new("value", DataType::Binary, false),
        ])),
        false,
    )
    .with_extension_type(VariantType)
}

/// Writes objects of type `fields` to `dir` as a Variant, shredded onto
/// `fields` when `shredded` is set.
fn write(dir: &TempDir, fields: &Fields, shredded: bool) {
    let id = Field::new("id", DataType::Int32, false);
    let s = Field::new("s", DataType::Struct(fields.clone()), false);
    let batches: Vec<RecordBatch> = (0..NUM_BATCHES)
        .map(|batch| {
            let rows =
                batch * WRITE_RECORD_BATCH_SIZE..(batch + 1) * WRITE_RECORD_BATCH_SIZE;
            let variant = cast_to_variant(&column(&s, rows.clone())).unwrap();
            let variant = if shredded {
                shred_variant(&variant, s.data_type()).unwrap()
            } else {
                variant
            };
            let field = Field::new("s", variant.data_type().clone(), false)
                .with_extension_type(VariantType);
            let schema = Arc::new(Schema::new(vec![id.clone(), field]));
            RecordBatch::try_new(schema, vec![column(&id, rows), variant.into()]).unwrap()
        })
        .collect();
    let file = std::fs::File::create(dir.path().join("data.parquet")).unwrap();
    let properties = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_max_row_group_row_count(Some(WRITE_RECORD_BATCH_SIZE))
        .build();
    let mut writer =
        ArrowWriter::try_new(file, batches[0].schema(), Some(properties)).unwrap();
    for batch in &batches {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
}

/// A session with `dir` registered as `t` under the canonical Variant schema.
fn context(dir: &TempDir) -> SessionContext {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        canonical_variant(),
    ]));
    let ctx = SessionContext::new();
    let url = ListingTableUrl::parse(dir.path().to_str().unwrap()).unwrap();
    let options = ListingOptions::new(Arc::new(ParquetFormat::default()))
        .with_file_extension(".parquet");
    let config = ListingTableConfig::new(url)
        .with_listing_options(options)
        .with_schema(schema);
    // The session's statistics cache, as `register_parquet` attaches it.
    let table = ListingTable::try_new(config)
        .unwrap()
        .with_cache(ctx.runtime_env().cache_manager.get_file_statistic_cache());
    ctx.register_table("t", Arc::new(table)).unwrap();
    ctx
}

fn criterion_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    for (shape, fields, cases) in shapes() {
        for (layout, shredded) in [("unshredded", false), ("shredded", true)] {
            let dir = TempDir::new().unwrap();
            write(&dir, &fields, shredded);
            let ctx = context(&dir);
            let mut group = c.benchmark_group(shape);
            // 8 MB results vary run to run; 10 samples is too few.
            group.sample_size(50);
            group.warm_up_time(Duration::from_secs(1));
            group.measurement_time(Duration::from_secs(3));
            for (case, sql) in cases {
                // Fail fast if the query cannot run.
                rt.block_on(async { ctx.sql(sql).await?.collect().await })
                    .unwrap();
                group.bench_function(format!("{layout}/{case}"), |b| {
                    b.iter(|| {
                        black_box(
                            rt.block_on(async { ctx.sql(sql).await?.collect().await }),
                        )
                        .unwrap()
                    })
                });
                group.bench_function(format!("plan/{layout}/{case}"), |b| {
                    b.iter(|| {
                        black_box(rt.block_on(async {
                            ctx.sql(sql).await?.create_physical_plan().await
                        }))
                        .unwrap()
                    })
                });
            }
            group.finish();
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
