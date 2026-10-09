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

//! Reading shredded Parquet Variant columns as a table's canonical Variant.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, ListArray, RecordBatch, StringArray, StructArray,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Fields, Schema};
use bytes::{BufMut, BytesMut};
use datafusion::common::Result;
use datafusion::datasource::listing::{
    ListingTable, ListingTableConfig, ListingTableConfigExt,
};
use datafusion::functions::core::expr_fn::get_field;
use datafusion::functions_nested::expr_fn::{array_element, array_transform};
use datafusion::functions_variant::VARIANT_GET_TEXT;
use datafusion::logical_expr::{LogicalPlan, col, lambda, lambda_var, lit};
use datafusion::prelude::SessionContext;
use datafusion_datasource::ListingTableUrl;
use datafusion_execution::object_store::ObjectStoreUrl;
use object_store::{ObjectStore, ObjectStoreExt, memory::InMemory, path::Path};
use parquet::arrow::ArrowWriter;
use parquet_variant_compute::{json_to_variant, shred_variant, variant_to_json};

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

/// `json` rows as Variants shredded with `typed_value: Int64`.
fn shredded(json: Vec<Option<&str>>) -> ArrayRef {
    let text: ArrayRef = Arc::new(StringArray::from(json));
    let variants = json_to_variant(&text).unwrap();
    ArrayRef::from(shred_variant(&variants, &DataType::Int64).unwrap())
}

/// The JSON text of each row of a Variant array.
fn json(array: &ArrayRef) -> Vec<Option<String>> {
    variant_to_json(array)
        .unwrap()
        .iter()
        .map(|row| row.map(str::to_owned))
        .collect()
}

/// A file with a shredded Variant at the top level (`v`), inside a Struct
/// whose second row is null (`s.v`), and as List elements (`l`), registered
/// as table `t` whose schema declares each as the canonical Variant.
async fn shredded_table() -> SessionContext {
    let v = shredded(vec![Some("1"), Some(r#""x""#), None]);
    let inner = shredded(vec![Some("10"), Some("20"), Some(r#"{"k":1}"#)]);
    let s: ArrayRef = Arc::new(StructArray::new(
        Fields::from(vec![Field::new("v", inner.data_type().clone(), true)]),
        vec![inner],
        Some(NullBuffer::from(vec![true, false, true])),
    ));
    let elements = shredded(vec![Some("1"), Some("2"), Some(r#""y""#)]);
    let l: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new("element", elements.data_type().clone(), true)),
        OffsetBuffer::<i32>::from_lengths([2, 0, 1]),
        elements,
        None,
    ));
    let batch = RecordBatch::try_from_iter([("v", v), ("s", s), ("l", l)]).unwrap();

    let mut out = BytesMut::new().writer();
    let mut writer = ArrowWriter::try_new(&mut out, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    store
        .put(
            &Path::from("variant.parquet"),
            out.into_inner().freeze().into(),
        )
        .await
        .unwrap();

    let table_schema = Arc::new(Schema::new(vec![
        variant_field("v"),
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![variant_field("v")])),
            true,
        ),
        Field::new(
            "l",
            DataType::List(Arc::new(variant_field("element"))),
            true,
        ),
    ]));
    let ctx = SessionContext::new();
    let url = ObjectStoreUrl::parse("memory://").unwrap();
    ctx.register_object_store(url.as_ref(), Arc::clone(&store));
    let config = ListingTableConfig::new(ListingTableUrl::parse("memory:///").unwrap())
        .infer_options(&ctx.state())
        .await
        .unwrap()
        .with_schema(table_schema);
    ctx.register_table("t", Arc::new(ListingTable::try_new(config).unwrap()))
        .unwrap();
    ctx
}

/// The only result column of `sql` as one array.
async fn column(ctx: &SessionContext, sql: &str) -> Result<ArrayRef> {
    let batches = ctx.sql(sql).await?.collect().await?;
    let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;
    Ok(Arc::clone(batch.column(0)))
}

#[tokio::test]
async fn shredded_variant_reads_as_canonical_at_every_depth() -> Result<()> {
    let ctx = shredded_table().await;

    let v = column(&ctx, "SELECT v FROM t").await?;
    assert_eq!(
        json(&v),
        vec![Some("1".to_owned()), Some(r#""x""#.to_owned()), None]
    );

    let s = column(&ctx, "SELECT s FROM t").await?;
    let s_v = Arc::clone(s.as_struct().column(0));
    assert_eq!(s.is_null(1), true);
    assert_eq!(json(&s_v)[0].as_deref(), Some("10"));
    assert_eq!(json(&s_v)[2].as_deref(), Some(r#"{"k":1}"#));

    let field = column(&ctx, "SELECT s['v'] FROM t").await?;
    assert_eq!(
        json(&field),
        vec![Some("10".to_owned()), None, Some(r#"{"k":1}"#.to_owned())]
    );

    let l = column(&ctx, "SELECT l FROM t").await?;
    let elements = Arc::clone(l.as_list::<i32>().values());
    assert_eq!(
        json(&elements),
        vec![
            Some("1".to_owned()),
            Some("2".to_owned()),
            Some(r#""y""#.to_owned())
        ]
    );
    Ok(())
}

/// One row per JSON text, as Variant column `v` of a query.
const DOCS: &str = r#"WITH t AS (
    SELECT parse_json(j) AS v FROM (VALUES
        ('{"a":{"b":"x"},"n":1,"l":[1,"two"],"z":null}'),
        ('{"a":{"b":2}}'),
        (NULL)
    ) AS s(j)
)"#;

#[tokio::test]
async fn variant_operators_and_functions_follow_the_contract() -> Result<()> {
    let ctx = SessionContext::new();
    let batches = ctx
        .sql(&format!(
            "{DOCS} SELECT v ->> 'n' AS n, v -> 'a' ->> 'b' AS ab, to_json(v -> 'l') AS l, \
             v -> 'l' ->> 1 AS l1, v ->> 'z' AS z, v ->> 'missing' AS missing, \
             CAST(v ->> 'n' AS BIGINT) + 1 AS n1 FROM t"
        ))
        .await?
        .collect()
        .await?;
    datafusion::assert_batches_eq!(
        [
            "+---+----+-----------+-----+---+---------+----+",
            "| n | ab | l         | l1  | z | missing | n1 |",
            "+---+----+-----------+-----+---+---------+----+",
            "| 1 | x  | [1,\"two\"] | two |   |         | 2  |",
            "|   | 2  |           |     |   |         |    |",
            "|   |    |           |     |   |         |    |",
            "+---+----+-----------+-----+---+---------+----+",
        ],
        &batches
    );
    Ok(())
}

#[tokio::test]
async fn arrow_chain_plans_one_path_lookup() -> Result<()> {
    let ctx = SessionContext::new();
    let plan = ctx
        .sql(&format!("{DOCS} SELECT v -> 'a' ->> 'b' FROM t"))
        .await?
        .into_optimized_plan()?;
    let plan = plan.display_indent().to_string();
    assert!(plan.contains(r#"variant_get_text("#), "{plan}");
    assert!(plan.contains(r#", Utf8("a"), Utf8("b"))"#), "{plan}");
    assert!(!plan.contains("variant_get("), "{plan}");
    Ok(())
}

#[tokio::test]
async fn parse_json_refuses_invalid_json_and_try_parse_json_nulls_it() -> Result<()> {
    let ctx = SessionContext::new();
    let error = ctx
        .sql("SELECT parse_json('{not json')")
        .await?
        .collect()
        .await
        .unwrap_err();
    assert!(error.to_string().contains("is not valid JSON"), "{error}");

    let batches = ctx
        .sql("SELECT to_json(try_parse_json('{not json')) AS j")
        .await?
        .collect()
        .await?;
    datafusion::assert_batches_eq!(
        ["+---+", "| j |", "+---+", "|   |", "+---+"],
        &batches
    );
    Ok(())
}

#[tokio::test]
async fn functions_read_shredded_files() -> Result<()> {
    let ctx = shredded_table().await;
    let batches = ctx
        .sql("SELECT to_json(v) AS v, s['v'] ->> 'k' AS k, to_json(l[1]) AS l1 FROM t")
        .await?
        .collect()
        .await?;
    datafusion::assert_batches_eq!(
        [
            "+-----+---+-----+",
            "| v   | k | l1  |",
            "+-----+---+-----+",
            "| 1   |   | 1   |",
            "| \"x\" |   |     |",
            "|     | 1 | \"y\" |",
            "+-----+---+-----+",
        ],
        &batches
    );
    Ok(())
}

#[tokio::test]
async fn malformed_variant_bytes_are_an_error_not_a_panic() -> Result<()> {
    use arrow::array::BinaryArray;
    use datafusion::datasource::MemTable;

    let field = variant_field("v");
    let DataType::Struct(children) = field.data_type() else {
        unreachable!("a Variant field is a struct")
    };
    let v = StructArray::new(
        children.clone(),
        vec![
            Arc::new(BinaryArray::from(vec![&[1u8, 0, 0][..]])),
            Arc::new(BinaryArray::from(vec![&[0xFFu8, 0xFF, 0xFF][..]])),
        ],
        None,
    );
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(v)])?;
    let ctx = SessionContext::new();
    ctx.register_table(
        "bad",
        Arc::new(MemTable::try_new(schema, vec![vec![batch]])?),
    )?;

    for sql in [
        "SELECT to_json(v) FROM bad",
        "SELECT v ->> 'k' FROM bad",
        "SELECT v -> 'k' FROM bad",
    ] {
        let error = ctx.sql(sql).await?.collect().await.unwrap_err();
        assert!(
            error.to_string().contains("not a valid Variant"),
            "{sql}: {error}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn unshredded_variant_under_a_null_struct_is_null() -> Result<()> {
    use arrow::array::BinaryArray;
    use datafusion::datasource::MemTable;

    let field = variant_field("v");
    let DataType::Struct(children) = field.data_type() else {
        unreachable!("a Variant field is a struct")
    };
    // Row 1's parent is null and its Variant holds empty placeholder bytes.
    let v = StructArray::new(
        children.clone(),
        vec![
            Arc::new(BinaryArray::from(vec![&[1u8, 0, 0][..], &[][..]])),
            Arc::new(BinaryArray::from(vec![&[0x0Cu8, 7][..], &[][..]])),
        ],
        None,
    );
    let s_field = Field::new("s", DataType::Struct(Fields::from(vec![field])), true);
    let s = StructArray::new(
        Fields::from(vec![variant_field("v")]),
        vec![Arc::new(v)],
        Some(NullBuffer::from(vec![true, false])),
    );
    let schema = Arc::new(Schema::new(vec![s_field]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(s)])?;
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))?;

    let batches = ctx
        .sql("SELECT to_json(s['v']) AS j, s['v'] ->> '' AS k FROM t")
        .await?
        .collect()
        .await?;
    datafusion::assert_batches_eq!(
        [
            "+---+---+",
            "| j | k |",
            "+---+---+",
            "| 7 |   |",
            "|   |   |",
            "+---+---+"
        ],
        &batches
    );
    Ok(())
}

/// Bytes the Parquet scans under `plan` read.
fn bytes_scanned(plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>) -> usize {
    plan.metrics()
        .and_then(|metrics| metrics.sum_by_name("bytes_scanned"))
        .map_or(0, |value| value.as_usize())
        + plan
            .children()
            .into_iter()
            .map(bytes_scanned)
            .sum::<usize>()
}

/// Runs `sql` and returns its first column as strings and the bytes scanned.
async fn run(ctx: &SessionContext, sql: &str) -> Result<(Vec<Option<String>>, usize)> {
    let plan = ctx.sql(sql).await?.into_unoptimized_plan();
    run_plan(ctx, &plan).await
}

/// Runs `plan` and returns its first column as strings and the bytes scanned.
async fn run_plan(
    ctx: &SessionContext,
    plan: &LogicalPlan,
) -> Result<(Vec<Option<String>>, usize)> {
    let plan = ctx.state().create_physical_plan(plan).await?;
    let batches =
        datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx()).await?;
    let column = arrow::compute::concat(
        &batches
            .iter()
            .map(|batch| batch.column(0).as_ref())
            .collect::<Vec<_>>(),
    )?;
    let column = arrow::compute::cast(&column, &DataType::Utf8)?;
    let values = column
        .as_string::<i32>()
        .iter()
        .map(|v| v.map(str::to_owned))
        .collect();
    Ok((values, bytes_scanned(&plan)))
}

/// A file whose Variant `v` (and `s.v`, and `v` in the one element of each
/// row's `l`) shreds `{a: Int64, b: Utf8}`, with a large distinct `b` per
/// row, written with the Variant extension type as a Variant writer does,
/// and registered as `t` with canonical Variant columns.
async fn object_shredded_table() -> SessionContext {
    use parquet_variant_compute::VariantType;

    let rows = 2048;
    let text: ArrayRef = Arc::new(StringArray::from_iter_values(
        (0..rows).map(|i| format!(r#"{{"a":{i},"b":"{i:0>1000}"}}"#)),
    ));
    let object = DataType::Struct(Fields::from(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Utf8, true),
    ]));
    let v =
        ArrayRef::from(shred_variant(&json_to_variant(&text).unwrap(), &object).unwrap());
    let v_field =
        Field::new("v", v.data_type().clone(), true).with_extension_type(VariantType);
    let s_field = Field::new(
        "s",
        DataType::Struct(Fields::from(vec![v_field.clone()])),
        true,
    );
    let s: ArrayRef = Arc::new(StructArray::new(
        Fields::from(vec![v_field.clone()]),
        vec![Arc::clone(&v)],
        None,
    ));
    let l_element = Arc::new(Field::new("item", s_field.data_type().clone(), true));
    let l: ArrayRef = Arc::new(ListArray::new(
        Arc::clone(&l_element),
        OffsetBuffer::from_lengths(std::iter::repeat_n(1, rows)),
        Arc::clone(&s),
        None,
    ));
    let l_field = Field::new("l", DataType::List(l_element), true);
    let schema = Arc::new(Schema::new(vec![v_field, s_field, l_field]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![v, s, l]).unwrap();

    let mut out = BytesMut::new().writer();
    let mut writer = ArrowWriter::try_new(&mut out, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    store
        .put(
            &Path::from("object.parquet"),
            out.into_inner().freeze().into(),
        )
        .await
        .unwrap();

    let s_type = DataType::Struct(Fields::from(vec![variant_field("v")]));
    let table_schema = Arc::new(Schema::new(vec![
        variant_field("v"),
        Field::new("s", s_type.clone(), true),
        Field::new("l", DataType::new_list(s_type, true), true),
    ]));
    let ctx = SessionContext::new();
    let url = ObjectStoreUrl::parse("memory://").unwrap();
    ctx.register_object_store(url.as_ref(), Arc::clone(&store));
    let config = ListingTableConfig::new(ListingTableUrl::parse("memory:///").unwrap())
        .infer_options(&ctx.state())
        .await
        .unwrap()
        .with_schema(table_schema);
    ctx.register_table("t", Arc::new(ListingTable::try_new(config).unwrap()))
        .unwrap();
    ctx
}

#[tokio::test]
async fn shredded_paths_read_only_their_leaves() -> Result<()> {
    let ctx = object_shredded_table().await;
    let expected = (0..2048).map(|i| Some(i.to_string())).collect::<Vec<_>>();

    let (whole_values, whole) = run(&ctx, "SELECT to_json(v) FROM t").await?;
    assert_eq!(
        whole_values[7].as_deref(),
        Some(&*format!(r#"{{"a":7,"b":"{:0>1000}"}}"#, 7))
    );
    let (values, leaf) = run(&ctx, "SELECT v ->> 'a' FROM t").await?;
    assert_eq!(values, expected);
    assert!(
        leaf * 10 < whole,
        "leaf read {leaf} bytes, whole read {whole}"
    );

    let (values, nested) = run(&ctx, "SELECT s['v'] ->> 'a' FROM t").await?;
    assert_eq!(values, expected);
    assert!(
        nested * 10 < whole,
        "nested leaf read {nested} bytes, whole read {whole}"
    );

    let (values, listed) = run(&ctx, "SELECT l[1]['v'] ->> 'a' FROM t").await?;
    assert_eq!(values, expected);
    assert!(
        listed * 10 < whole,
        "leaf below a List read {listed} bytes, whole read {whole}"
    );

    // array_transform(l, x -> x['v'] ->> 'a')[1]: SQL cannot yet bind a
    // lambda parameter over a table column.
    let body = VARIANT_GET_TEXT.call(vec![get_field(lambda_var("x"), "v"), lit("a")]);
    let table = ctx.table("t").await?;
    let transform = array_element(array_transform(col("l"), lambda(["x"], body)), lit(1))
        .resolve_lambda_variables(table.schema())?
        .data;
    let plan = table.select(vec![transform])?.into_unoptimized_plan();
    let (values, transformed) = run_plan(&ctx, &plan).await?;
    assert_eq!(values, expected);
    assert!(
        transformed * 10 < whole,
        "leaf in a lambda over a List read {transformed} bytes, whole read {whole}"
    );

    let (values, unnested) = run(&ctx, "SELECT unnest(l)['v'] ->> 'a' FROM t").await?;
    assert_eq!(values, expected);
    assert!(
        unnested * 10 < whole,
        "leaf of unnested elements read {unnested} bytes, whole read {whole}"
    );

    ctx.sql("SET datafusion.execution.parquet.pushdown_filters = true")
        .await?
        .collect()
        .await?;
    for sql in [
        "SELECT v ->> 'a' FROM t WHERE v ->> 'a' = '7'",
        "SELECT s['v'] ->> 'a' FROM t WHERE s['v'] ->> 'a' = '7'",
    ] {
        let (values, filtered) = run(&ctx, sql).await?;
        assert_eq!(values, vec![Some("7".to_owned())], "{sql}");
        assert!(
            filtered * 10 < whole,
            "{sql}: read {filtered} bytes, whole read {whole}"
        );
    }
    Ok(())
}
