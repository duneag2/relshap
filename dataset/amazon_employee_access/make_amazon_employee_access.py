import os
import openml
import duckdb
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="make_amazon_employee_access"
)

parser.add_argument(
    "--base-dir",
    type=str,
    required=True,
    help="data directory",
)

parser.add_argument(
    "--db",
    type=str,
    required=True,
    help="duckdb file",
)

parser.add_argument(
    "--query",
    type=str,
    required=True,
    help="query file",
)

parser.add_argument(
    "--flattened",
    type=str,
    required=True,
    help="flattened csv file",
)

args = parser.parse_args()
DB_PATH = os.path.join(args.base_dir, args.db)
SQL_PATH = os.path.join(args.base_dir, args.query)
CSV_PATH = os.path.join(args.base_dir, args.flattened)
RAW_CSV_PATH = os.path.join(args.base_dir, "original_data.csv")


dataset_id = 43900

LABEL_COL = "ACTION"  # target

# -----------------------
# Load from OpenML
# -----------------------
dataset = openml.datasets.get_dataset(dataset_id)
df, _, _, _ = dataset.get_data(dataset_format="dataframe")
df.to_csv(RAW_CSV_PATH, index=False)

# label은 DB 밖
df_nolabel = df.drop(columns=[LABEL_COL])

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

con = duckdb.connect(DB_PATH)

# -----------------------
# RAW (row_id를 우리가 만든다: 원본 row 순서 고정)
# -----------------------
con.register("raw_df", df_nolabel)
con.execute("DROP TABLE IF EXISTS raw_amazon_employee_access;")
con.execute("""
CREATE TABLE raw_amazon_employee_access AS
SELECT
  row_number() OVER () - 1 AS row_id,   -- 0..n-1 (df index와 맞추기 좋게)
  *
FROM raw_df;
""")

# -----------------------
# ROLE (최소 정규화: Role만 분리)
# -----------------------
con.execute("DROP TABLE IF EXISTS Role;")
con.execute("""
CREATE TABLE Role (
  role_code   VARCHAR PRIMARY KEY,
  role_title  VARCHAR NOT NULL UNIQUE,
  role_family VARCHAR
);
""")

con.execute("""
INSERT INTO Role
SELECT DISTINCT
  CAST(ROLE_CODE AS VARCHAR)   AS role_code,
  CAST(ROLE_TITLE AS VARCHAR)  AS role_title,
  CAST(ROLE_FAMILY AS VARCHAR) AS role_family
FROM raw_amazon_employee_access
WHERE ROLE_CODE IS NOT NULL;
""")

# -----------------------
# MAIN (원본 레코드 유지 + role_code만 FK)
# -----------------------
con.execute("DROP TABLE IF EXISTS amazon_employee_access;")
con.execute("""
CREATE TABLE amazon_employee_access (
  row_id   BIGINT PRIMARY KEY,  -- 원본 row 식별자 역할

  resource      VARCHAR,
  mgr_id        VARCHAR,

  role_rollup_1 VARCHAR,
  role_rollup_2 VARCHAR,
  role_deptname VARCHAR,
  role_family_desc VARCHAR,

  role_code     VARCHAR NOT NULL,
  FOREIGN KEY (role_code) REFERENCES Role(role_code)
);
""")

con.execute("""
INSERT INTO amazon_employee_access
SELECT
  row_id,

  CAST(RESOURCE AS VARCHAR)      AS resource,
  CAST(MGR_ID AS VARCHAR)        AS mgr_id,

  CAST(ROLE_ROLLUP_1 AS VARCHAR) AS role_rollup_1,
  CAST(ROLE_ROLLUP_2 AS VARCHAR) AS role_rollup_2,
  CAST(ROLE_DEPTNAME AS VARCHAR) AS role_deptname,
  CAST(ROLE_FAMILY_DESC AS VARCHAR) AS role_family_desc,

  CAST(ROLE_CODE AS VARCHAR)     AS role_code
FROM raw_amazon_employee_access;
""")

# -----------------------
# FLATTEN SQL (네가 말한 "최종 테이블 만드는 쿼리"를 여기서 우리가 생성)
# -----------------------
flat_sql = r"""
SELECT
  a.row_id,

  a.resource      AS RESOURCE,
  a.mgr_id        AS MGR_ID,
  a.role_rollup_1 AS ROLE_ROLLUP_1,
  a.role_rollup_2 AS ROLE_ROLLUP_2,
  a.role_deptname AS ROLE_DEPTNAME,
  a.role_family_desc AS ROLE_FAMILY_DESC,

  r.role_title    AS ROLE_TITLE,
  r.role_family   AS ROLE_FAMILY,
  r.role_code     AS ROLE_CODE
FROM amazon_employee_access a
JOIN Role r
  ON a.role_code = r.role_code
ORDER BY a.row_id;
"""

with open(SQL_PATH, "w") as f:
    f.write(flat_sql)

# -----------------------
# Execute flatten
# -----------------------
flat = con.execute(flat_sql).fetchdf()

# -----------------------
# Attach label (ACTION) + 원본 row/col 순서 복원
# -----------------------
# row_id = df index와 맞춰놨기 때문에, df의 원래 순서대로 안전하게 붙일 수 있음
flat["ACTION"] = df[LABEL_COL].to_numpy()

flat_final = (
    flat
    .set_index("row_id")
    .loc[np.arange(len(df))]
    .reset_index(drop=False)
)

cols_df = list(df.columns)
flat_check = flat_final[cols_df]

for col in cols_df:
    df[col] = df[col].astype(flat_check[col].dtype)

assert df.equals(flat_check), "FINAL CHECK FAILED (with class)"

flat_save = flat_final[["row_id"] + cols_df]
flat_save.to_csv(CSV_PATH, index=False)
print("Saved (with row_id)")
