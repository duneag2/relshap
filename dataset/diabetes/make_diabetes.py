import os
import openml
import duckdb
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="make_diabetes"
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

dataset_id = 37

dataset = openml.datasets.get_dataset(dataset_id)

df, _, _, _ = dataset.get_data(
    dataset_format="dataframe"
)

# row_id (0..n-1) 추가
df = df.reset_index(drop=True)
df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))

# 원본 저장 (row_id + class 포함)
df.to_csv(RAW_CSV_PATH, index=False)

# label 분리(원하면 DB 밖으로): class는 DB에 안 넣음
if "class" not in df.columns:
    raise KeyError("Expected column 'class' not found in OpenML dataframe.")
df_nolabel = df.drop(columns=["class"])

# DB 새로 만들기
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
con = duckdb.connect(DB_PATH)

# 단일 테이블 생성
con.register("pima_df", df_nolabel)
con.execute("DROP TABLE IF EXISTS Pima;")
con.execute("CREATE TABLE Pima AS SELECT * FROM pima_df;")
con.execute("ALTER TABLE Pima ADD PRIMARY KEY (row_id);")

# -----------------------
# Read flattened SELECT
# -----------------------
with open(SQL_PATH, "r") as f:
    flat_sql = f.read()

# -----------------------
# Execute SELECT → DataFrame
# -----------------------
flat = con.execute(flat_sql).fetchdf()

# -----------------------
# Attach class (row_id로 정렬해서 안전하게)
# -----------------------
df_sorted = df.sort_values("row_id").reset_index(drop=True)
flat_sorted = flat.sort_values("row_id").reset_index(drop=True)

# SELECT 결과에 row_id가 반드시 있어야 함
if "row_id" not in flat_sorted.columns:
    raise KeyError("Your SQL must SELECT row_id (e.g., SELECT * FROM Pima).")

flat_sorted["class"] = df_sorted["class"].values

# (선택) 완전 일치 검증: SQL이 'SELECT * FROM Pima'면 원본과 동일해짐(단 class 제외였던 걸 다시 붙였으니 동일)
# dtype 맞추기
for col in flat_sorted.columns:
    if col in df_sorted.columns:
        df_sorted[col] = df_sorted[col].astype(flat_sorted[col].dtype, copy=False)

assert df_sorted.equals(flat_sorted), "FINAL CHECK FAILED (including class)"
print("FINAL VERIFICATION PASSED (including class)")

# 최종 flattened 저장: 원본(original_data.csv)와 동일한 컬럼 순서 유지
flat_final = flat_sorted[df.columns]
flat_final.to_csv(CSV_PATH, index=False)

print("Saved flattened.csv")
print("Saved original_data.csv")
print("Saved duckdb:", DB_PATH)