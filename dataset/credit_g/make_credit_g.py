import os
import openml
import duckdb
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="make_credit_g"
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

dataset_id = 31

EXPECTED_COLS = [
    "checking_status","duration","credit_history","purpose","credit_amount","savings_status",
    "employment","installment_commitment","personal_status","other_parties","residence_since",
    "property_magnitude","age","other_payment_plans","housing","existing_credits","job",
    "num_dependents","own_telephone","foreign_worker","class"
]

# Applicant / Application semantic split (Option A)
APPLICANT_COLS = [
    "personal_status",
    "age",
    "job",
    "housing",
    "foreign_worker",
    "own_telephone",
    "num_dependents",
    "other_parties",
]

APPLICATION_COLS = [
    "checking_status",
    "credit_history",
    "purpose",
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "existing_credits",
    "property_magnitude",
    "savings_status",
    "employment",
    "other_payment_plans",
]


# -----------------------
# Load from OpenML -> CSV
# -----------------------
dataset = openml.datasets.get_dataset(dataset_id)

df, _, _, _ = dataset.get_data(
    dataset_format="dataframe"
)

df.to_csv(RAW_CSV_PATH, index=False)

# -----------------------
# Load CSV
# -----------------------
# label은 DB 밖
if list(df.columns) != EXPECTED_COLS:
    raise ValueError(
        f"Unexpected columns.\nExpected: {EXPECTED_COLS}\nGot: {list(df.columns)}"
    )

df_nolabel = df.drop(columns=["class"])

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

con = duckdb.connect(DB_PATH)
con.execute("PRAGMA disable_progress_bar;")


# -----------------------
# RAW (with application_id)
# -----------------------
con.register("raw_df", df_nolabel)
con.execute("DROP TABLE IF EXISTS raw_credit_g")
con.execute("""
CREATE TABLE raw_credit_g AS
SELECT
  row_number() OVER () AS application_id,
  *
FROM raw_df;
""")

# -----------------------
# APPLICANT
# -----------------------
con.execute("""
DROP TABLE IF EXISTS Applicant;
CREATE TABLE Applicant (
  applicant_id INTEGER PRIMARY KEY,

  personal_status VARCHAR,
  age INTEGER,
  job VARCHAR,
  housing VARCHAR,
  foreign_worker VARCHAR,
  own_telephone VARCHAR,
  num_dependents INTEGER,
  other_parties VARCHAR
);
""")

con.execute(f"""
INSERT INTO Applicant
SELECT
  application_id AS applicant_id,

  personal_status::VARCHAR,
  age::INTEGER,
  job::VARCHAR,
  housing::VARCHAR,
  foreign_worker::VARCHAR,
  own_telephone::VARCHAR,
  num_dependents::INTEGER,
  other_parties::VARCHAR
FROM raw_credit_g
ORDER BY application_id;
""")

# -----------------------
# APPLICATION
# -----------------------
con.execute("""
DROP TABLE IF EXISTS Application;
CREATE TABLE Application (
  application_id INTEGER PRIMARY KEY,
  applicant_id INTEGER NOT NULL UNIQUE,

  checking_status VARCHAR,
  credit_history VARCHAR,
  purpose VARCHAR,

  duration INTEGER,
  credit_amount DOUBLE,
  installment_commitment INTEGER,
  residence_since INTEGER,
  existing_credits INTEGER,

  property_magnitude VARCHAR,
  savings_status VARCHAR,
  employment VARCHAR,
  other_payment_plans VARCHAR,

  FOREIGN KEY (applicant_id) REFERENCES Applicant(applicant_id)
);
""")

con.execute("""
INSERT INTO Application
SELECT
  r.application_id,
  r.application_id AS applicant_id,

  r.checking_status::VARCHAR,
  r.credit_history::VARCHAR,
  r.purpose::VARCHAR,

  r.duration::INTEGER,
  r.credit_amount::DOUBLE,
  r.installment_commitment::INTEGER,
  r.residence_since::INTEGER,
  r.existing_credits::INTEGER,

  r.property_magnitude::VARCHAR,
  r.savings_status::VARCHAR,
  r.employment::VARCHAR,
  r.other_payment_plans::VARCHAR
FROM raw_credit_g r
ORDER BY r.application_id;
""")

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
# Attach class
# -----------------------
# churn은 phone_number로 정렬했지만, credit_g는 자연키 없으니까 application_id 정렬/복구
if "application_id" not in flat.columns:
    raise ValueError("flatten SQL must include application_id for verification/order restoration.")

df_sorted = df.copy().reset_index(drop=True)  # 원본 row order가 application_id 1..N과 동일
flat_sorted = flat.sort_values("application_id").reset_index(drop=True)

flat_sorted["class"] = df_sorted["class"].values

# dtype 맞추기 (churn과 동일)
# 주의: df_sorted에는 application_id 없으니, 비교 전 flat에서 application_id drop
flat_noid = flat_sorted.drop(columns=["application_id"])

for col in flat_noid.columns:
    df_sorted[col] = df_sorted[col].astype(flat_noid[col].dtype)

assert df_sorted.equals(flat_noid), "FINAL CHECK FAILED (with class)"

print("FINAL VERIFICATION PASSED (including class)")

# -----------------------
# Save flattened.csv with original row + column order
# -----------------------
# 이미 application_id 기준 정렬 상태 = 원본 row order
flat_final = flat_noid[df.columns]
assert flat_final.shape == df.shape

flat_save = flat_sorted[["application_id"]].join(flat_final)
flat_save.to_csv(CSV_PATH, index=False)

print("Saved flattened.csv with original row + column order")

con.close()
