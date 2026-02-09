import os
import openml
import duckdb
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="make_churn_modeling"
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

# -----------------------
# Load CSV
# -----------------------

df = pd.read_csv(RAW_CSV_PATH, index_col=False)
y = df["Exited"].to_numpy()          # label은 DB 밖
df_nolabel = df.drop(columns=["Exited"])

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

con = duckdb.connect(DB_PATH)


# -----------------------
# RAW
# -----------------------
con.register("raw_df", df_nolabel)
con.execute("DROP TABLE IF EXISTS raw_churn_modeling")
con.execute("CREATE TABLE raw_churn_modeling AS SELECT * FROM raw_df")

# -----------------------
# CUSTOMER
# -----------------------
con.execute("""
DROP TABLE IF EXISTS Customer;
CREATE TABLE Customer (
  CustomerId INTEGER PRIMARY KEY,
  Surname VARCHAR,
  Geography VARCHAR,
  Gender VARCHAR,
  Age INTEGER
);
""")

con.execute("""
INSERT INTO Customer
SELECT DISTINCT
  CustomerId,
  Surname,
  Geography,
  Gender,
  Age
FROM raw_churn_modeling;
""")

# -----------------------
# PROFILE
# -----------------------
con.execute("""
DROP TABLE IF EXISTS Profile;
CREATE TABLE Profile (
  RowNumber INTEGER PRIMARY KEY,
  CustomerId INTEGER UNIQUE,
  CreditScore INTEGER,
  Tenure INTEGER,
  Balance DOUBLE,
  NumOfProducts INTEGER,
  HasCrCard INTEGER,
  IsActiveMember INTEGER,
  EstimatedSalary DOUBLE,
  FOREIGN KEY (CustomerId) REFERENCES Customer(CustomerId)
);
""")

con.execute("""
INSERT INTO Profile
SELECT
  RowNumber,
  CustomerId,
  CreditScore,
  Tenure,
  Balance,
  NumOfProducts,
  HasCrCard,
  IsActiveMember,
  EstimatedSalary
FROM raw_churn_modeling;
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
# Attach Exited
# -----------------------

df_sorted = df.sort_values("RowNumber").reset_index(drop=True)
flat_sorted = flat.sort_values("RowNumber").reset_index(drop=True)

flat_sorted["Exited"] = df_sorted["Exited"].values

for col in flat_sorted.columns:
    df_sorted[col] = df_sorted[col].astype(flat_sorted[col].dtype)

assert df_sorted.equals(flat_sorted), "FINAL CHECK FAILED (with Exited)"

print("FINAL VERIFICATION PASSED (including Exited)")

flat_final = (
    flat_sorted
    .set_index("RowNumber")
    .loc[df["RowNumber"].values]
    .reset_index()
)

flat_final = flat_final[df.columns]

assert flat_final.shape == df.shape

flat_final.to_csv(CSV_PATH, index=False)

print("Saved flattened.csv with original row + column order")