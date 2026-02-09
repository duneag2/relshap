import os
import openml
import duckdb
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="make_churn"
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

dataset_id = 40701

dataset = openml.datasets.get_dataset(dataset_id)

df, _, _, _ = dataset.get_data(
    dataset_format="dataframe"
)

df.to_csv(RAW_CSV_PATH, index=False)

# -----------------------
# Load CSV
# -----------------------

# y = df["class"].to_numpy()          # label은 DB 밖
df_nolabel = df.drop(columns=["class"])

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

con = duckdb.connect(DB_PATH)


# -----------------------
# RAW
# -----------------------
con.register("raw_df", df_nolabel)
con.execute("DROP TABLE IF EXISTS raw_churn")
con.execute("CREATE TABLE raw_churn AS SELECT * FROM raw_df")

# -----------------------
# CUSTOMER
# -----------------------
con.execute("""
DROP TABLE IF EXISTS Customer;
CREATE TABLE Customer (
  customer_id INTEGER PRIMARY KEY,
  phone_number DOUBLE UNIQUE,
  state INTEGER,
  account_length INTEGER,
  area_code INTEGER,
  international_plan INTEGER,
  voice_mail_plan INTEGER,
  number_vmail_messages INTEGER,
  number_customer_service_calls INTEGER
);
""")

con.execute("""
INSERT INTO Customer
SELECT
  row_number() OVER (ORDER BY phone_number) AS customer_id,
  phone_number,
  state,
  account_length,
  area_code,
  international_plan,
  voice_mail_plan,
  number_vmail_messages,
  number_customer_service_calls
FROM raw_churn;
""")

# -----------------------
# USAGE
# -----------------------
con.execute("""
DROP TABLE IF EXISTS Usage;
CREATE TABLE Usage (
  customer_id INTEGER,
  period VARCHAR,
  total_minutes DOUBLE,
  total_calls INTEGER,
  total_charge DOUBLE,
  PRIMARY KEY (customer_id, period),
  FOREIGN KEY (customer_id) REFERENCES Customer(customer_id)
);
""")

con.execute("""
INSERT INTO Usage
SELECT c.customer_id, 'DAY',
       r.total_day_minutes, r.total_day_calls, r.total_day_charge
FROM raw_churn r JOIN Customer c USING (phone_number)
UNION ALL
SELECT c.customer_id, 'EVE',
       r.total_eve_minutes, r.total_eve_calls, r.total_eve_charge
FROM raw_churn r JOIN Customer c USING (phone_number)
UNION ALL
SELECT c.customer_id, 'NIGHT',
       r.total_night_minutes, r.total_night_calls, r.total_night_charge
FROM raw_churn r JOIN Customer c USING (phone_number)
UNION ALL
SELECT c.customer_id, 'INTL',
       r.total_intl_minutes, r.total_intl_calls, r.total_intl_charge
FROM raw_churn r JOIN Customer c USING (phone_number);
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

df_sorted = df.sort_values("phone_number").reset_index(drop=True)
flat_sorted = flat.sort_values("phone_number").reset_index(drop=True)

flat_sorted["class"] = df_sorted["class"].values

for col in flat_sorted.columns:
    df_sorted[col] = df_sorted[col].astype(flat_sorted[col].dtype)

assert df_sorted.equals(flat_sorted), "FINAL CHECK FAILED (with class)"

print("FINAL VERIFICATION PASSED (including class)")

flat_final = (
    flat_sorted
    .set_index("phone_number")
    .loc[df["phone_number"].values]
    .reset_index()
)

flat_final = flat_final[df.columns]

assert flat_final.shape == df.shape

flat_final.to_csv(CSV_PATH, index=False)

print("Saved flattened.csv with original row + column order")



# print(df_sorted.head(-5))
# print(flat_sorted.head(-5))

# diff_mask = (df_sorted != flat_sorted) & ~(df_sorted.isna() & flat_sorted.isna())

# # 다른 셀의 (row, col) 위치
# diff_locs = diff_mask.stack()
# diff_locs = diff_locs[diff_locs]

# print(f"Number of differing cells: {len(diff_locs)}")
# print(diff_locs.head(20))

# for (row, col) in diff_locs.index[:20]:
#     v1 = df_sorted.loc[row, col]
#     v2 = flat_sorted.loc[row, col]
#     print(f"[row={row}, col={col}] raw={v1!r}, flat={v2!r}")