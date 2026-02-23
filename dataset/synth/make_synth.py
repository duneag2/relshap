import os

SEED = 2026
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["SEGMENT_DISABLE"] = "1" 
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import math
import argparse
import duckdb
import pandas as pd

parser = argparse.ArgumentParser(
    description="make_synth"
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

SEED = 7
N_CUSTOMERS = 400 #100 #400
N_ORDERS = 3000 #100 #3000

rng = random.Random(SEED)

regions = ["ASIA", "NAM", "EU"]
channels = ["web", "app", "store"]
tiers = ["basic", "standard", "premium"]
genders = ["F", "M", "X"]

REGION_TO_CURRENCIES = {
    "ASIA": ["KRW", "JPY", "CNY", "TWD"],
    "NAM":   ["USD", "CAD", "MXN"],
    "EU":   ["EUR", "GBP", "CHF"],
}

def rchoice(xs):
    return xs[rng.randrange(len(xs))]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def life_stage_from_age(age: int) -> str:
    if age < 18:
        return "child"
    if age < 65:
        return "adult"
    return "senior"

customers = []
for cid in range(1, N_CUSTOMERS + 1):
    age = rng.randint(10, 80)
    region = rchoice(regions)
    currency = rchoice(REGION_TO_CURRENCIES[region])
    gender = rchoice(genders)
    life_stage = life_stage_from_age(age)
    customers.append((cid, age, gender, region, currency, life_stage))

orders = []
order_items = []

for oid in range(1, N_ORDERS + 1):
    cid = rng.randint(1, N_CUSTOMERS)
    channel = rchoice(channels)
    order_total = round(rng.uniform(10, 500), 2)
    order_code = f"O{oid:06d}"
    orders.append((oid, order_code, cid, channel, order_total))

    n_lines = rng.randint(1, 5)
    for line_no in range(1, n_lines + 1):
        tier = rchoice(tiers)
        unit_price = round(rng.uniform(5, 200), 2)
        qty = rng.randint(1, 5)
        order_items.append((oid, line_no, tier, unit_price, qty))

cust_df = pd.DataFrame(
    customers,
    columns=["customer_id", "age", "gender", "region", "currency", "life_stage"],
)
ord_df = pd.DataFrame(
    orders,
    columns=["order_id", "order_code", "customer_id", "channel", "order_total"],
)
item_df = pd.DataFrame(
    order_items,
    columns=["order_id", "line_no", "tier", "unit_price", "qty"],
)


if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

con = duckdb.connect(DB_PATH)

# con.execute("""
# CREATE TABLE customer (
#   customer_id INTEGER PRIMARY KEY,
#   age INTEGER,
#   gender VARCHAR,
#   region VARCHAR,
#   currency VARCHAR,
#   life_stage VARCHAR
# );
# """)

con.execute("""
CREATE TABLE Customer (
  customer_id INTEGER PRIMARY KEY,
  age INTEGER,
  gender VARCHAR,
  region VARCHAR,
  currency VARCHAR,
  life_stage VARCHAR,
  CONSTRAINT ck_adult_age
    CHECK (life_stage <> 'adult' OR (age BETWEEN 18 AND 64)),
  CONSTRAINT ck_child_age
    CHECK (life_stage <> 'child' OR (age BETWEEN 10 AND 17)),
  CONSTRAINT ck_senior_age
    CHECK (life_stage <> 'senior' OR (age BETWEEN 65 AND 80)),
  CONSTRAINT ck_region_eu
    CHECK (region <> 'EU' OR currency IN ('EUR','CHF','GBP')),
  CONSTRAINT ck_region_asia
    CHECK (region <> 'ASIA' OR currency IN ('TWD','KRW','JPY','CNY')),
  CONSTRAINT ck_region_nam
    CHECK (region <> 'NAM' OR currency IN ('CAD','MXN','USD'))
);
""")

con.execute("""
CREATE TABLE Orders (
  order_id INTEGER PRIMARY KEY,
  order_code VARCHAR UNIQUE,
  customer_id INTEGER,
  channel VARCHAR,
  order_total DOUBLE,
  FOREIGN KEY(customer_id) REFERENCES customer(customer_id)
);
""")

con.execute("""
CREATE TABLE Item (
  order_id INTEGER,
  line_no INTEGER,
  tier VARCHAR,
  unit_price DOUBLE,
  qty INTEGER,
  PRIMARY KEY(order_id, line_no),
  FOREIGN KEY(order_id) REFERENCES orders(order_id)
);
""")

con.register("cust_df", cust_df)
con.register("ord_df", ord_df)
con.register("item_df", item_df)

con.execute("INSERT INTO Customer SELECT * FROM cust_df")
con.execute("INSERT INTO Orders SELECT * FROM ord_df")
con.execute("INSERT INTO Item SELECT * FROM item_df")


with open(SQL_PATH, "r") as f:
    QUERY = f.read()

flat = con.execute(QUERY).df()

rng_y = random.Random(SEED + 999)

# Promotion response target: mostly deterministic, very easy to predict (>= 90% typical)
# Business-ish story: responds when "high intent" basket patterns appear.
def make_y_row(row):
    hi_intent = (
        (row["channel"] == "app" and row["top_tier"] == "premium" and float(row["order_total"]) >= 140.0)
        or (float(row["top_unit_price"]) >= 150.0 and float(row["n_items"]) >= 3.0)
        or (float(row["order_total"]) >= 260.0 and float(row["total_qty"]) >= 8.0)
    )

    # base probability
    p = 0.90 if hi_intent else 0.08

    # -------------------------
    # demographic stochastic nudges
    # (group-wise distributions, tiny magnitude)
    # -------------------------

    # age / life_stage: older -> slightly higher mean
    if row["life_stage"] == "senior":
        mu_age, sigma_age = 0.015, 0.005
    elif row["life_stage"] == "adult":
        mu_age, sigma_age = 0.010, 0.005
    else:
        mu_age, sigma_age = 0.005, 0.005

    p += rng_y.normalvariate(mu_age, sigma_age)

    # gender: very small separation
    if row["gender"] == "F":
        mu_g, sigma_g = 0.004, 0.004
    elif row["gender"] == "M":
        mu_g, sigma_g = 0.000, 0.004
    else:
        mu_g, sigma_g = 0.002, 0.004

    p += rng_y.normalvariate(mu_g, sigma_g)

    # region: different business exposure patterns
    if row["region"] == "ASIA":
        mu_r, sigma_r = 0.004, 0.004
    elif row["region"] == "NAM":
        mu_r, sigma_r = 0.006, 0.004
    else:
        mu_r, sigma_r = 0.002, 0.004

    p += rng_y.normalvariate(mu_r, sigma_r)

    # clamp
    p = max(0.01, min(0.99, p))

    # tiny noise so it's not perfectly deterministic
    if rng_y.random() < 0.02:
        p = 1.0 - p

    return 1 if rng_y.random() < p else 0

flat = flat.sort_values(["customer_id", "order_id"]).reset_index(drop=True)
flat["y"] = flat.apply(make_y_row, axis=1).astype(int)


flat.to_csv(CSV_PATH, index=False)
con.close()

print("DB created:", DB_PATH)
print("Query created:", SQL_PATH)
print("Flattened CSV created:", CSV_PATH)

