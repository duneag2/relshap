import duckdb
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
    description="make_tpch"
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

SF = 1

TABLES = ["customer","orders","lineitem","part","partsupp","supplier","nation","region"]


if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(DB_PATH)

# 1) Generate raw TPC-H tables
con.execute("INSTALL tpch;")
con.execute("LOAD tpch;")

# for t in TABLES:
#     con.execute(f"DROP TABLE IF EXISTS {t};")
#     con.execute(f"DROP TABLE IF EXISTS {t}_raw;")

con.execute(f"CALL dbgen(sf = {SF});")

# (참고) dbgen이 만든 건 보통 NOT NULL 정도만 있음
# print("Existing constraints (raw):",
#       con.execute("SELECT constraint_type, table_name FROM duckdb_constraints() ORDER BY table_name, constraint_type;").fetchall()
# )

# 2) Rename raw tables to *_raw  (raw에는 FK가 없어서 의존성 문제 없음)
for t in TABLES:
    con.execute(f"ALTER TABLE {t} RENAME TO {t}_raw;")

# 3) Create final tables WITH PK/FK using canonical names
con.execute("""
CREATE TABLE region (
  r_regionkey INTEGER PRIMARY KEY,
  r_name      VARCHAR,
  r_comment   VARCHAR
);

CREATE TABLE nation (
  n_nationkey INTEGER PRIMARY KEY,
  n_name      VARCHAR,
  n_regionkey INTEGER,
  n_comment   VARCHAR,
  FOREIGN KEY (n_regionkey) REFERENCES region(r_regionkey)
);

CREATE TABLE supplier (
  s_suppkey   INTEGER PRIMARY KEY,
  s_name      VARCHAR,
  s_address   VARCHAR,
  s_nationkey INTEGER,
  s_phone     VARCHAR,
  s_acctbal   DECIMAL(15,2),
  s_comment   VARCHAR,
  FOREIGN KEY (s_nationkey) REFERENCES nation(n_nationkey)
);

CREATE TABLE customer (
  c_custkey    INTEGER PRIMARY KEY,
  c_name       VARCHAR,
  c_address    VARCHAR,
  c_nationkey  INTEGER,
  c_phone      VARCHAR,
  c_acctbal    DECIMAL(15,2),
  c_mktsegment VARCHAR,
  c_comment    VARCHAR,
  FOREIGN KEY (c_nationkey) REFERENCES nation(n_nationkey)
);

CREATE TABLE part (
  p_partkey   INTEGER PRIMARY KEY,
  p_name      VARCHAR,
  p_mfgr      VARCHAR,
  p_brand     VARCHAR,
  p_type      VARCHAR,
  p_size      INTEGER,
  p_container VARCHAR,
  p_retailprice DECIMAL(15,2),
  p_comment   VARCHAR
);

CREATE TABLE partsupp (
  ps_partkey  INTEGER,
  ps_suppkey  INTEGER,
  ps_availqty INTEGER,
  ps_supplycost DECIMAL(15,2),
  ps_comment  VARCHAR,
  PRIMARY KEY (ps_partkey, ps_suppkey),
  FOREIGN KEY (ps_partkey) REFERENCES part(p_partkey),
  FOREIGN KEY (ps_suppkey) REFERENCES supplier(s_suppkey)
);

CREATE TABLE orders (
  o_orderkey      INTEGER PRIMARY KEY,
  o_custkey       INTEGER,
  o_orderstatus   VARCHAR,
  o_totalprice    DECIMAL(15,2),
  o_orderdate     DATE,
  o_orderpriority VARCHAR,
  o_clerk         VARCHAR,
  o_shippriority  INTEGER,
  o_comment       VARCHAR,
  FOREIGN KEY (o_custkey) REFERENCES customer(c_custkey)
);

CREATE TABLE lineitem (
  l_orderkey      INTEGER,
  l_partkey       INTEGER,
  l_suppkey       INTEGER,
  l_linenumber    INTEGER,
  l_quantity      DECIMAL(15,2),
  l_extendedprice DECIMAL(15,2),
  l_discount      DECIMAL(15,2),
  l_tax           DECIMAL(15,2),
  l_returnflag    VARCHAR,
  l_linestatus    VARCHAR,
  l_shipdate      DATE,
  l_commitdate    DATE,
  l_receiptdate   DATE,
  l_shipinstruct  VARCHAR,
  l_shipmode      VARCHAR,
  l_comment       VARCHAR,
  PRIMARY KEY (l_orderkey, l_linenumber),
  FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey),
  FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp(ps_partkey, ps_suppkey)
);
""")

# 4) Copy data raw -> final (parents first)
con.execute("INSERT INTO region   SELECT * FROM region_raw;")
con.execute("INSERT INTO nation   SELECT * FROM nation_raw;")
con.execute("INSERT INTO supplier SELECT * FROM supplier_raw;")
con.execute("INSERT INTO customer SELECT * FROM customer_raw;")
con.execute("INSERT INTO part     SELECT * FROM part_raw;")
con.execute("INSERT INTO partsupp SELECT * FROM partsupp_raw;")
con.execute("INSERT INTO orders   SELECT * FROM orders_raw;")
con.execute("INSERT INTO lineitem SELECT * FROM lineitem_raw;")

# 5) Drop raw tables
for t in TABLES:
    con.execute(f"DROP TABLE {t}_raw;")

# 6) Verify: PK/FK exist in schema
print("\nFinal constraints:")
print(con.execute("""
  SELECT table_name, constraint_type, constraint_name, referenced_table
  FROM duckdb_constraints()
  WHERE constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
  ORDER BY table_name, constraint_type, constraint_name;
""").fetchall())

# quick sanity counts
print("\nRow counts:")
for t in ["region","nation","supplier","customer","part","partsupp","orders","lineitem"]:
    n = con.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
    print(f"  {t}: {n}")

with open(SQL_PATH, "r") as f:
    flat_sql = f.read()
flat = con.execute(flat_sql).fetchdf()

late_thr   = flat["late_rate"].quantile(0.75)
return_thr = flat["return_rate"].quantile(0.75)

flat["supplier_risk"] = (
    (flat["late_rate"]   >= late_thr) |
    (flat["return_rate"] >= return_thr)
).astype(int)

flat = flat.drop(columns=["late_rate", "return_rate"])

flat.to_csv(CSV_PATH, index=False)

con.close()
print("\nDone")
