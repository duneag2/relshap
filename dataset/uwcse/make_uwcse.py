import mysql.connector
import duckdb
import os
import re
import argparse

parser = argparse.ArgumentParser(
    description="make_uwcse"
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


if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# -----------------------
# MySQL connection (source)
# -----------------------
mysql_con = mysql.connector.connect(
    host="relational.fel.cvut.cz",
    port=3306,
    user="guest",
    password="ctu-relational",
    database="UW_std",
)
mysql_cur = mysql_con.cursor()

# -----------------------
# DuckDB connection (target)
# -----------------------
duck = duckdb.connect(DB_PATH)

# -----------------------
# DDL normalizer (DuckDB-safe, FK 유지)
# -----------------------
def normalize_mysql_ddl(sql: str) -> str:
    sql = sql.replace("`", "")
    sql = sql.replace("ENGINE=InnoDB", "")
    sql = sql.replace("AUTO_INCREMENT", "")

    # table options
    sql = re.sub(r"DEFAULT CHARSET=\w+", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"CHARACTER SET\s+\w+", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"COLLATE\s*=\s*\w+", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"COLLATE\s+\w+", "", sql, flags=re.IGNORECASE)

    # DEFAULT
    sql = re.sub(r"DEFAULT\s+[^,\n]+", "", sql, flags=re.IGNORECASE)

    # MySQL secondary indexes
    sql = re.sub(r"\n\s*KEY\s+[^\n,]+,?", "", sql, flags=re.IGNORECASE)

    # FK actions (DuckDB does not support CASCADE)
    sql = re.sub(r"ON DELETE CASCADE", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"ON UPDATE CASCADE", "", sql, flags=re.IGNORECASE)

    # type widths
    sql = re.sub(r"\bint\(\d+\)", "INTEGER", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\btinyint\(\d+\)", "INTEGER", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bsmallint\(\d+\)", "INTEGER", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bbigint\(\d+\)", "BIGINT", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bvarchar\(\d+\)", "VARCHAR", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bchar\(\d+\)", "VARCHAR", sql, flags=re.IGNORECASE)

    return sql.strip()

# -----------------------
# Explicit creation order (from ERD)
# -----------------------
create_order = [
    "person",
    "course",
    "advisedBy",
    "taughtBy",
]

# -----------------------
# CREATE TABLE (FK 포함)
# -----------------------
for t in create_order:
    mysql_cur.execute(f"SHOW CREATE TABLE {t}")
    create_sql = mysql_cur.fetchone()[1]

    ddl = normalize_mysql_ddl(create_sql)

    try:
        duck.execute(ddl)
    except Exception:
        print("\nORIGINAL:\n", create_sql)
        print("\nNORMALIZED:\n", ddl)
        raise

# -----------------------
# INSERT DATA
# -----------------------
for t in create_order:
    mysql_cur.execute(f"SELECT * FROM {t}")
    rows = mysql_cur.fetchall()

    if not rows:
        continue

    placeholders = ",".join(["?"] * len(rows[0]))
    duck.executemany(
        f"INSERT INTO {t} VALUES ({placeholders})",
        rows,
    )

# -----------------------
# Read flattened SELECT
# -----------------------
with open(SQL_PATH, "r") as f:
    flat_sql = f.read()

# -----------------------
# Execute SELECT → DataFrame
# -----------------------
flat = duck.execute(flat_sql).fetchdf()
valid_phases = {"Post_Quals", "Post_Generals", "Pre_Quals"}
flat = flat[flat["inPhase"].isin(valid_phases)]
flat.to_csv(CSV_PATH, index=False)

# -----------------------
# Cleanup
# -----------------------
mysql_con.close()
duck.close()



print("Done")
