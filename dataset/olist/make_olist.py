import os
from pathlib import Path
import itertools
import duckdb
import argparse

parser = argparse.ArgumentParser(
    description="make_olist"
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

CSV_DIR = Path(os.path.join(args.base_dir, "olist"))
# CSV_DIR = Path(os.path.join('/Users/seungeun/nyu/relshap2026/copy_safe_feb92pm/dataset/olist', "olist"))
DB_PATH = os.path.join(args.base_dir, args.db)
SQL_PATH = os.path.join(args.base_dir, args.query)
CSV_PATH = os.path.join(args.base_dir, args.flattened)

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# 파일명(원본 CSV)
FILES = {
    "olist_orders_dataset": "olist_orders_dataset.csv",
    "olist_order_items_dataset": "olist_order_items_dataset.csv",
    "olist_order_payments_dataset": "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset": "olist_order_reviews_dataset.csv",
    "olist_products_dataset": "olist_products_dataset.csv",
    "olist_sellers_dataset": "olist_sellers_dataset.csv",
    # 그림에서 "olist_order_customer_dataset" = customers csv
    "olist_order_customer_dataset": "olist_customers_dataset.csv",
    "olist_geolocation_dataset": "olist_geolocation_dataset.csv",
    # (그림엔 없지만 보통 같이 둠)
    "product_category_name_translation": "product_category_name_translation.csv",
}

# ====== 유틸 ======
def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def qtable(name: str) -> str:
    return qident(name)

def read_csv_view(con, view_name: str, csv_path: Path) -> None:
    # con.execute(f"DROP VIEW IF EXISTS {qident(view_name)};")
    con.execute(
        f"""
        CREATE VIEW {qident(view_name)} AS
        SELECT * FROM read_csv_auto({repr(str(csv_path))}, header=True);
        """
    )

def get_columns(con, view_name: str):
    desc = con.execute(f"DESCRIBE SELECT * FROM {qident(view_name)};").fetchall()
    # desc row: (col_name, col_type, null, key, default, extra)
    cols = [(r[0], r[1]) for r in desc]
    return cols

def rowcount(con, view_name: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {qident(view_name)};").fetchone()[0]

def count_nulls(con, view_name: str, col: str) -> int:
    return con.execute(
        f"SELECT COUNT(*) FROM {qident(view_name)} WHERE {qident(col)} IS NULL;"
    ).fetchone()[0]

def distinct_count(con, view_name: str, col: str) -> int:
    return con.execute(
        f"SELECT COUNT(DISTINCT {qident(col)}) FROM {qident(view_name)};"
    ).fetchone()[0]

def distinct_count_combo(con, view_name: str, cols: list[str]) -> int:
    # struct_pack으로 DISTINCT 안전하게 계산
    pack_items = ", ".join(f"{c}:={qident(c)}" for c in cols)
    q = f"""
    SELECT COUNT(DISTINCT struct_pack({pack_items}))
    FROM {qident(view_name)};
    """
    return con.execute(q).fetchone()[0]

def null_any_in_cols(con, view_name: str, cols: list[str]) -> int:
    cond = " OR ".join(f"{qident(c)} IS NULL" for c in cols)
    q = f"SELECT COUNT(*) FROM {qident(view_name)} WHERE {cond};"
    return con.execute(q).fetchone()[0]

def candidate_key_search(con, view_name: str, cols: list[str], max_combo_cols=3, max_id_cols=6):
    """
    1) 단일 컬럼 후보: distinct == n AND null==0
    2) 없으면, id-like 컬럼들에서 2~3개 조합으로 후보 탐색
    """
    n = rowcount(con, view_name)

    # 1) 단일
    singles = []
    for c, _t in cols:
        n_null = count_nulls(con, view_name, c)
        if n_null != 0:
            continue
        d = distinct_count(con, view_name, c)
        if d == n:
            singles.append([c])

    # 2) 조합(필요할 때만)
    combos = []
    if not singles:
        # id-like 컬럼 우선
        id_like = []
        for c, _t in cols:
            lc = c.lower()
            if lc.endswith("_id") or lc in ("order_id", "customer_id", "product_id", "seller_id", "review_id"):
                id_like.append(c)
        # 너무 많으면 자르기
        id_like = id_like[:max_id_cols]

        # 그래도 없으면 prefix 같은 것도 후보에 포함(특히 translation/geo)
        if not id_like:
            for c, _t in cols:
                lc = c.lower()
                if "zip" in lc or "prefix" in lc or "category" in lc or "name" in lc:
                    id_like.append(c)
            id_like = id_like[:max_id_cols]

        # 2~max_combo_cols
        for k in range(2, max_combo_cols + 1):
            for comb in itertools.combinations(id_like, k):
                comb = list(comb)
                # NULL 포함되면 PK 불가(원칙)
                if null_any_in_cols(con, view_name, comb) != 0:
                    continue
                d = distinct_count_combo(con, view_name, comb)
                if d == n:
                    combos.append(comb)

    return n, singles, combos

def choose_pk(table: str, singles: list[list[str]], combos: list[list[str]]):
    """
    자동 선택 규칙:
    - 선호 PK(관행) 있으면 그걸 우선
    - 그 외: 단일 후보 있으면 첫 번째, 없으면 조합 후보 첫 번째
    """
    preferred = {
        "olist_orders_dataset": ["order_id"],
        "olist_products_dataset": ["product_id"],
        "olist_sellers_dataset": ["seller_id"],
        "olist_order_customer_dataset": ["customer_id"],
        "olist_order_items_dataset": ["order_id", "order_item_id"],
        "olist_order_payments_dataset": ["order_id", "payment_sequential"],
        "olist_order_reviews_dataset": ["review_id"],
        "product_category_name_translation": ["product_category_name"],
        # geolocation은 원본에서 자연 PK 없음(대개 중복)
        "olist_geolocation_dataset": [],
    }

    pref = preferred.get(table, None)
    if pref:
        if pref in singles:
            return pref
        if pref in combos:
            return pref

    if singles:
        return singles[0]
    if combos:
        return combos[0]
    return []  # PK 없음

def create_table_from_view(con, table: str, view: str, pk_cols: list[str], fks: list[tuple[str,str,str]]):
    cols = get_columns(con, view)
    col_defs = [f"{qident(c)} {t}" for (c, t) in cols]

    constraints = []
    if pk_cols:
        constraints.append(
            "CONSTRAINT " + qident(f"pk__{table}") +
            " PRIMARY KEY (" + ", ".join(qident(c) for c in pk_cols) + ")"
        )

    for (child_col, parent_table, parent_col) in fks:
        constraints.append(
            "CONSTRAINT " + qident(f"fk__{table}__{child_col}__{parent_table}") +
            f" FOREIGN KEY ({qident(child_col)}) REFERENCES {qtable(parent_table)} ({qident(parent_col)})"
        )

    ddl = f"CREATE TABLE {qtable(table)} (\n  " + ",\n  ".join(col_defs + constraints) + "\n);"
    con.execute(ddl)
    con.execute(f"INSERT INTO {qtable(table)} SELECT * FROM {qident(view)};")

def main():
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # 존재 확인
    for t, fn in FILES.items():
        p = CSV_DIR / fn
        if not p.exists():
            raise FileNotFoundError(f"Missing CSV for {t}: {p}")

    con = duckdb.connect(str(DB_PATH))

    # ====== 0) CSV -> VIEW ======
    for t, fn in FILES.items():
        read_csv_view(con, f"v__{t}", CSV_DIR / fn)

    # ====== 1) PK 후보 계산 & 확정(출력) ======
    pk_map = {}
    print("=== PK CANDIDATE SEARCH ===")
    for t in FILES.keys():
        view = f"v__{t}"
        cols = get_columns(con, view)
        n, singles, combos = candidate_key_search(con, view, cols)

        pk = choose_pk(t, singles, combos)
        pk_map[t] = pk

        print(f"\n[{t}] rows={n}")
        if singles:
            print(f"  single-col candidates: {singles[:10]}{' ...' if len(singles) > 10 else ''}")
        else:
            print("  single-col candidates: []")

        if combos:
            print(f"  combo candidates: {combos[:10]}{' ...' if len(combos) > 10 else ''}")
        else:
            print("  combo candidates: []")

        print(f"  ==> CHOSEN PK: {pk if pk else '(none)'}")

    # geolocation은 (대개) PK 없음으로 두는 게 맞음
    # (자동 탐색이 우연히 단일 유니크를 잡더라도, 원본 성격상 PK로 두지 않는 쪽이 안전)
    pk_map["olist_geolocation_dataset"] = []

    # ====== 2) DROP TABLE (자식부터) ======
    # 파생 테이블까지 포함
    # all_tables_to_drop = list(FILES.keys()) + ["olist_zip_code_prefix"]
    # for t in reversed(all_tables_to_drop):
        # con.execute(f"DROP TABLE IF EXISTS {qtable(t)};")

    # ====== 3) FK 설계 (그림 기반) ======
    # NOTE:
    # - customers/sellers -> geolocation(zip_prefix) 는 geolocation 쪽이 유니크가 아니라 직접 FK 불가할 수 있음
    # - 그래서 zip_prefix 마스터 테이블을 파생해서 거기에 FK를 건다.
    #
    # 파생 테이블: olist_zip_code_prefix(zip_code_prefix PK)
    #   sources: geolocation_zip_code_prefix, customer_zip_code_prefix, seller_zip_code_prefix
    #
    # 그 외 FK는 그림 그대로:
    #   orders.customer_id -> order_customer.customer_id
    #   order_items.order_id -> orders.order_id
    #   order_items.product_id -> products.product_id
    #   order_items.seller_id -> sellers.seller_id
    #   payments.order_id -> orders.order_id
    #   reviews.order_id -> orders.order_id

    # ====== 4) CREATE 원본 CSV 테이블들 (FK는 zip_prefix 제외하고 먼저 만들기) ======
    # 생성 순서: 부모 먼저
    creation_order = [
        "olist_geolocation_dataset",
        "olist_order_customer_dataset",
        "olist_sellers_dataset",
        "olist_products_dataset",
        "olist_orders_dataset",
        "olist_order_items_dataset",
        "olist_order_payments_dataset",
        "olist_order_reviews_dataset",
        "product_category_name_translation",
    ]

    # 먼저 geolocation, customers, sellers, products는 FK 없이 생성
    # (zip_prefix FK는 파생 테이블 만든 뒤에 걸어야 하므로)
    fk_map = {t: [] for t in FILES.keys()}

    # 그림 FK
    fk_map["olist_orders_dataset"] = [
        ("customer_id", "olist_order_customer_dataset", "customer_id"),
    ]
    fk_map["olist_order_items_dataset"] = [
        ("order_id", "olist_orders_dataset", "order_id"),
        ("product_id", "olist_products_dataset", "product_id"),
        ("seller_id", "olist_sellers_dataset", "seller_id"),
    ]
    fk_map["olist_order_payments_dataset"] = [
        ("order_id", "olist_orders_dataset", "order_id"),
    ]
    fk_map["olist_order_reviews_dataset"] = [
        ("order_id", "olist_orders_dataset", "order_id"),
    ]

    # 일단 zip_prefix FK는 나중에(ALTER는 DuckDB에서 제약 추가 제한 있을 수 있어)
    # -> 그래서 customers/sellers는 zip_prefix 컬럼 타입만 통일해서 저장하고,
    #    zip_prefix 제약은 "검증 쿼리"로 대신하거나,
    #    아예 customers/sellers를 zip_prefix FK 포함해서 '재생성'하는 방식이 가장 확실.
    #
    # 여기서는: zip_prefix 부모 테이블을 먼저 만들고,
    # customers/sellers 생성할 때부터 FK 포함해서 한 번에 만들게 구성한다.
    #
    # 따라서: creation_order를 zip_prefix 포함 순서로 재정렬한다.

    # ====== 5) 파생 테이블 olist_zip_code_prefix 만들기 위한 VIEW 기반 추출 ======
    # zip_prefix를 안전하게 VARCHAR로 통일
    # con.execute(f"DROP TABLE IF EXISTS {qtable('olist_zip_code_prefix')};")
    con.execute(
        f"""
        CREATE TABLE {qtable('olist_zip_code_prefix')} (
          {qident('zip_code_prefix')} VARCHAR,
          CONSTRAINT {qident('pk__olist_zip_code_prefix')}
            PRIMARY KEY ({qident('zip_code_prefix')})
        );
        """
    )

    union_sql = f"""
    SELECT DISTINCT TRIM(CAST(geolocation_zip_code_prefix AS VARCHAR)) AS z
    FROM {qident('v__olist_geolocation_dataset')}
    WHERE geolocation_zip_code_prefix IS NOT NULL
    UNION
    SELECT DISTINCT TRIM(CAST(customer_zip_code_prefix AS VARCHAR)) AS z
    FROM {qident('v__olist_order_customer_dataset')}
    WHERE customer_zip_code_prefix IS NOT NULL
    UNION
    SELECT DISTINCT TRIM(CAST(seller_zip_code_prefix AS VARCHAR)) AS z
    FROM {qident('v__olist_sellers_dataset')}
    WHERE seller_zip_code_prefix IS NOT NULL
    """

    con.execute(
        f"""
        INSERT INTO {qtable('olist_zip_code_prefix')}
        SELECT z
        FROM ({union_sql})
        WHERE z IS NOT NULL AND z <> '';
        """
    )

    # ====== 6) 이제 원본 테이블 생성 (customers/sellers에 zip_prefix FK 포함) ======
    # customers/sellers FK 추가
    fk_map["olist_order_customer_dataset"] = [
        ("customer_zip_code_prefix", "olist_zip_code_prefix", "zip_code_prefix"),
    ]
    fk_map["olist_sellers_dataset"] = [
        ("seller_zip_code_prefix", "olist_zip_code_prefix", "zip_code_prefix"),
    ]

    # 실제 생성 순서(부모 먼저)
    creation_order = [
        "olist_geolocation_dataset",           # csv table
        "olist_zip_code_prefix",               # derived parent
        "olist_order_customer_dataset",        # csv table + FK to zip_prefix
        "olist_sellers_dataset",               # csv table + FK to zip_prefix
        "olist_products_dataset",              # csv table
        "olist_orders_dataset",                # csv table + FK to customers
        "olist_order_items_dataset",           # csv table + FK to orders/products/sellers
        "olist_order_payments_dataset",        # csv table + FK to orders
        "olist_order_reviews_dataset",         # csv table + FK to orders
        "product_category_name_translation",   # csv table
    ]

    # DROP & recreate (since we already created zip_prefix table)
    # (지금은 zip_prefix만 만들어놓은 상태라, 나머지를 만드는 동안 문제 없음)

    # geolocation 먼저
    for t in creation_order:
        if t == "olist_zip_code_prefix":
            continue  # already created

        view = f"v__{t}"
        pk_cols = pk_map.get(t, [])
        fks = fk_map.get(t, [])

        # customers/sellers zip_prefix 타입이 숫자/문자 혼합일 수 있으니,
        # FK 호환을 위해 CREATE 전에 VIEW에서 CAST한 "정규화 view"를 하나 더 씌운다.
        if t == "olist_order_customer_dataset":
            # con.execute(f"DROP VIEW IF EXISTS {qident('v__norm__' + t)};")
            con.execute(
                f"""
                CREATE VIEW {qident('v__norm__' + t)} AS
                SELECT
                    *,
                    TRIM(CAST(customer_zip_code_prefix AS VARCHAR)) AS customer_zip_code_prefix
                FROM {qident(view)};
                """
            )
            view = f"v__norm__{t}"

        if t == "olist_sellers_dataset":
            # con.execute(f"DROP VIEW IF EXISTS {qident('v__norm__' + t)};")
            con.execute(
                f"""
                CREATE VIEW {qident('v__norm__' + t)} AS
                SELECT
                    *,
                    TRIM(CAST(seller_zip_code_prefix AS VARCHAR)) AS seller_zip_code_prefix
                FROM {qident(view)};
                """
            )
            view = f"v__norm__{t}"

        create_table_from_view(con, t, view, pk_cols, fks)

        # FK 컬럼 인덱스(조인/검사 빠르게)
        for (child_col, _, _) in fks:
            idx = f"idx__{t}__{child_col}"
            con.execute(f"CREATE INDEX {qident(idx)} ON {qtable(t)}({qident(child_col)});")

    # ====== 7) FK 검증(깨진 row count) ======
    print("\n=== FK VIOLATION CHECK ===")
    for t, fks in fk_map.items():
        for (child_col, parent_table, parent_col) in fks:
            q = f"""
            SELECT COUNT(*) AS n_bad
            FROM {qtable(t)} c
            LEFT JOIN {qtable(parent_table)} p
              ON c.{qident(child_col)} = p.{qident(parent_col)}
            WHERE c.{qident(child_col)} IS NOT NULL
              AND p.{qident(parent_col)} IS NULL;
            """
            n_bad = con.execute(q).fetchone()[0]
            if n_bad != 0:
                print(f"[WARN] FK violation: {t}.{child_col} -> {parent_table}.{parent_col} : {n_bad} rows")

    with open(SQL_PATH, "r") as f:
        flat_sql = f.read()

    flat = con.execute(flat_sql).fetchdf()
    flat['review_score'] = (flat['review_score'] >= flat['review_score'].median()).astype(int)
    flat.to_csv(CSV_PATH, index=False)


    con.close()
    print(f"\nDone.")

if __name__ == "__main__":
    main()
