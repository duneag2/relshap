import duckdb
import pandas as pd
import numpy as np
from folktables import ACSDataSource, ACSEmployment

data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=["AL"], download=True)
features, label, group = ACSEmployment.df_to_numpy(acs_data)

# =========================
# ACS -> DuckDB normalize + enforce checks + recreate tables with CHECK
# =========================

# !pip -q install folktables

import os
from typing import List, Dict, Optional, Tuple
import duckdb
import pandas as pd
from folktables import ACSDataSource, ACSPublicCoverage


# ---------- helpers ----------

def _norm_col(c: str) -> str:
    return c.strip().upper()

def _table_info(con: duckdb.DuckDBPyConnection, table: str):
    # PRAGMA table_info('table') -> (cid, name, type, notnull, dflt_value, pk)
    return con.execute(f"PRAGMA table_info('{table}')").fetchall()

def _table_cols(con: duckdb.DuckDBPyConnection, table: str) -> set:
    rows = _table_info(con, table)
    return {r[1].upper() for r in rows}

def _has_cols(con: duckdb.DuckDBPyConnection, table: str, cols: List[str]) -> bool:
    have = _table_cols(con, table)
    return all(c.upper() in have for c in cols)

def pick_cols(df: pd.DataFrame, desired: List[str], aliases: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """
    Pick columns that exist in df, with optional alias fallbacks.
    Returns actual df column names (preserving df's original column case).
    """
    aliases = aliases or {}
    colmap = {_norm_col(c): c for c in df.columns}  # normalized -> actual

    picked = []
    for want in desired:
        w = _norm_col(want)
        if w in colmap:
            picked.append(colmap[w])
            continue
        for alt in aliases.get(want, []) + aliases.get(w, []):
            a = _norm_col(alt)
            if a in colmap:
                picked.append(colmap[a])
                break

    # de-dup while preserving order
    seen = set()
    out = []
    for c in picked:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def create_table_from_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame):
    con.register("_tmp_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp_df;")
    con.unregister("_tmp_df")


# ---------- main builder ----------

def build_acs_duckdb(acs_f: pd.DataFrame, out_path: str = "acs.duckdb") -> str:
    # required keys
    need = {"SERIALNO", "SPORDER"}
    have = {_norm_col(c) for c in acs_f.columns}
    missing = sorted([c for c in need if c not in have])
    if missing:
        raise ValueError(f"acs_f is missing required columns: {missing}")

    if os.path.exists(out_path):
        os.remove(out_path)
    con = duckdb.connect(out_path)

    # 0) raw snapshot (optional but useful)
    create_table_from_df(con, "raw_person", acs_f)

    # ----- GEO -----
    geo_cols = pick_cols(acs_f, ["ST", "PUMA", "DIVISION", "REGION"])
    geo_df = acs_f[geo_cols].drop_duplicates() if geo_cols else pd.DataFrame()
    create_table_from_df(con, "geo", geo_df)

    # ----- HOUSEHOLD -----
    hh_cols = pick_cols(acs_f, ["SERIALNO", "ST", "PUMA", "RT"])
    serial_col = pick_cols(acs_f, ["SERIALNO"])[0]
    household_df = acs_f[hh_cols].drop_duplicates(subset=[serial_col])
    create_table_from_df(con, "household", household_df)

    # ----- PERSON_DEMOGRAPHIC -----
    demo_cols = pick_cols(
        acs_f,
        desired=[
            "SERIALNO", "SPORDER",
            "AGEP", "SEX", "RAC1P", "ANC",
            "MAR", "SCHL", "MIG", "MIL", "ESP", "RELP", "GCL",
        ],
        aliases={
            "AGEP": ["AGE", "AGES"],
            "SCHL": ["SCHOOL"],
        }
    )
    create_table_from_df(con, "person_demographic", acs_f[demo_cols].copy())

    # ----- PERSON_EMPLOYMENT -----
    empl_cols = pick_cols(
        acs_f,
        desired=[
            "SERIALNO", "SPORDER",
            "COW", "OCCP", "WKHP", "PINCP",
            "ESR",
            "JWMNP", "JWTR",
            "POVPIP", "POWPUMA"
        ],
        aliases={
            "PINCP": ["PINCH"],
            "WAGP": ["EAR", "EARN", "EARNING"],
        }
    )
    create_table_from_df(con, "person_employment", acs_f[empl_cols].copy())

    # ----- PERSON_BIRTH -----
    birth_cols = pick_cols(
        acs_f,
        desired=["SERIALNO", "SPORDER", "POBP", "CIT"],
        aliases={"POBP": ["POMP"]}
    )
    create_table_from_df(con, "person_birth", acs_f[birth_cols].copy())

    # ----- CIT_NATIVITY (small mapping table) -----
    cit_map = pd.DataFrame(
        {
            "CIT": ["1", "2", "3", "4", "5"],
            "NATIVITY": ["1", "1", "1", "2", "2"],  # folktables tutorial/codebook convention
        }
    )
    create_table_from_df(con, "cit_nativity", cit_map)

    # ----- PERSON_MEDICAL -----
    med_cols = pick_cols(
        acs_f,
        desired=[
            "SERIALNO", "SPORDER",
            "AGEP", "SEX",
            "PUBCOV", "FER",
            "DIS", "DEAR", "DEYE", "DREM",
        ],
        aliases={"DREM": ["DREAM"]}
    )
    create_table_from_df(con, "person_medical", acs_f[med_cols].copy())


    # ----- indexes -----
    con.execute("CREATE INDEX idx_household_serialno ON household(SERIALNO);")
    if _has_cols(con, "geo", ["ST", "PUMA"]):
        con.execute("CREATE INDEX idx_geo_key ON geo(ST, PUMA);")
    con.execute("CREATE INDEX idx_demo_key ON person_demographic(SERIALNO, SPORDER);")
    con.execute("CREATE INDEX idx_empl_key ON person_employment(SERIALNO, SPORDER);")
    con.execute("CREATE INDEX idx_birth_key ON person_birth(SERIALNO, SPORDER);")
    con.execute("CREATE INDEX idx_med_key ON person_medical(SERIALNO, SPORDER);")

    con.close()
    return out_path


# ---------- CHECK handling (DuckDB-safe: recreate table with CHECK in CREATE TABLE) ----------

def _recreate_table_with_checks(
    con: duckdb.DuckDBPyConnection,
    table: str,
    checks: List[Tuple[str, str]],
):
    """
    DuckDB may not support ALTER TABLE ADD CONSTRAINT CHECK in your environment.
    Workaround: create a new table with the same columns + CHECKs, copy data, swap.
    checks: list of (constraint_name, check_sql_expression)
            example: ("ck_age_lt15_mar5", "(AGEP >= 15 OR MAR = 5)")
    """
    # verify table exists
    exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table]
    ).fetchone()[0]
    if exists == 0:
        return

    info = _table_info(con, table)
    if not info:
        return

    tmp = f"{table}__tmp_with_checks"

    # build column DDL
    # (cid, name, type, notnull, dflt_value, pk)
    col_ddls = []
    for _, name, typ, notnull, dflt_value, _pk in info:
        ddl = f'"{name}" {typ}'
        if dflt_value is not None:
            ddl += f" DEFAULT {dflt_value}"
        if notnull:
            ddl += " NOT NULL"
        col_ddls.append(ddl)

    # build CHECK clauses
    check_ddls = []
    for cname, expr in checks:
        check_ddls.append(f'CONSTRAINT "{cname}" CHECK ({expr})')

    ddl = f'CREATE OR REPLACE TABLE "{tmp}" (\n  ' + ",\n  ".join(col_ddls + check_ddls) + "\n);"

    # swap in a transaction
    con.execute("BEGIN;")
    try:
        con.execute(f'DROP TABLE IF EXISTS "{tmp}";')
        con.execute(ddl)
        con.execute(f'INSERT INTO "{tmp}" SELECT * FROM "{table}";')
        con.execute(f'DROP TABLE "{table}";')
        con.execute(f'ALTER TABLE "{tmp}" RENAME TO "{table}";')
        con.execute("COMMIT;")
    except Exception as e:
        con.execute("ROLLBACK;")
        print(f"[WARN] failed to recreate table with checks: {table}\n  -> {e}\n")

def enforce_and_add_acs_checks(con: duckdb.DuckDBPyConnection):
    """
    (A) Enforce your listed check conditions by updating offending values
    (B) Recreate tables with CHECK constraints (DuckDB-safe workaround)

    NOTE:
    - The "gating" constraints (JWTR/JWMNP/POWPUMA) are intentionally written as
      simple OR-implications, split into atomic checks, to make downstream parsing easy.
      (No NOT((A OR B ...) AND rhs IS NOT NULL) forms.)
    """

    # ---------- (A) ENFORCE (make data satisfy checks) ----------

    # person_demographic
    if _has_cols(con, "person_demographic", ["AGEP", "MAR"]):
        # AGEP < 15 -> MAR = 5
        con.execute("""
            UPDATE person_demographic
            SET MAR = 5
            WHERE AGEP < 15 AND (MAR IS NULL OR MAR <> 5);
        """)
    if _has_cols(con, "person_demographic", ["AGEP", "SCHL"]):
        # AGEP < 3 -> SCHL IS NULL
        con.execute("""
            UPDATE person_demographic
            SET SCHL = NULL
            WHERE AGEP < 3 AND SCHL IS NOT NULL;
        """)
    if _has_cols(con, "person_demographic", ["RELP", "MAR"]):
        # RELP == 1 -> MAR == 1
        con.execute("""
            UPDATE person_demographic
            SET MAR = 1
            WHERE RELP = 1 AND (MAR IS NULL OR MAR <> 1);
        """)
    # person_medical enforce (atomic)
    # AGEP < 15  -> FER = 0
    # AGEP > 50  -> FER = 0
    # SEX = 1    -> FER = 0
    if _has_cols(con, "person_medical", ["AGEP", "SEX", "FER"]):
        con.execute("""
            UPDATE person_medical
            SET FER = 0
            WHERE AGEP < 15 AND (FER IS NULL OR FER <> 0);
        """)
        con.execute("""
            UPDATE person_medical
            SET FER = 0
            WHERE AGEP > 50 AND (FER IS NULL OR FER <> 0);
        """)
        con.execute("""
            UPDATE person_medical
            SET FER = 0
            WHERE SEX = 1 AND (FER IS NULL OR FER <> 0);
        """)



    # person_employment
    # AGEP<16 -> {COW,OCCP,WKHP,ESR,POWPUMA,JWMNP} IS NULL
    # AGEP<15 -> PINCP IS NULL
    # NOTE: Your build_acs_duckdb currently puts AGEP in person_demographic, not person_employment.
    # If AGEP is absent in person_employment, these updates/checks will be skipped safely.
    if _has_cols(con, "person_employment", ["AGEP"]):
        if _has_cols(con, "person_employment", ["AGEP", "COW"]):
            con.execute("UPDATE person_employment SET COW = NULL WHERE AGEP < 16 AND COW IS NOT NULL;")
        if _has_cols(con, "person_employment", ["AGEP", "OCCP"]):
            con.execute("UPDATE person_employment SET OCCP = NULL WHERE AGEP < 16 AND OCCP IS NOT NULL;")
        if _has_cols(con, "person_employment", ["AGEP", "WKHP"]):
            con.execute("UPDATE person_employment SET WKHP = NULL WHERE AGEP < 16 AND WKHP IS NOT NULL;")
        if _has_cols(con, "person_employment", ["AGEP", "ESR"]):
            con.execute("UPDATE person_employment SET ESR = NULL WHERE AGEP < 16 AND ESR IS NOT NULL;")
        if _has_cols(con, "person_employment", ["AGEP", "POWPUMA"]):
            con.execute("UPDATE person_employment SET POWPUMA = NULL WHERE AGEP < 16 AND POWPUMA IS NOT NULL;")
        if _has_cols(con, "person_employment", ["AGEP", "JWMNP"]):
            con.execute("UPDATE person_employment SET JWMNP = NULL WHERE AGEP < 16 AND JWMNP IS NOT NULL;")
        if _has_cols(con, "person_employment", ["AGEP", "PINCP"]):
            con.execute("UPDATE person_employment SET PINCP = NULL WHERE AGEP < 15 AND PINCP IS NOT NULL;")

    # COW IS NULL -> OCCP IS NULL
    if _has_cols(con, "person_employment", ["COW", "OCCP"]):
        con.execute("""
            UPDATE person_employment
            SET OCCP = NULL
            WHERE COW IS NULL AND OCCP IS NOT NULL;
        """)

    # (COW IS NULL OR OCCP IS NULL) -> WKHP IS NULL
    if _has_cols(con, "person_employment", ["COW", "OCCP", "WKHP"]):
        con.execute("""
            UPDATE person_employment
            SET WKHP = NULL
            WHERE (COW IS NULL OR OCCP IS NULL) AND WKHP IS NOT NULL;
        """)

    # JWTR gating: if any bad condition holds -> JWTR must be NULL
    if _has_cols(con, "person_employment", ["JWTR"]):
        # (COW IS NULL OR COW=9 OR ESR IS NULL OR ESR=3 OR OCCP IS NULL OR OCCP=9920) -> JWTR IS NULL
        if _has_cols(con, "person_employment", ["COW", "JWTR"]):
            con.execute("""
                UPDATE person_employment
                SET JWTR = NULL
                WHERE COW IS NULL AND JWTR IS NOT NULL;
            """)
            con.execute("""
                UPDATE person_employment
                SET JWTR = NULL
                WHERE COW = 9 AND JWTR IS NOT NULL;
            """)
        if _has_cols(con, "person_employment", ["ESR", "JWTR"]):
            con.execute("""
                UPDATE person_employment
                SET JWTR = NULL
                WHERE ESR IS NULL AND JWTR IS NOT NULL;
            """)
            con.execute("""
                UPDATE person_employment
                SET JWTR = NULL
                WHERE ESR = 3 AND JWTR IS NOT NULL;
            """)
        if _has_cols(con, "person_employment", ["OCCP", "JWTR"]):
            con.execute("""
                UPDATE person_employment
                SET JWTR = NULL
                WHERE OCCP IS NULL AND JWTR IS NOT NULL;
            """)
            con.execute("""
                UPDATE person_employment
                SET JWTR = NULL
                WHERE OCCP = 9920 AND JWTR IS NOT NULL;
            """)

    # JWMNP gating: if any bad condition holds -> JWMNP must be NULL
    if _has_cols(con, "person_employment", ["JWMNP"]):
        # (AGEP < 16 OR ESR IS NULL OR ESR=3 OR COW IS NULL OR OCCP IS NULL OR WKHP IS NULL) -> JWMNP IS NULL
        if _has_cols(con, "person_employment", ["AGEP", "JWMNP"]):
            con.execute("""
                UPDATE person_employment
                SET JWMNP = NULL
                WHERE AGEP < 16 AND JWMNP IS NOT NULL;
            """)
        if _has_cols(con, "person_employment", ["ESR", "JWMNP"]):
            con.execute("""
                UPDATE person_employment
                SET JWMNP = NULL
                WHERE ESR IS NULL AND JWMNP IS NOT NULL;
            """)
            con.execute("""
                UPDATE person_employment
                SET JWMNP = NULL
                WHERE ESR = 3 AND JWMNP IS NOT NULL;
            """)
        if _has_cols(con, "person_employment", ["COW", "JWMNP"]):
            con.execute("""
                UPDATE person_employment
                SET JWMNP = NULL
                WHERE COW IS NULL AND JWMNP IS NOT NULL;
            """)
        if _has_cols(con, "person_employment", ["OCCP", "JWMNP"]):
            con.execute("""
                UPDATE person_employment
                SET JWMNP = NULL
                WHERE OCCP IS NULL AND JWMNP IS NOT NULL;
            """)
        if _has_cols(con, "person_employment", ["WKHP", "JWMNP"]):
            con.execute("""
                UPDATE person_employment
                SET JWMNP = NULL
                WHERE WKHP IS NULL AND JWMNP IS NOT NULL;
            """)

    # POWPUMA gating: if any bad condition holds -> POWPUMA must be NULL
    if _has_cols(con, "person_employment", ["POWPUMA"]):
        # (AGEP < 16 OR ESR IS NULL OR ESR=3 OR COW IS NULL OR OCCP IS NULL OR OCCP=9920 OR WKHP IS NULL) -> POWPUMA IS NULL
        if _has_cols(con, "person_employment", ["AGEP", "POWPUMA"]):
            con.execute("""
                UPDATE person_employment
                SET POWPUMA = NULL
                WHERE AGEP < 16 AND POWPUMA IS NOT NULL;
            """)
        if _has_cols(con, "person_employment", ["ESR", "POWPUMA"]):
            con.execute("""
                UPDATE person_employment
                SET POWPUMA = NULL
                WHERE ESR IS NULL AND POWPUMA IS NOT NULL;
            """)
            con.execute("""
                UPDATE person_employment
                SET POWPUMA = NULL
                WHERE ESR = 3 AND POWPUMA IS NOT NULL;
            """)
        if _has_cols(con, "person_employment", ["COW", "POWPUMA"]):
            con.execute("""
                UPDATE person_employment
                SET POWPUMA = NULL
                WHERE COW IS NULL AND POWPUMA IS NOT NULL;
            """)
        if _has_cols(con, "person_employment", ["OCCP", "POWPUMA"]):
            con.execute("""
                UPDATE person_employment
                SET POWPUMA = NULL
                WHERE OCCP IS NULL AND POWPUMA IS NOT NULL;
            """)
            con.execute("""
                UPDATE person_employment
                SET POWPUMA = NULL
                WHERE OCCP = 9920 AND POWPUMA IS NOT NULL;
            """)
        if _has_cols(con, "person_employment", ["WKHP", "POWPUMA"]):
            con.execute("""
                UPDATE person_employment
                SET POWPUMA = NULL
                WHERE WKHP IS NULL AND POWPUMA IS NOT NULL;
            """)

    # ---------- (B) "ADD" CHECKS by recreating tables with CHECK in CREATE TABLE ----------

    # ---- person_demographic checks ----
    demo_checks = []
    if _has_cols(con, "person_demographic", ["AGEP", "MAR"]):
        demo_checks.append(("ck_age_lt15_mar5", "(AGEP >= 15 OR MAR = 5)"))
    if _has_cols(con, "person_demographic", ["AGEP", "SCHL"]):
        demo_checks.append(("ck_age_lt3_schl_null", "(AGEP >= 3 OR SCHL IS NULL)"))
    if _has_cols(con, "person_demographic", ["RELP", "MAR"]):
        demo_checks.append(("ck_relp1_mar1", "(RELP <> 1 OR MAR = 1)"))

    if demo_checks:
        _recreate_table_with_checks(con, "person_demographic", demo_checks)

    # ---- person_employment checks ----
    empl_checks = []

    # AGEP-based nulling (only if AGEP exists in this table)
    if _has_cols(con, "person_employment", ["AGEP", "COW"]):
        empl_checks.append(("ck_age_lt16_cow_null", "(AGEP >= 16 OR COW IS NULL)"))
    if _has_cols(con, "person_employment", ["AGEP", "OCCP"]):
        empl_checks.append(("ck_age_lt16_occp_null", "(AGEP >= 16 OR OCCP IS NULL)"))
    if _has_cols(con, "person_employment", ["AGEP", "WKHP"]):
        empl_checks.append(("ck_age_lt16_wkhp_null", "(AGEP >= 16 OR WKHP IS NULL)"))
    if _has_cols(con, "person_employment", ["AGEP", "ESR"]):
        empl_checks.append(("ck_age_lt16_esr_null", "(AGEP >= 16 OR ESR IS NULL)"))
    if _has_cols(con, "person_employment", ["AGEP", "POWPUMA"]):
        empl_checks.append(("ck_age_lt16_powpuma_null", "(AGEP >= 16 OR POWPUMA IS NULL)"))
    if _has_cols(con, "person_employment", ["AGEP", "JWMNP"]):
        empl_checks.append(("ck_age_lt16_jwmnp_null", "(AGEP >= 16 OR JWMNP IS NULL)"))
    if _has_cols(con, "person_employment", ["AGEP", "PINCP"]):
        empl_checks.append(("ck_age_lt15_pincp_null", "(AGEP >= 15 OR PINCP IS NULL)"))

    # ---- person_medical checks (atomic) ----
    med_checks = []
    if _has_cols(con, "person_medical", ["AGEP", "SEX", "FER"]):
        # AGEP < 15 -> FER = 0
        med_checks.append(("ck_fer_age_lt15", "(AGEP >= 15 OR FER = 0)"))
        # AGEP > 50 -> FER = 0
        med_checks.append(("ck_fer_age_gt50", "(AGEP <= 50 OR FER = 0)"))
        # SEX = 1 -> FER = 0
        med_checks.append(("ck_fer_male", "(SEX <> 1 OR FER = 0)"))

    if med_checks:
        _recreate_table_with_checks(con, "person_medical", med_checks)



    # COW NULL -> OCCP NULL
    if _has_cols(con, "person_employment", ["COW", "OCCP"]):
        empl_checks.append(("ck_cow_null_occp_null", "(COW IS NOT NULL OR OCCP IS NULL)"))

    # (COW NULL OR OCCP NULL) -> WKHP NULL
    if _has_cols(con, "person_employment", ["COW", "OCCP", "WKHP"]):
        empl_checks.append((
            "ck_cow_or_occp_null_wkhp_null",
            "((COW IS NOT NULL AND OCCP IS NOT NULL) OR WKHP IS NULL)"
        ))

    # --- ATOMIC gating checks (parser-friendly) ---

    # JWTR gating
    if _has_cols(con, "person_employment", ["JWTR"]):
        if _has_cols(con, "person_employment", ["COW", "JWTR"]):
            empl_checks.append(("ck_jwtr_cow_null", "(COW IS NOT NULL OR JWTR IS NULL)"))
            empl_checks.append(("ck_jwtr_cow_9",    "(COW <> 9 OR JWTR IS NULL)"))
        if _has_cols(con, "person_employment", ["ESR", "JWTR"]):
            empl_checks.append(("ck_jwtr_esr_null", "(ESR IS NOT NULL OR JWTR IS NULL)"))
            empl_checks.append(("ck_jwtr_esr_3",    "(ESR <> 3 OR JWTR IS NULL)"))
        if _has_cols(con, "person_employment", ["OCCP", "JWTR"]):
            empl_checks.append(("ck_jwtr_occp_null","(OCCP IS NOT NULL OR JWTR IS NULL)"))
            empl_checks.append(("ck_jwtr_occp_9920","(OCCP <> 9920 OR JWTR IS NULL)"))

    # JWMNP gating
    if _has_cols(con, "person_employment", ["JWMNP"]):
        if _has_cols(con, "person_employment", ["AGEP", "JWMNP"]):
            empl_checks.append(("ck_jwmnp_age_lt16", "(AGEP >= 16 OR JWMNP IS NULL)"))
        if _has_cols(con, "person_employment", ["ESR", "JWMNP"]):
            empl_checks.append(("ck_jwmnp_esr_null", "(ESR IS NOT NULL OR JWMNP IS NULL)"))
            empl_checks.append(("ck_jwmnp_esr_3",    "(ESR <> 3 OR JWMNP IS NULL)"))
        if _has_cols(con, "person_employment", ["COW", "JWMNP"]):
            empl_checks.append(("ck_jwmnp_cow_null", "(COW IS NOT NULL OR JWMNP IS NULL)"))
        if _has_cols(con, "person_employment", ["OCCP", "JWMNP"]):
            empl_checks.append(("ck_jwmnp_occp_null","(OCCP IS NOT NULL OR JWMNP IS NULL)"))
        if _has_cols(con, "person_employment", ["WKHP", "JWMNP"]):
            empl_checks.append(("ck_jwmnp_wkhp_null","(WKHP IS NOT NULL OR JWMNP IS NULL)"))

    # POWPUMA gating
    if _has_cols(con, "person_employment", ["POWPUMA"]):
        if _has_cols(con, "person_employment", ["AGEP", "POWPUMA"]):
            empl_checks.append(("ck_powpuma_age_lt16", "(AGEP >= 16 OR POWPUMA IS NULL)"))
        if _has_cols(con, "person_employment", ["ESR", "POWPUMA"]):
            empl_checks.append(("ck_powpuma_esr_null", "(ESR IS NOT NULL OR POWPUMA IS NULL)"))
            empl_checks.append(("ck_powpuma_esr_3",    "(ESR <> 3 OR POWPUMA IS NULL)"))
        if _has_cols(con, "person_employment", ["COW", "POWPUMA"]):
            empl_checks.append(("ck_powpuma_cow_null", "(COW IS NOT NULL OR POWPUMA IS NULL)"))
        if _has_cols(con, "person_employment", ["OCCP", "POWPUMA"]):
            empl_checks.append(("ck_powpuma_occp_null","(OCCP IS NOT NULL OR POWPUMA IS NULL)"))
            empl_checks.append(("ck_powpuma_occp_9920","(OCCP <> 9920 OR POWPUMA IS NULL)"))
        if _has_cols(con, "person_employment", ["WKHP", "POWPUMA"]):
            empl_checks.append(("ck_powpuma_wkhp_null","(WKHP IS NOT NULL OR POWPUMA IS NULL)"))

    if empl_checks:
        _recreate_table_with_checks(con, "person_employment", empl_checks)

# def enforce_and_add_acs_checks(con: duckdb.DuckDBPyConnection):
#     """
#     (A) Enforce your listed check conditions by updating offending values
#     (B) Recreate tables with CHECK constraints (DuckDB-safe workaround)
#     """

#     # ---------- (A) ENFORCE (make data satisfy checks) ----------

#     # person_demographic
#     if _has_cols(con, "person_demographic", ["AGEP", "MAR"]):
#         # AGEP < 15 -> MAR = 5
#         con.execute("""
#             UPDATE person_demographic
#             SET MAR = 5
#             WHERE AGEP < 15 AND (MAR IS NULL OR MAR <> 5);
#         """)
#     if _has_cols(con, "person_demographic", ["AGEP", "SCHL"]):
#         # AGEP < 3 -> SCHL IS NULL
#         con.execute("""
#             UPDATE person_demographic
#             SET SCHL = NULL
#             WHERE AGEP < 3 AND SCHL IS NOT NULL;
#         """)
#     if _has_cols(con, "person_demographic", ["RELP", "MAR"]):
#         # RELP == 1 -> MAR == 1  (your rule)
#         con.execute("""
#             UPDATE person_demographic
#             SET MAR = 1
#             WHERE RELP = 1 AND (MAR IS NULL OR MAR <> 1);
#         """)

#     # person_employment
#     # AGEP<16 -> {COW,OCCP,WKHP,ESR,POWPUMA,JWMNP} IS NULL
#     # AGEP<15 -> PINCP IS NULL
#     if _has_cols(con, "person_employment", ["AGEP"]):
#         if _has_cols(con, "person_employment", ["AGEP", "COW"]):
#             con.execute("UPDATE person_employment SET COW = NULL WHERE AGEP < 16 AND COW IS NOT NULL;")
#         if _has_cols(con, "person_employment", ["AGEP", "OCCP"]):
#             con.execute("UPDATE person_employment SET OCCP = NULL WHERE AGEP < 16 AND OCCP IS NOT NULL;")
#         if _has_cols(con, "person_employment", ["AGEP", "WKHP"]):
#             con.execute("UPDATE person_employment SET WKHP = NULL WHERE AGEP < 16 AND WKHP IS NOT NULL;")
#         if _has_cols(con, "person_employment", ["AGEP", "ESR"]):
#             con.execute("UPDATE person_employment SET ESR = NULL WHERE AGEP < 16 AND ESR IS NOT NULL;")
#         if _has_cols(con, "person_employment", ["AGEP", "POWPUMA"]):
#             con.execute("UPDATE person_employment SET POWPUMA = NULL WHERE AGEP < 16 AND POWPUMA IS NOT NULL;")
#         if _has_cols(con, "person_employment", ["AGEP", "JWMNP"]):
#             con.execute("UPDATE person_employment SET JWMNP = NULL WHERE AGEP < 16 AND JWMNP IS NOT NULL;")
#         if _has_cols(con, "person_employment", ["AGEP", "PINCP"]):
#             con.execute("UPDATE person_employment SET PINCP = NULL WHERE AGEP < 15 AND PINCP IS NOT NULL;")

#     # COW IS NULL -> OCCP IS NULL
#     if _has_cols(con, "person_employment", ["COW", "OCCP"]):
#         con.execute("""
#             UPDATE person_employment
#             SET OCCP = NULL
#             WHERE COW IS NULL AND OCCP IS NOT NULL;
#         """)

#     # (COW IS NULL OR OCCP IS NULL) -> WKHP IS NULL
#     if _has_cols(con, "person_employment", ["COW", "OCCP", "WKHP"]):
#         con.execute("""
#             UPDATE person_employment
#             SET WKHP = NULL
#             WHERE (COW IS NULL OR OCCP IS NULL) AND WKHP IS NOT NULL;
#         """)

#     # JWTR gating:
#     if _has_cols(con, "person_employment", ["COW", "ESR", "OCCP", "JWTR"]):
#         con.execute("""
#             UPDATE person_employment
#             SET JWTR = NULL
#             WHERE (COW IS NULL OR COW = 9 OR ESR IS NULL OR ESR = 3 OR OCCP IS NULL OR OCCP = 9920)
#               AND JWTR IS NOT NULL;
#         """)

#     # JWMNP gating:
#     if _has_cols(con, "person_employment", ["AGEP", "ESR", "COW", "OCCP", "WKHP", "JWMNP"]):
#         con.execute("""
#             UPDATE person_employment
#             SET JWMNP = NULL
#             WHERE (AGEP < 16 OR ESR IS NULL OR ESR = 3 OR COW IS NULL OR OCCP IS NULL OR WKHP IS NULL)
#               AND JWMNP IS NOT NULL;
#         """)

#     # POWPUMA gating:
#     if _has_cols(con, "person_employment", ["AGEP", "ESR", "COW", "OCCP", "WKHP", "POWPUMA"]):
#         con.execute("""
#             UPDATE person_employment
#             SET POWPUMA = NULL
#             WHERE (AGEP < 16 OR ESR IS NULL OR ESR = 3 OR COW IS NULL OR OCCP IS NULL OR OCCP = 9920 OR WKHP IS NULL)
#               AND POWPUMA IS NOT NULL;
#         """)

#     # ---------- (B) "ADD" CHECKS by recreating tables with CHECK in CREATE TABLE ----------

#     demo_checks = []
#     if _has_cols(con, "person_demographic", ["AGEP", "MAR"]):
#         demo_checks.append(("ck_age_lt15_mar5", "(AGEP >= 15 OR MAR = 5)"))
#     if _has_cols(con, "person_demographic", ["AGEP", "SCHL"]):
#         demo_checks.append(("ck_age_lt3_schl_null", "(AGEP >= 3 OR SCHL IS NULL)"))
#     if _has_cols(con, "person_demographic", ["RELP", "MAR"]):
#         demo_checks.append(("ck_relp1_mar1", "(RELP <> 1 OR MAR = 1)"))

#     if demo_checks:
#         _recreate_table_with_checks(con, "person_demographic", demo_checks)

#     empl_checks = []
#     if _has_cols(con, "person_employment", ["AGEP", "COW"]):
#         empl_checks.append(("ck_age_lt16_cow_null", "(AGEP >= 16 OR COW IS NULL)"))
#     if _has_cols(con, "person_employment", ["AGEP", "OCCP"]):
#         empl_checks.append(("ck_age_lt16_occp_null", "(AGEP >= 16 OR OCCP IS NULL)"))
#     if _has_cols(con, "person_employment", ["AGEP", "WKHP"]):
#         empl_checks.append(("ck_age_lt16_wkhp_null", "(AGEP >= 16 OR WKHP IS NULL)"))
#     if _has_cols(con, "person_employment", ["AGEP", "ESR"]):
#         empl_checks.append(("ck_age_lt16_esr_null", "(AGEP >= 16 OR ESR IS NULL)"))
#     if _has_cols(con, "person_employment", ["AGEP", "POWPUMA"]):
#         empl_checks.append(("ck_age_lt16_powpuma_null", "(AGEP >= 16 OR POWPUMA IS NULL)"))
#     if _has_cols(con, "person_employment", ["AGEP", "JWMNP"]):
#         empl_checks.append(("ck_age_lt16_jwmnp_null", "(AGEP >= 16 OR JWMNP IS NULL)"))
#     if _has_cols(con, "person_employment", ["AGEP", "PINCP"]):
#         empl_checks.append(("ck_age_lt15_pincp_null", "(AGEP >= 15 OR PINCP IS NULL)"))
#     if _has_cols(con, "person_employment", ["COW", "OCCP"]):
#         empl_checks.append(("ck_cow_null_occp_null", "(COW IS NOT NULL OR OCCP IS NULL)"))
#     if _has_cols(con, "person_employment", ["COW", "OCCP", "WKHP"]):
#         empl_checks.append(("ck_cow_or_occp_null_wkhp_null", "((COW IS NOT NULL AND OCCP IS NOT NULL) OR WKHP IS NULL)"))
#     if _has_cols(con, "person_employment", ["COW", "ESR", "OCCP", "JWTR"]):
#         empl_checks.append((
#             "ck_jwtr_gating",
#             "NOT ((COW IS NULL OR COW = 9 OR ESR IS NULL OR ESR = 3 OR OCCP IS NULL OR OCCP = 9920) AND JWTR IS NOT NULL)"
#         ))
#     if _has_cols(con, "person_employment", ["AGEP", "ESR", "COW", "OCCP", "WKHP", "JWMNP"]):
#         empl_checks.append((
#             "ck_jwmnp_gating",
#             "NOT ((AGEP < 16 OR ESR IS NULL OR ESR = 3 OR COW IS NULL OR OCCP IS NULL OR WKHP IS NULL) AND JWMNP IS NOT NULL)"
#         ))
#     if _has_cols(con, "person_employment", ["AGEP", "ESR", "COW", "OCCP", "WKHP", "POWPUMA"]):
#         empl_checks.append((
#             "ck_powpuma_gating",
#             "NOT ((AGEP < 16 OR ESR IS NULL OR ESR = 3 OR COW IS NULL OR OCCP IS NULL OR OCCP = 9920 OR WKHP IS NULL) AND POWPUMA IS NOT NULL)"
#         ))

#     if empl_checks:
#         _recreate_table_with_checks(con, "person_employment", empl_checks)


# ---------- run end-to-end ----------

data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=["AL"], download=True)

out_path = build_acs_duckdb(acs_data, out_path="./acs/acsemployment_al.duckdb")

con = duckdb.connect(out_path)
enforce_and_add_acs_checks(con)
con.close()

print("Wrote:", out_path)


import duckdb
from typing import List, Tuple

DB_PATH = "./acs/acsemployment_al.duckdb"

def q(con, sql, params=None):
    return con.execute(sql, params or []).fetchall()

def table_exists(con, table: str) -> bool:
    return q(con, """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [table])[0][0] > 0

def pragma_table_info(con, table: str):
    return q(con, f"PRAGMA table_info('{table}')")

def get_existing_checks(con, table: str) -> List[Tuple[str, str]]:
    rows = q(con, """
        SELECT DISTINCT
            cc.constraint_name,
            cc.check_clause
        FROM information_schema.table_constraints tc
        JOIN information_schema.check_constraints cc
          ON tc.constraint_name = cc.constraint_name
         AND tc.constraint_schema = cc.constraint_schema
        WHERE tc.table_name = ?
          AND tc.constraint_type = 'CHECK'
        ORDER BY 1, 2
    """, [table])
    out = []
    for name, clause in rows:
        s = clause.strip()
        if s.upper().startswith("CHECK"):
            l = s.find("("); r = s.rfind(")")
            inner = s[l+1:r] if (l != -1 and r != -1 and r > l) else s
        else:
            inner = s
        out.append((name, inner))
    return out

def recreate_table_with_pk_fk_keep_checks(
    con: duckdb.DuckDBPyConnection,
    table: str,
    pk_cols: List[str],
    fk_specs: List[Tuple[str, List[str], str, List[str]]],
    keep_checks: bool = True,
):
    """
    fk_specs: (fk_name, child_cols, parent_table, parent_cols)
    """
    if not table_exists(con, table):
        return

    info = pragma_table_info(con, table)
    tmp = f"{table}__tmp_fix_fk"

    # columns
    col_ddls = []
    for _cid, name, typ, notnull, dflt_value, _pk in info:
        ddl = f'"{name}" {typ}'
        if dflt_value is not None:
            ddl += f" DEFAULT {dflt_value}"
        if notnull:
            ddl += " NOT NULL"
        col_ddls.append(ddl)

    # checks
    check_ddls = []
    if keep_checks:
        for cname, expr in get_existing_checks(con, table):
            check_ddls.append(f'CONSTRAINT "{cname}" CHECK ({expr})')

    # pk
    pk_ddl = f'CONSTRAINT "{table}_pk" PRIMARY KEY (' + ", ".join([f'"{c}"' for c in pk_cols]) + ")"

    # fks
    fk_ddls = []
    for fk_name, child_cols, parent_table, parent_cols in fk_specs:
        fk_ddls.append(
            f'CONSTRAINT "{fk_name}" FOREIGN KEY (' +
            ", ".join([f'"{c}"' for c in child_cols]) +
            f') REFERENCES "{parent_table}"(' +
            ", ".join([f'"{c}"' for c in parent_cols]) +
            ")"
        )

    ddl = f'CREATE OR REPLACE TABLE "{tmp}" (\n  ' + ",\n  ".join(
        col_ddls + check_ddls + [pk_ddl] + fk_ddls
    ) + "\n);"

    con.execute("BEGIN;")
    try:
        con.execute(f'DROP TABLE IF EXISTS "{tmp}";')
        con.execute(ddl)
        con.execute(f'INSERT INTO "{tmp}" SELECT * FROM "{table}";')
        con.execute(f'DROP TABLE "{table}";')
        con.execute(f'ALTER TABLE "{tmp}" RENAME TO "{table}";')
        con.execute("COMMIT;")
    except Exception as e:
        con.execute("ROLLBACK;")
        raise RuntimeError(f"Failed recreating {table}: {e}")

def fix_all_fks(db_path: str = DB_PATH):
    con = duckdb.connect(db_path)

    assert table_exists(con, "household"), "household table missing"

    # household PK stays
    recreate_table_with_pk_fk_keep_checks(
        con, "household",
        pk_cols=["SERIALNO"],
        fk_specs=[],
        keep_checks=True
    )

    # geo pk
    if table_exists(con, "geo"):
        recreate_table_with_pk_fk_keep_checks(con, "geo", ["ST","PUMA"], [], True)

    # cit_nativity pk
    if table_exists(con, "cit_nativity"):
        recreate_table_with_pk_fk_keep_checks(con, "cit_nativity", ["CIT"], [], True)

    # person_* tables: FK(serialno) -> household(serialno)
    person_tables = [
        "person_birth",
        "person_demographic",
        "person_employment",
        "person_medical",
    ]
    for t in person_tables:
        if not table_exists(con, t): 
            continue
        recreate_table_with_pk_fk_keep_checks(
            con, t,
            pk_cols=["SERIALNO","SPORDER"],
            fk_specs=[(f"{t}_fk_household", ["SERIALNO"], "household", ["SERIALNO"])],
            keep_checks=True
        )


    con.close()

def dump_fks_correctly(db_path: str = DB_PATH):
    con = duckdb.connect(db_path)
    rows = q(con, """
        SELECT DISTINCT
            tc.table_name AS child_table,
            tc.constraint_name,
            kcu.column_name AS child_col,
            kcu.ordinal_position
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.constraint_schema = kcu.constraint_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
        ORDER BY 1, 2, 4
    """)
    # We need parent mapping too; DuckDB's constraint_column_usage can be ambiguous for FK.
    # Use duckdb_constraints() if available:
    try:
        fk2 = q(con, "SELECT * FROM duckdb_constraints() WHERE constraint_type='FOREIGN KEY'")
        print("\n=== duckdb_constraints() FOREIGN KEYS ===")
        for r in fk2:
            print(r)
    except Exception:
        pass

    print("\n=== information_schema FOREIGN KEYS (child side) ===")
    if not rows:
        print("(none)")
    else:
        cur = None
        acc = []
        for child, cname, col, _pos in rows:
            key = (child, cname)
            if cur is None:
                cur = key
            if key != cur:
                print(f"{cur[1]} on {cur[0]}: {acc}")
                cur = key
                acc = [col]
            else:
                acc.append(col)
        print(f"{cur[1]} on {cur[0]}: {acc}")

    con.close()

# run
fix_all_fks(DB_PATH)
dump_fks_correctly(DB_PATH)
print("FKs fixed.")


DB_PATH = "./acs/acsemployment_al.duckdb"

# 1) DuckDB 연결
con = duckdb.connect(DB_PATH)


SQL_PATH = "./acs/query_acsemployment_al_demo.sql"
with open(SQL_PATH, "r") as f:
    QUERY = f.read()


# 3) 실행 → pandas DataFrame
df = con.execute(QUERY).fetch_df()

con.close()
df = df.fillna(0) # acsemployment는 어쩐일인지 nan을 다 0으로 대치함

df_original = df
df = df.drop(["SERIALNO", "SPORDER", "ESR"],axis=1)
df = df.astype(np.float64) # join하면서 str 형식이 남아있는게 있음. typecast

features = pd.DataFrame(features)
label = pd.DataFrame(label)

df = df[df.apply(tuple, axis=1).isin(
    features.apply(tuple, axis=1)
)]
# print(df.head(5))
# print(features.head(5))
# print(df.shape)
# print(features.shape)
# print(df.shape == features.shape)

assert df.shape == features.shape
# if (df.shape == features.shape):
#     flattened_al_demo = pd.concat([features, label],axis=1)

df_original["ESR"] = (df_original["ESR"] == 1).astype(int)
flattened_al_demo = df_original

task = ACSEmployment
feature_list = task.features
feature_list.append(task.target)

feature_list = np.array(feature_list)
feature_list = np.concatenate([["SERIALNO", "SPORDER"], feature_list])


flattened_al_demo.to_csv(
    "./acs/flattened_acsemployment_al_demo.csv",
    header=feature_list,
    index=False
)