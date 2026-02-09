import os
from typing import Dict, Tuple, List
import re
import itertools
import argparse
import duckdb
import pandas as pd


def _extract_relational(db_path: str, schema: str):
    """
    Returns:
      table_cols: {table -> [col1, col2, ...]}
      pk:   [(table, (pk_cols...)), ...]
      uq:   [(table, (uq_cols...)), ...]
      fk:   [(child_table, (child_cols...), ref_table, (ref_cols...)), ...]
      check_constraints: [(table, constraint_name, clause_text), ...]  clause_text is like "CHECK(...)"
    """
    con = duckdb.connect(db_path, read_only=True)

    def _fetchall(q, params=()):
        return con.execute(q, params).fetchall()

    def _try_fetchall(q, params=()):
        try:
            return _fetchall(q, params)
        except Exception:
            return None

    # ----------------------------
    # 1) columns per table
    # ----------------------------
    table_cols: Dict[str, List[str]] = {}
    cols = _try_fetchall(
        """
        SELECT table_name, column_name, ordinal_position
        FROM information_schema.columns
        WHERE table_schema=?
        ORDER BY table_name, ordinal_position
        """,
        (schema,),
    )
    if cols:
        for t, c, _ in cols:
            t, c = str(t), str(c)
            table_cols.setdefault(t, []).append(c)

    # ----------------------------
    # 2) constraints metadata (PK/UQ/FK)
    # ----------------------------
    tc = _try_fetchall(
        """
        SELECT constraint_name, table_name, constraint_type
        FROM information_schema.table_constraints
        WHERE table_schema=?
        """,
        (schema,),
    )

    kcu = _try_fetchall(
        """
        SELECT constraint_name, table_name, column_name, ordinal_position
        FROM information_schema.key_column_usage
        WHERE table_schema=?
        """,
        (schema,),
    )

    rc = _try_fetchall(
        """
        SELECT constraint_name, unique_constraint_name
        FROM information_schema.referential_constraints
        WHERE constraint_schema=?
        """,
        (schema,),
    )

    pk: List[Tuple[str, Tuple[str, ...]]] = []
    uq: List[Tuple[str, Tuple[str, ...]]] = []
    fk: List[Tuple[str, Tuple[str, ...], str, Tuple[str, ...]]] = []
    check_constraints: List[Tuple[str, str, str]] = []

    # =========================================================
    # 3) CHECK extraction (robust)
    #    Prefer duckdb_constraints(), fallback to PRAGMA show_constraints()
    # =========================================================
    check_rows = _try_fetchall(
        """
        SELECT
          schema_name,
          table_name,
          constraint_name,
          constraint_type,
          constraint_text
        FROM duckdb_constraints()
        WHERE schema_name = ?
          AND constraint_type = 'CHECK'
        """,
        (schema,),
    )

    if check_rows:
        for sname, tname, cname, ctype, ctext in check_rows:
            if not ctext:
                continue
            clause = str(ctext).strip()
            # normalize to include CHECK(...)
            if "CHECK" not in clause.upper():
                clause = f"CHECK({clause})"
            check_constraints.append((str(tname), str(cname), clause))
    else:
        # fallback: PRAGMA show_constraints per table
        for tname in table_cols.keys():
            dfc = None
            try:
                dfc = con.execute(f"PRAGMA show_constraints('{tname}')").df()
            except Exception:
                try:
                    dfc = con.execute(f"PRAGMA show_constraints('{schema}.{tname}')").df()
                except Exception:
                    dfc = None

            if dfc is None or dfc.empty:
                continue

            # try to locate columns robustly across versions
            col_lut = {c.lower(): c for c in dfc.columns}
            name_col = col_lut.get("constraint_name") or col_lut.get("name") or dfc.columns[0]
            type_col = col_lut.get("constraint_type") or col_lut.get("type") or dfc.columns[1]
            text_col = (
                col_lut.get("constraint_text")
                or col_lut.get("expression")
                or col_lut.get("sql")
                or col_lut.get("definition")
            )

            for _, r in dfc.iterrows():
                ctype = str(r[type_col]).upper()
                if ctype != "CHECK":
                    continue
                cname = str(r[name_col])
                expr = ""
                if text_col is not None and text_col in dfc.columns:
                    v = r.get(text_col)
                    if v is not None and pd.notna(v):
                        expr = str(v).strip()
                if not expr:
                    continue
                clause = expr
                if "CHECK" not in clause.upper():
                    clause = f"CHECK({clause})"
                check_constraints.append((str(tname), cname, clause))

    # ----------------------------
    # 4) PK / UQ / FK extraction
    # ----------------------------
    if tc and kcu:
        # (constraint_name, table_name) -> [(ordinal, col), ...]
        colmap: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
        for cname, tname, col, ordpos in kcu:
            colmap.setdefault((str(cname), str(tname)), []).append(
                (int(ordpos) if ordpos is not None else 0, str(col))
            )
        for k in colmap:
            colmap[k].sort()

        # FK constraint -> referenced UNIQUE constraint name
        ref_map: Dict[str, str] = {}
        if rc:
            for fk_c, uq_c in rc:
                ref_map[str(fk_c)] = str(uq_c)

        # constraint_name -> (table, (cols...)) for targets
        ref_targets: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
        for (cname, tname), cols_sorted in colmap.items():
            ref_targets[cname] = (tname, tuple(c for _, c in cols_sorted))

        for cname, tname, ctype in tc:
            cname, tname, ctype = str(cname), str(tname), str(ctype).upper()
            cols_tup = tuple(c for _, c in colmap.get((cname, tname), []))
            if not cols_tup:
                continue

            if ctype == "PRIMARY KEY":
                pk.append((tname, cols_tup))
            elif ctype == "UNIQUE":
                uq.append((tname, cols_tup))
            elif ctype == "FOREIGN KEY":
                # map to referenced unique constraint columns
                if cname in ref_map and ref_map[cname] in ref_targets:
                    rt, rcols = ref_targets[ref_map[cname]]
                    fk.append((tname, cols_tup, rt, rcols))

    con.close()
    return table_cols, pk, uq, fk, check_constraints



def build_relational_constraint_dfs(db_path: str, schema: str = "main") -> Dict[str, pd.DataFrame]:
    table_cols, pks, uqs, fks, checks = _extract_relational(db_path, schema)

    # --- PK -> rest ---
    df_fd_pk = pd.DataFrame(
        [
            {
                "table": t,
                "lhs": ",".join(cols),
                "rhs": ",".join(c for c in table_cols.get(t, []) if c not in cols),
            }
            for t, cols in pks
        ],
        columns=["table", "lhs", "rhs"],
    )

    # --- UNIQUE -> rest (like PK) ---
    pk_set = {(t, tuple(cols)) for t, cols in pks}
    df_fd_uq = pd.DataFrame(
        [
            {
                "table": t,
                "lhs": ",".join(cols),
                "rhs": ",".join(c for c in table_cols.get(t, []) if c not in cols),
            }
            for t, cols in uqs
            if (t, tuple(cols)) not in pk_set  # avoid duplicates if UNIQUE == PK
        ],
        columns=["table", "lhs", "rhs"],
    )

    # --- FK-derived: child FK -> (parent PK's non-key attrs) (only when FK references PK) ---
    pk_lhs = {t: cols for t, cols in pks}
    pk_rhs = {t: [c for c in table_cols.get(t, []) if c not in cols] for t, cols in pks}

    df_fd_fk = pd.DataFrame(
        [
            {
                "child_table": t,
                "lhs": ",".join(cols),
                "rhs": ",".join(f"{c}" for c in pk_rhs.get(rt, []))
                if tuple(rcols) == tuple(pk_lhs.get(rt, ()))
                else "",
                "ref_table": rt,
            }
            for t, cols, rt, rcols in fks
        ],
        columns=["child_table", "lhs", "rhs", "ref_table"],
    )

    return {
        "FD from PK": df_fd_pk,
        "FD from UNIQUE": df_fd_uq,
        "FD from FK": df_fd_fk,
        "CHECK_RAW": pd.DataFrame(
            [{"table": t, "constraint": c, "clause": cl} for (t, c, cl) in checks],
            columns=["table", "constraint", "clause"],
        ),
    }


def print_relational_constraint_dfs(dfs: Dict[str, pd.DataFrame]):
    for k in ["FD from PK", "FD from UNIQUE", "FD from FK"]:
        print(f"\n{k}")
        df = dfs[k]
        if df.empty:
            print("(empty)")
        else:
            print(df.to_string(index=False))


def parse_check_to_domain_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse DuckDB CHECK clauses written as implications, including DuckDB-normalized forms:

      CHECK(((lhs != 'v') OR (rhs BETWEEN a AND b)))
      CHECK(((lhs != 'v') OR ((rhs >= a) AND (rhs <= b))))
      CHECK(((lhs != 'v') OR (rhs IN ('x','y',...))))

    Output:
      cid,group_id,lhs,op,value,rhs,domain_op,domain_value
    """
    rules = []
    cid_counter = itertools.count(1)

    def _normalize_list(list_inside_parens: str) -> str:
        items = []
        for tok in list_inside_parens.split(","):
            tok = tok.strip().strip("'").strip('"')
            if tok:
                items.append(tok)
        return "[" + ", ".join(items) + "]"

    def _add(cid, gid, lhs, lhs_op, lhs_val, rhs, rhs_op, rhs_val):
        rules.append(
            {
                "cid": cid,
                "group_id": gid,
                "lhs": lhs,
                "lhs_op": lhs_op,
                "lhs_value": lhs_val,
                "rhs": rhs,
                "rhs_op": rhs_op,
                "rhs_value": rhs_val,
            }
        )


    for _, row in df.iterrows():
        clause = str(row.get("clause", "")).strip()
        if not clause:
            continue

        # Skip non-CHECK style things that show up in your CHECK_RAW
        # e.g., "customer_id IS NOT NULL"
        if "CHECK" not in clause.upper():
            continue

        # Peel CHECK(...) wrapper
        clause = re.sub(r"^\s*CHECK\s*\((.*)\)\s*$", r"\1", clause, flags=re.IGNORECASE)

        # Peel redundant parentheses
        while clause.startswith("(") and clause.endswith(")"):
            inner = clause[1:-1].strip()
            if inner.count("(") == inner.count(")"):
                clause = inner
            else:
                break

        cid = next(cid_counter)
        NUM = r"[-+]?\d+(?:\.\d+)?"
        group_id = 0

        # -------------------------
        # NEW Pattern A:
        #   (LHS >= k) OR (RHS = v)
        #   (LHS >= k) OR (RHS IS NULL)
        # => LHS < k  ->  RHS = v / RHS IS NULL
        # We'll encode as: lhs "<k" (domain), rhs "=v" or "IS NULL"
        # -------------------------
        mA = re.search(
            r"^\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*>=\s*([0-9]+)\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+)\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mA:
            lhs_attr, k, rhs_attr, v = mA.group(1), mA.group(2), mA.group(3), mA.group(4)
            # cid = next(cid_counter)
            _add(cid, 0, lhs_attr, "<", k, rhs_attr, "=", v)
            continue

        mA_null = re.search(
            r"^\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*>=\s*([0-9]+)\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mA_null:
            lhs_attr, k, rhs_attr = mA_null.group(1), mA_null.group(2), mA_null.group(3)
            # cid = next(cid_counter)
            _add(cid, 0, lhs_attr, "<", k, rhs_attr, "=", "IS_NULL")   # 여기서 "IS","NULL" 대신 "=","NULL"
            continue
        
        # -------------------------
        # NEW Pattern A2 (symmetric):
        #   (LHS <= k) OR (RHS = v)
        # => LHS > k  ->  RHS = v
        # -------------------------
        mA2 = re.search(
            r"^\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*<=\s*([0-9]+)\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+)\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mA2:
            lhs_attr, k, rhs_attr, v = mA2.group(1), mA2.group(2), mA2.group(3), mA2.group(4)
            _add(cid, 0, lhs_attr, ">", k, rhs_attr, "=", v)
            continue

        mA2_null = re.search(
            r"^\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*<=\s*([0-9]+)\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mA2_null:
            lhs_attr, k, rhs_attr = mA2_null.group(1), mA2_null.group(2), mA2_null.group(3)
            _add(cid, 0, lhs_attr, ">", k, rhs_attr, "=", "IS_NULL")
            continue

        # -------------------------
        # NEW Pattern A2_rest:
        #   (LHS <= k) OR (rest)
        # => LHS > k -> rest
        # rest는 아래의 BETWEEN / >=&<= / IN / single-compare 파서로 분해해서 add
        # -------------------------
        mA2rest = re.search(
            rf"^\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*<=\s*({NUM})\s*\)*\s*OR\s*(.+)$",
            clause, flags=re.IGNORECASE
        )
        if mA2rest:
            lhs_attr, k, rest = mA2rest.group(1), mA2rest.group(2), mA2rest.group(3).strip()

            # rest 괄호 peel
            while rest.startswith("(") and rest.endswith(")"):
                inner = rest[1:-1].strip()
                if inner.count("(") == inner.count(")"):
                    rest = inner
                else:
                    break

            mb = re.search(rf"([A-Za-z_][A-Za-z0-9_]*)\s*BETWEEN\s*({NUM})\s*AND\s*({NUM})", rest, flags=re.IGNORECASE)
            if mb:
                rhs, a, b = mb.group(1), mb.group(2), mb.group(3)
                _add(cid, group_id, lhs_attr, ">", k, rhs, ">=", a)
                _add(cid, group_id, lhs_attr, ">", k, rhs, "<=", b)
                continue

            mb2 = re.search(
                rf"\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*>=\s*({NUM})\s*\)*\s*AND\s*\(*\s*\1\s*<=\s*({NUM})\s*\)*",
                rest, flags=re.IGNORECASE
            )
            if mb2:
                rhs, a, b = mb2.group(1), mb2.group(2), mb2.group(3)
                _add(cid, group_id, lhs_attr, ">", k, rhs, ">=", a)
                _add(cid, group_id, lhs_attr, ">", k, rhs, "<=", b)
                continue

            mi = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*IN\s*\(([^)]+)\)", rest, flags=re.IGNORECASE)
            if mi:
                rhs, inside = mi.group(1), mi.group(2)
                _add(cid, group_id, lhs_attr, ">", k, rhs, "IN", _normalize_list(inside))
                continue

            ms = re.search(rf"([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|=|!=|<>)\s*('([^']+)'|{NUM})", rest, flags=re.IGNORECASE)
            if ms:
                rhs, op, raw = ms.group(1), ms.group(2), ms.group(3).strip()
                val = raw.strip("'") if raw.startswith("'") and raw.endswith("'") else raw
                _add(cid, group_id, lhs_attr, ">", k, rhs, op, val)
                continue


        # -------------------------
        # NEW Pattern A_flipped:
        #   (k <= LHS) OR (RHS = v / RHS IS NULL)
        #  == (LHS >= k) OR ...
        # => LHS < k -> RHS ...
        # -------------------------
        mAf = re.search(
            rf"^\(*\s*({NUM})\s*<=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*({NUM})\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mAf:
            k, lhs_attr, rhs_attr, v = mAf.group(1), mAf.group(2), mAf.group(3), mAf.group(4)
            _add(cid, 0, lhs_attr, "<", k, rhs_attr, "=", v)
            continue

        mAf_null = re.search(
            rf"^\(*\s*({NUM})\s*<=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mAf_null:
            k, lhs_attr, rhs_attr = mAf_null.group(1), mAf_null.group(2), mAf_null.group(3)
            _add(cid, 0, lhs_attr, "<", k, rhs_attr, "=", "IS_NULL")
            continue


        # -------------------------
        # NEW Pattern A2_flipped:
        #   (k >= LHS) OR (RHS = v / RHS IS NULL)
        #  == (LHS <= k) OR ...
        # => LHS > k -> RHS ...
        # -------------------------
        mA2f = re.search(
            rf"^\(*\s*({NUM})\s*>=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*({NUM})\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mA2f:
            k, lhs_attr, rhs_attr, v = mA2f.group(1), mA2f.group(2), mA2f.group(3), mA2f.group(4)
            _add(cid, 0, lhs_attr, ">", k, rhs_attr, "=", v)
            continue

        mA2f_null = re.search(
            rf"^\(*\s*({NUM})\s*>=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mA2f_null:
            k, lhs_attr, rhs_attr = mA2f_null.group(1), mA2f_null.group(2), mA2f_null.group(3)
            _add(cid, 0, lhs_attr, ">", k, rhs_attr, "=", "IS_NULL")
            continue


        # -------------------------
        # NEW Pattern A3:
        #   (LHS < k) OR (RHS = v / RHS IS NULL)
        # => LHS >= k -> RHS ...
        # -------------------------
        mA3 = re.search(
            rf"^\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*<\s*({NUM})\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*({NUM})\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mA3:
            lhs_attr, k, rhs_attr, v = mA3.group(1), mA3.group(2), mA3.group(3), mA3.group(4)
            _add(cid, 0, lhs_attr, ">=", k, rhs_attr, "=", v)
            continue

        mA3_null = re.search(
            rf"^\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*<\s*({NUM})\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mA3_null:
            lhs_attr, k, rhs_attr = mA3_null.group(1), mA3_null.group(2), mA3_null.group(3)
            _add(cid, 0, lhs_attr, ">=", k, rhs_attr, "=", "IS_NULL")
            continue
        
        # -------------------------
        # NEW Pattern A4:
        #   (LHS > k) OR (RHS = v / RHS IS NULL)
        # => LHS <= k -> RHS ...
        # -------------------------
        mA4 = re.search(
            rf"^\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s*({NUM})\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*({NUM})\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mA4:
            lhs_attr, k, rhs_attr, v = mA4.group(1), mA4.group(2), mA4.group(3), mA4.group(4)
            _add(cid, 0, lhs_attr, "<=", k, rhs_attr, "=", v)
            continue

        mA4_null = re.search(
            rf"^\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s*({NUM})\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mA4_null:
            lhs_attr, k, rhs_attr = mA4_null.group(1), mA4_null.group(2), mA4_null.group(3)
            _add(cid, 0, lhs_attr, "<=", k, rhs_attr, "=", "IS_NULL")
            continue


        # -------------------------
        # NEW Pattern B:
        #   (LHS != k) OR (RHS = v)
        # => LHS = k -> RHS = v
        # -------------------------
        mB = re.search(
            r"^\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<>|!=)\s*([0-9]+)\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+)\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mB:
            lhs_attr, k, rhs_attr, v = mB.group(1), mB.group(3), mB.group(4), mB.group(5)
            # cid = next(cid_counter)
            _add(cid, 0, lhs_attr, "=", k, rhs_attr, "=", v)
            continue

        # -------------------------
        # NEW Pattern C:
        #   ((A IS NOT NULL) AND (B IS NOT NULL)) OR (C IS NULL)
        # => (A IS NULL -> C IS NULL) and (B IS NULL -> C IS NULL)
        # -------------------------
        mC = re.search(
            r"^\(*\s*\(\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NOT\s+NULL\s*\)*\s*\)\s*AND\s*\(\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NOT\s+NULL\s*\)*\s*\)\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if mC:
            a, b, c = mC.group(1), mC.group(2), mC.group(3)
            # cid = next(cid_counter)
            _add(cid, 0, a, "=", "IS_NULL", c, "=", "IS_NULL")
            _add(cid, 0, b, "=", "IS_NULL", c, "=", "IS_NULL")
            continue


        # parse_check_to_domain_rules() 안에서, CHECK(...) 껍질 벗긴 뒤에 clause 가지고 처리할 때
        # 기존 m = re.search(lhs != 'v' OR rest) 전에 아래를 먼저 넣기

        # Pattern 1: (LHS IS NOT NULL OR RHS IS NULL)
        # (LHS IS NOT NULL) OR (RHS IS NULL)  =>  LHS IS NULL -> RHS IS NULL
        m0 = re.search(
            r"^\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NOT\s+NULL\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if m0:
            lhs_attr = m0.group(1)
            rhs_attr = m0.group(2)
            # cid = next(cid_counter)
            _add(cid, 0, lhs_attr, "=", "IS_NULL", rhs_attr, "=", "IS_NULL")  # rhs도 "="로 통일
            continue


        # Pattern 2: (LHS <> k OR RHS IS NULL)  (k numeric)
        m1 = re.search(
            r"^\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<>|!=)\s*([0-9]+)\s*\)*\s*OR\s*\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\s*\)*$",
            clause, flags=re.IGNORECASE
        )
        if m1:
            lhs_attr, k, rhs_attr = m1.group(1), m1.group(3), m1.group(4)
            # cid = next(cid_counter)
            _add(cid, 0, lhs_attr, "=", k, rhs_attr, "=", "IS_NULL")
            continue



        # IMPORTANT: DuckDB uses "!=" (not "<>") in your output
        # Match: lhs != 'v' OR rest
        m = re.search(
            r"\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*(!=|<>)\s*'([^']+)'\s*\)*\s*OR\s*(.+)$",
            clause,
            flags=re.IGNORECASE,
        )

        if not m:
            continue

        lhs_attr = m.group(1)
        lhs_val = m.group(3)
        rest = m.group(4).strip()

        # Peel parentheses on rest
        while rest.startswith("(") and rest.endswith(")"):
            inner = rest[1:-1].strip()
            if inner.count("(") == inner.count(")"):
                rest = inner
            else:
                break

        # Case A: rhs BETWEEN a AND b
        mb = re.search(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*BETWEEN\s*([0-9]+)\s*AND\s*([0-9]+)",
            rest,
            flags=re.IGNORECASE,
        )
        if mb:
            rhs = mb.group(1)
            a = mb.group(2)
            b = mb.group(3)
            _add(cid, group_id, lhs_attr, "=", lhs_val, rhs, ">=", a)
            _add(cid, group_id, lhs_attr, "=", lhs_val, rhs, "<=", b)

            continue

        # Case B: (rhs >= a) AND (rhs <= b)  (DuckDB-normalized BETWEEN)
        mb2 = re.search(
            r"\(*\s*([A-Za-z_][A-Za-z0-9_]*)\s*>=\s*([0-9]+)\s*\)*\s*AND\s*\(*\s*\1\s*<=\s*([0-9]+)\s*\)*",
            rest,
            flags=re.IGNORECASE,
        )
        if mb2:
            rhs = mb2.group(1)
            a = mb2.group(2)
            b = mb2.group(3)
            _add(cid, group_id, lhs_attr, "=", lhs_val, rhs, ">=", a)
            _add(cid, group_id, lhs_attr, "=", lhs_val, rhs, "<=", b)

            continue

        # Case C: rhs IN (...)
        mi = re.search(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*IN\s*\(([^)]+)\)",
            rest,
            flags=re.IGNORECASE,
        )
        if mi:
            rhs = mi.group(1)
            inside = mi.group(2)
            _add(cid, group_id, lhs_attr, "=", lhs_val, rhs, "IN", _normalize_list(inside))
            continue


        # Case D: single comparison rhs >= a / <= b / = value
        ms = re.search(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|=)\s*('([^']+)'|[0-9]+)",
            rest,
            flags=re.IGNORECASE,
        )
        if ms:
            rhs = ms.group(1)
            op = ms.group(2)
            raw = ms.group(3).strip()
            val = raw.strip("'") if raw.startswith("'") and raw.endswith("'") else raw
            _add(cid, group_id, lhs_attr, "=", lhs_val, rhs, op, val)
            continue

    return pd.DataFrame(
        rules,
        columns=["cid","group_id","lhs","lhs_op","lhs_value","rhs","rhs_op","rhs_value"],
    )




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="constraint_schema"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="base directory",
    )

    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="duckdb file",
    )

    parser.add_argument(
        "--fd-s",
        type=str,
        required=True,
        help="fd schema csv",
    )

    parser.add_argument(
        "--domain-s",
        type=str,
        required=True,
        help="domain constraint schema csv",
    )

    args = parser.parse_args()
    db_path = os.path.join(args.base_dir, args.db)
    dfs = build_relational_constraint_dfs(db_path)
    print_relational_constraint_dfs(dfs)

    fd_s_path = os.path.join(args.base_dir, args.fd_s)
    os.makedirs(os.path.dirname(fd_s_path) or ".", exist_ok=True)

    parts = []
    for name, df in dfs.items():
        if df is None or df.empty:
            continue
        if "lhs" not in df.columns or "rhs" not in df.columns:
            continue

        tmp = df[["lhs", "rhs"]].copy()
        tmp["lhs"] = tmp["lhs"].astype(str).str.strip()
        tmp["rhs"] = tmp["rhs"].astype(str).str.strip()

        # tmp["lhs"] = tmp["lhs"].str.strip('"')
        # tmp["rhs"] = tmp["rhs"].str.strip('"')

        tmp = tmp[(tmp["lhs"] != "") & (tmp["rhs"] != "")]
        if tmp.empty:
            continue

        tmp["rhs"] = tmp["rhs"].apply(lambda s: [x.strip() for x in str(s).split(",") if x.strip()])
        tmp = tmp.explode("rhs", ignore_index=True)

        tmp["rhs"] = tmp["rhs"].astype(str).str.strip()
        tmp = tmp[(tmp["lhs"] != "") & (tmp["rhs"] != "")]

        parts.append(tmp)

    new_fd = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["lhs", "rhs"])
    merged = new_fd.copy()

    merged["lhs"] = merged["lhs"].astype(str).str.strip()
    merged["rhs"] = merged["rhs"].astype(str).str.strip()
    merged = merged[(merged["lhs"] != "") & (merged["rhs"] != "")]
    merged = merged.drop_duplicates().reset_index(drop=True)
    

    merged.to_csv(fd_s_path, index=False)

    check_raw = dfs.get("CHECK_RAW", pd.DataFrame())
    domain_df = parse_check_to_domain_rules(check_raw)

    print("\nDomain Constraints")
    print(check_raw)
    print(domain_df)
    domain_path = os.path.join(args.base_dir, args.domain_s)
    os.makedirs(os.path.dirname(domain_path) or ".", exist_ok=True)

    # ALWAYS write (even if empty) so you can see pipeline ran
    domain_df.to_csv(domain_path, index=False)
