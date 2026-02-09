import os
import re
import argparse
import pandas as pd
import sqlglot
from sqlglot import exp
import duckdb


def _sql(e):
    return e.sql(dialect="duckdb")


def _iter_expr_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [e for e in x if isinstance(e, exp.Expression)]
    exprs = getattr(x, "expressions", None)
    if isinstance(exprs, list):
        return [e for e in exprs if isinstance(e, exp.Expression)]
    if isinstance(x, exp.Expression):
        return [x]
    return []


def _output_name(proj):
    if isinstance(proj, exp.Alias) and proj.alias:
        return proj.alias
    if isinstance(proj, exp.Column):
        return proj.name
    return None


def _proj_expr(proj):
    return proj.this if isinstance(proj, exp.Alias) else proj


def _is_int_lit_1(e):
    return isinstance(e, exp.Literal) and e.is_int and e.this == "1"


def _is_agg_func(node):
    return isinstance(node, exp.AggFunc)


def _find_windows(expr):
    return list(expr.find_all(exp.Window))


def _window_is_row_number(win):
    return isinstance(win.this, exp.RowNumber)


def _window_is_agg(win):
    return isinstance(win.this, exp.AggFunc)

def load_pk_map_from_duckdb(db_path: str):
    """
    Returns: dict[str, set[str]]
      table_name -> {pk_col1, pk_col2, ...}
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute("""
            SELECT
              kcu.table_name,
              kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_name = kcu.table_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
        """).fetchall()
    finally:
        con.close()

    pk_map = {}
    for table_name, col_name in rows:
        pk_map.setdefault(str(table_name), set()).add(str(col_name))
    return pk_map

def _is_projection_expression(expr: exp.Expression) -> bool:
    """
    True  -> expression-based (keep)
    False -> simple projection / rename (drop later)
    """
    if isinstance(expr, exp.Column):
        return False

    if isinstance(expr, exp.Cast) and isinstance(expr.this, exp.Column):
        return False

    return True

def _extract_select_scope_tables(sel: exp.Select):
    alias_to_table = {}
    frm = sel.args.get("from")
    if frm:
        for tnode in frm.find_all(exp.Table):
            base = tnode.name
            a = tnode.args.get("alias")
            if a and a.name:
                alias_to_table[a.name] = base
            alias_to_table[base] = base
    for j in sel.args.get("joins") or []:
        for tnode in j.find_all(exp.Table):
            base = tnode.name
            a = tnode.args.get("alias")
            if a and a.name:
                alias_to_table[a.name] = base
            alias_to_table[base] = base
    return alias_to_table



def _detect_role_split_selfjoin(tree: exp.Expression) -> bool:
    """
    True if a base table appears >=2 times with different aliases
    in the SAME FROM/JOIN scope of any SELECT.
    (This matches ps/pp style 'self/partner' role-split.)
    """

    for sel in tree.find_all(exp.Select):
        # Map: base_table_name -> set(aliases_used_in_this_select_from)
        base2aliases = {}

        # Collect all table references in FROM/JOIN for this SELECT
        for tbl in sel.find_all(exp.Table):
            base = (tbl.name or "").lower()
            if not base:
                continue

            alias = (tbl.alias_or_name or "").lower()  # if no alias, same as base
            # IMPORTANT: if you want to require "real alias", enforce alias != base
            base2aliases.setdefault(base, set()).add(alias)

        # If any base has 2+ aliases in same SELECT scope -> role-split self-join
        for base, aliases in base2aliases.items():
            # require 2+ DISTINCT references (aliases could be same if repeated accidentally)
            if len(aliases) >= 2:
                return True

    return False



def _detect_self_join_in_tree(tree: exp.Expression) -> bool:
    """
    True only if:
      - a base table appears with >=2 aliases AND
      - there exists a join equality that connects two different aliases of that same base
        (i.e., alias-alias join within the same base).
    This avoids flagging pivot-style repeated joins (same table sliced by predicates).
    """
    alias_to_base, base_to_aliases = _build_global_alias_to_table_and_base_aliases(tree)

    # quick reject: no base has >=2 real aliases
    has_multi_alias_base = any(
        len({a for a in aliases if a != base}) >= 2
        for base, aliases in base_to_aliases.items()
    )
    if not has_multi_alias_base:
        return False

    # decisive check: do we ever join alias1 <-> alias2 where both map to same base?
    for j in tree.find_all(exp.Join):
        on = j.args.get("on")
        if not on:
            continue
        for eq in on.find_all(exp.EQ):
            l, r = eq.left, eq.right
            if not (isinstance(l, exp.Column) and isinstance(r, exp.Column)):
                continue

            la, ra = l.table, r.table
            if not la or not ra or la == ra:
                continue

            lb = alias_to_base.get(la, la)
            rb = alias_to_base.get(ra, ra)

            # self-join if the join connects two different aliases of the SAME base table
            if lb == rb:
                return True

    return False


def _build_subquery_lineage(tree: exp.Expression):
    subq_lineage = {}
    for sub in tree.find_all(exp.Subquery):
        sub_alias = sub.args.get("alias")
        sub_alias = sub_alias.name if sub_alias and sub_alias.name else None
        if not sub_alias:
            continue
        inner = sub.this
        if not isinstance(inner, exp.Select):
            continue

        inner_alias_map = _extract_select_scope_tables(inner)
        proj_map = {}

        for proj in inner.expressions:
            out = _output_name(proj)
            if not out:
                continue
            inner_expr = _proj_expr(proj)
            if isinstance(inner_expr, exp.Column):
                src_alias = inner_expr.table
                src_col = inner_expr.name
                base = inner_alias_map.get(src_alias, src_alias) if src_alias else None
                if base:
                    proj_map[out] = f"{base}.{src_col}"
                else:
                    proj_map[out] = src_col

        subq_lineage[sub_alias] = proj_map
    return subq_lineage


def _build_global_alias_to_table(tree: exp.Expression):
    alias_to_table = {}
    for tnode in tree.find_all(exp.Table):
        base = tnode.name
        a = tnode.args.get("alias")
        if a and a.name:
            alias_to_table[a.name] = base
        alias_to_table[base] = base
    return alias_to_table


def _resolve_col_str(col_str: str, global_alias_to_table, subq_lineage):
    s = str(col_str).strip()
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$", s)
    if not m:
        return None, s

    a, c = m.group(1), m.group(2)

    if a in subq_lineage and c in subq_lineage[a]:
        basecol = subq_lineage[a][c]
        m2 = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$", basecol)
        if m2:
            return m2.group(1), m2.group(2)
        return None, basecol

    base = global_alias_to_table.get(a, a)
    return base, c

def _build_global_alias_to_table_and_base_aliases(tree: exp.Expression):
    """
    Returns:
      alias_to_base: dict[str,str]
      base_to_aliases: dict[str,set[str]]
    """
    alias_to_base = {}
    base_to_aliases = {}

    for tnode in tree.find_all(exp.Table):
        base = tnode.name
        a = tnode.args.get("alias")
        if a and a.name:
            alias = a.name
            alias_to_base[alias] = base
            base_to_aliases.setdefault(base, set()).add(alias)

        # keep base->base for compatibility-like lookups
        alias_to_base[base] = base
        base_to_aliases.setdefault(base, set()).add(base)

    return alias_to_base, base_to_aliases


def _resolve_col_str_selfjoin(col_str: str, alias_to_base, base_to_aliases, subq_lineage):
    """
    Self-join mode: if a base table has multiple aliases overall, preserve alias as the "base key".
    That way e.age and m.age won't collapse.
    Returns: (role_key, col) where role_key is either alias (e/m) or true base.
    """
    s = str(col_str).strip()
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$", s)
    if not m:
        return None, s

    a, c = m.group(1), m.group(2)

    # subquery lineage: keep mapping if available
    if a in subq_lineage and c in subq_lineage[a]:
        basecol = subq_lineage[a][c]
        m2 = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$", basecol)
        if m2:
            # NOTE: basecol is already a "base.col" string. Treat it as role_key=base here.
            return m2.group(1), m2.group(2)
        return None, basecol

    base = alias_to_base.get(a, a)

    # if this base has multiple aliases overall => treat alias as role key
    aliases = base_to_aliases.get(base, set())
    if len({x for x in aliases if x != base}) >= 2:
        return a, c  # role-aware
    return base, c  # fallback


def _cols_in_expr_as_basecols(expr, global_alias_to_table, subq_lineage):
    cols = []
    for c in expr.find_all(exp.Column):
        base, col = _resolve_col_str(_sql(c), global_alias_to_table, subq_lineage)
        cols.append((base, col))
    uniq = sorted(set(cols), key=lambda x: ("" if x[0] is None else x[0], x[1]))
    return uniq


def _join_equalities(tree: exp.Expression, global_alias_to_table, subq_lineage):
    pairs = set()
    for j in tree.find_all(exp.Join):
        on = j.args.get("on")
        if not on:
            continue
        for eq in on.find_all(exp.EQ):
            l, r = eq.left, eq.right
            if not (isinstance(l, exp.Column) and isinstance(r, exp.Column)):
                continue
            lb, lc = _resolve_col_str(_sql(l), global_alias_to_table, subq_lineage)
            rb, rc = _resolve_col_str(_sql(r), global_alias_to_table, subq_lineage)
            pairs.add(((lb, lc), (rb, rc)))
    return pairs

def _cols_in_expr_as_rolecols(expr, alias_to_base, base_to_aliases, subq_lineage):
    cols = []
    for c in expr.find_all(exp.Column):
        role, col = _resolve_col_str_selfjoin(_sql(c), alias_to_base, base_to_aliases, subq_lineage)
        cols.append((role, col))
    uniq = sorted(set(cols), key=lambda x: ("" if x[0] is None else x[0], x[1]))
    return uniq


def _join_equalities_selfjoin(tree: exp.Expression, alias_to_base, base_to_aliases, subq_lineage):
    pairs = set()
    for j in tree.find_all(exp.Join):
        on = j.args.get("on")
        if not on:
            continue
        for eq in on.find_all(exp.EQ):
            l, r = eq.left, eq.right
            if not (isinstance(l, exp.Column) and isinstance(r, exp.Column)):
                continue
            lb, lc = _resolve_col_str_selfjoin(_sql(l), alias_to_base, base_to_aliases, subq_lineage)
            rb, rc = _resolve_col_str_selfjoin(_sql(r), alias_to_base, base_to_aliases, subq_lineage)
            pairs.add(((lb, lc), (rb, rc)))
    return pairs


def _detect_group_by_fds(sel: exp.Select, global_alias_to_table, subq_lineage, pk_map=None):
    out = []
    gb = sel.args.get("group")
    if not gb:
        return out

    keys = _iter_expr_list(getattr(gb, "expressions", gb))
    if not keys:
        return out

    # (base_table, col) list
    lhs_basecols = []
    for k in keys:
        lhs_basecols.extend(_cols_in_expr_as_basecols(k, global_alias_to_table, subq_lineage))
    lhs_basecols = sorted(set(lhs_basecols))  # list[(base, col)]

    if pk_map:
        by_base = {}
        for b, c in lhs_basecols:
            by_base.setdefault(b, set()).add(c)

        minimized = []
        for b, cols in by_base.items():
            if not b:
                for c in sorted(cols):
                    minimized.append((b, c))
                continue

            pk = pk_map.get(b, set())
            if pk and pk.issubset(cols):
                for c in sorted(pk):
                    minimized.append((b, c))
            else:
                for c in sorted(cols):
                    minimized.append((b, c))

        lhs_basecols = tuple(sorted(set(minimized)))
    else:
        lhs_basecols = tuple(sorted(set(lhs_basecols)))

    for proj in sel.expressions:
        rhs = _output_name(proj)
        if not rhs:
            continue
        expr = _proj_expr(proj)
        if any(_is_agg_func(x) for x in expr.walk()):
            out.append((lhs_basecols, rhs, "group_by"))
    return out

def _detect_group_by_fds_selfjoin(
    sel: exp.Select,
    alias_to_base,
    base_to_aliases,
    subq_lineage,
    pk_map=None,
):
    out = []
    gb = sel.args.get("group")
    if not gb:
        return out

    keys = _iter_expr_list(getattr(gb, "expressions", gb))
    if not keys:
        return out

    # (role_key, col) list
    lhs_rolecols = []
    for k in keys:
        lhs_rolecols.extend(_cols_in_expr_as_rolecols(k, alias_to_base, base_to_aliases, subq_lineage))
    lhs_rolecols = sorted(set(lhs_rolecols))

    if pk_map:
        # role-wise
        by_role = {}
        for role, c in lhs_rolecols:
            by_role.setdefault(role, set()).add(c)

        minimized = []
        for role, cols in by_role.items():
            if not role:
                for c in sorted(cols):
                    minimized.append((role, c))
                continue

            base = alias_to_base.get(role, role)  # role->base
            pk = pk_map.get(base, set())

            if pk and pk.issubset(cols):
                for c in sorted(pk):
                    minimized.append((role, c))
            else:
                for c in sorted(cols):
                    minimized.append((role, c))

        lhs_rolecols = tuple(sorted(set(minimized)))
    else:
        lhs_rolecols = tuple(sorted(set(lhs_rolecols)))

    for proj in sel.expressions:
        rhs = _output_name(proj)
        if not rhs:
            continue
        expr = _proj_expr(proj)
        if any(_is_agg_func(x) for x in expr.walk()):
            out.append((lhs_rolecols, rhs, "group_by"))
    return out


def _detect_window_agg_fds(sel: exp.Select, global_alias_to_table, subq_lineage):
    out = []
    for proj in sel.expressions:
        rhs = _output_name(proj)
        if not rhs:
            continue
        expr = _proj_expr(proj)
        wins = _find_windows(expr)
        for w in wins:
            if not _window_is_agg(w):
                continue
            part = _iter_expr_list(w.args.get("partition_by"))
            if not part:
                continue
            lhs_basecols = []
            for p in part:
                lhs_basecols.extend(_cols_in_expr_as_basecols(p, global_alias_to_table, subq_lineage))
            lhs_basecols = tuple(sorted(set(lhs_basecols)))
            out.append((lhs_basecols, rhs, "window_agg"))
    return out

def _detect_window_agg_fds_selfjoin(
    sel: exp.Select,
    alias_to_base,
    base_to_aliases,
    subq_lineage,
):
    out = []
    for proj in sel.expressions:
        rhs = _output_name(proj)
        if not rhs:
            continue
        expr = _proj_expr(proj)
        wins = _find_windows(expr)
        for w in wins:
            if not _window_is_agg(w):
                continue
            part = _iter_expr_list(w.args.get("partition_by"))
            if not part:
                continue

            lhs_rolecols = []
            for p in part:
                lhs_rolecols.extend(_cols_in_expr_as_rolecols(p, alias_to_base, base_to_aliases, subq_lineage))
            lhs_rolecols = tuple(sorted(set(lhs_rolecols)))

            out.append((lhs_rolecols, rhs, "window_agg"))
    return out


def _detect_top1_fds(sel: exp.Select, global_alias_to_table, subq_lineage):
    out = []
    q = sel.args.get("qualify")
    if not q or not q.this:
        return out

    pred = q.this
    if not isinstance(pred, exp.EQ):
        return out

    l, r = pred.left, pred.right
    if _is_int_lit_1(l):
        l, r = r, l
    if not _is_int_lit_1(r):
        return out

    win = l.find(exp.Window) if l else None
    if not win or not _window_is_row_number(win):
        return out

    part = _iter_expr_list(win.args.get("partition_by"))
    if not part:
        return out

    lhs_basecols = []
    for p in part:
        lhs_basecols.extend(_cols_in_expr_as_basecols(p, global_alias_to_table, subq_lineage))
    lhs_basecols = tuple(sorted(set(lhs_basecols)))
    lhs_names = set(col for _, col in lhs_basecols)

    order_by_base_tables = set()
    order_node = win.args.get("order")
    if order_node:
        for od in _iter_expr_list(getattr(order_node, "expressions", order_node)):
            for b, c in _cols_in_expr_as_basecols(od, global_alias_to_table, subq_lineage):
                if b:
                    order_by_base_tables.add(b)

    for proj in sel.expressions:
        rhs = _output_name(proj)
        if not rhs:
            continue
        expr = _proj_expr(proj)
        if any(_is_agg_func(x) for x in expr.walk()):
            continue
        rhs_cols = _cols_in_expr_as_basecols(expr, global_alias_to_table, subq_lineage)
        if not rhs_cols:
            continue

        rhs_bases = {b for b, _ in rhs_cols if b}
        if order_by_base_tables and not (rhs_bases <= order_by_base_tables):
            continue

        if rhs in lhs_names:
            continue

        out.append((lhs_basecols, rhs, "top1"))
    return out

def _detect_top1_fds_selfjoin(
    sel: exp.Select,
    alias_to_base,
    base_to_aliases,
    subq_lineage,
):
    out = []
    q = sel.args.get("qualify")
    if not q or not q.this:
        return out

    pred = q.this
    if not isinstance(pred, exp.EQ):
        return out

    l, r = pred.left, pred.right
    if _is_int_lit_1(l):
        l, r = r, l
    if not _is_int_lit_1(r):
        return out

    win = l.find(exp.Window) if l else None
    if not win or not _window_is_row_number(win):
        return out

    part = _iter_expr_list(win.args.get("partition_by"))
    if not part:
        return out

    lhs_rolecols = []
    for p in part:
        lhs_rolecols.extend(_cols_in_expr_as_rolecols(p, alias_to_base, base_to_aliases, subq_lineage))
    lhs_rolecols = tuple(sorted(set(lhs_rolecols)))

    lhs_names = set(col for _, col in lhs_rolecols)

    # 기존: order_by_base_tables
    order_by_roles = set()
    order_node = win.args.get("order")
    if order_node:
        for od in _iter_expr_list(getattr(order_node, "expressions", order_node)):
            for role, c in _cols_in_expr_as_rolecols(od, alias_to_base, base_to_aliases, subq_lineage):
                if role:
                    order_by_roles.add(role)

    for proj in sel.expressions:
        rhs = _output_name(proj)
        if not rhs:
            continue
        expr = _proj_expr(proj)

        if any(_is_agg_func(x) for x in expr.walk()):
            continue

        rhs_rolecols = _cols_in_expr_as_rolecols(expr, alias_to_base, base_to_aliases, subq_lineage)
        if not rhs_rolecols:
            continue

        rhs_roles = {role for role, _ in rhs_rolecols if role}

        if order_by_roles and not (rhs_roles <= order_by_roles):
            continue

        if rhs in lhs_names:
            continue

        out.append((lhs_rolecols, rhs, "top1"))

    return out


def _detect_simple_deterministic_fds(
    sel: exp.Select,
    global_alias_to_table,
    subq_lineage,
    existing_pairs=None,
):
    out = []
    existing_pairs = existing_pairs or set()

    for proj in sel.expressions:
        rhs = _output_name(proj)
        if not rhs:
            continue
        expr = _proj_expr(proj)

        if any(_is_agg_func(x) for x in expr.walk()):
            continue
        if any(isinstance(x, exp.Window) for x in expr.walk()):
            continue

        basecols = _cols_in_expr_as_basecols(expr, global_alias_to_table, subq_lineage)
        if not basecols:
            continue

        if len(basecols) == 1:
            b, c = basecols[0]
            if c == rhs:
                continue

        lhs_key = tuple(basecols)
        if (lhs_key, rhs) in existing_pairs:
            continue

        kind = (
            "projection_expression"
            if _is_projection_expression(expr)
            else "projection"
        )
        out.append((lhs_key, rhs, kind))

    return out

def _detect_simple_deterministic_fds_selfjoin(
    sel: exp.Select,
    alias_to_base,
    base_to_aliases,
    subq_lineage,
    existing_pairs=None,
):
    out = []
    existing_pairs = existing_pairs or set()

    for proj in sel.expressions:
        rhs = _output_name(proj)
        if not rhs:
            continue
        expr = _proj_expr(proj)

        if any(_is_agg_func(x) for x in expr.walk()):
            continue
        if any(isinstance(x, exp.Window) for x in expr.walk()):
            continue

        rolecols = _cols_in_expr_as_rolecols(expr, alias_to_base, base_to_aliases, subq_lineage)
        if not rolecols:
            continue

        if len(rolecols) == 1:
            _, c = rolecols[0]
            if c == rhs:
                continue

        lhs_key = tuple(rolecols)
        if (lhs_key, rhs) in existing_pairs:
            continue

        kind = (
            "projection_expression"
            if _is_projection_expression(expr)
            else "projection"
        )
        out.append((lhs_key, rhs, kind))

    return out


def _lhs_basecols_to_base(lhs_basecols):
    bases = [b for b, _ in lhs_basecols if b]
    if not bases:
        return None
    return ",".join(sorted(set(bases)))


def _is_col_lit_eq(e):
    return isinstance(e, exp.EQ) and isinstance(e.left, exp.Column) and isinstance(e.right, exp.Literal)


def _is_col_lit_neq(e):
    return isinstance(e, exp.NEQ) and isinstance(e.left, exp.Column) and isinstance(e.right, exp.Literal)


def _unparen(e):
    while isinstance(e, exp.Paren):
        e = e.this
    return e


def _negate_guard_to_eq(guard):
    guard = _unparen(guard)
    if _is_col_lit_neq(guard):
        return guard.left, guard.right
    if isinstance(guard, exp.Not):
        inner = _unparen(guard.this)
        if _is_col_lit_eq(inner):
            return inner.left, inner.right
    return None


def _guard_eq(guard):
    guard = _unparen(guard)
    if _is_col_lit_eq(guard):
        return guard.left, guard.right
    return None


def _flatten_or(expr):
    if isinstance(expr, exp.Or):
        return _flatten_or(expr.left) + _flatten_or(expr.right)
    return [expr]


def _flatten_and(expr):
    if isinstance(expr, exp.And):
        return _flatten_and(expr.left) + _flatten_and(expr.right)
    return [expr]


def _flatten_and_nodes(expr):
    expr = _unparen(expr)
    if isinstance(expr, exp.And):
        return _flatten_and_nodes(expr.left) + _flatten_and_nodes(expr.right)
    return [expr]


def _lit_to_py(lit: exp.Literal):
    return lit.this


def _is_trueish(expr):
    if expr is None:
        return False
    s = _sql(expr).strip().upper()
    return s in ("TRUE", "1=1", "(1=1)")


def _emit_atomic_domain_rows(constraint_expr, global_alias_to_table, subq_lineage):
    out = []
    atoms = _flatten_and_nodes(constraint_expr)

    for a in atoms:
        a = _unparen(a)

        if isinstance(a, exp.Between) and isinstance(a.this, exp.Column):
            _, rhs = _resolve_col_str(_sql(a.this), global_alias_to_table, subq_lineage)
            lo = a.args.get("low")
            hi = a.args.get("high")
            if isinstance(lo, exp.Literal):
                out.append((rhs, ">=", _lit_to_py(lo)))
            if isinstance(hi, exp.Literal):
                out.append((rhs, "<=", _lit_to_py(hi)))
            continue

        if isinstance(a, exp.In) and isinstance(a.this, exp.Column):
            _, rhs = _resolve_col_str(_sql(a.this), global_alias_to_table, subq_lineage)
            vals = []
            for v in (a.args.get("expressions") or []):
                if isinstance(v, exp.Literal):
                    vals.append(_lit_to_py(v))
                else:
                    vals.append(_sql(v))
            out.append((rhs, "IN", "[" + ", ".join(vals) + "]"))
            continue

        def _binop(op_str, node_type):
            if isinstance(a, node_type) and isinstance(a.left, exp.Column) and isinstance(a.right, exp.Literal):
                _, rhs = _resolve_col_str(_sql(a.left), global_alias_to_table, subq_lineage)
                out.append((rhs, op_str, _lit_to_py(a.right)))
                return True
            return False

        if _binop("=", exp.EQ):
            continue
        if _binop("!=", exp.NEQ):
            continue
        if _binop(">=", exp.GTE):
            continue
        if _binop("<=", exp.LTE):
            continue
        if _binop(">", exp.GT):
            continue
        if _binop("<", exp.LT):
            continue

    return out


def _rows_from_guard_and_constraint(guard_expr, constraint_expr, global_alias_to_table, subq_lineage, source):
    guard_expr = _unparen(guard_expr)
    eq = _guard_eq(guard_expr)
    if not eq:
        return []
    col_node, lit_node = eq
    base, guard_col = _resolve_col_str(_sql(col_node), global_alias_to_table, subq_lineage)
    if not guard_col:
        return []
    return [
        {
            "table": base,
            "guard_col": guard_col,
            "guard_op": "=",
            "guard_val": lit_node.this if isinstance(lit_node, exp.Literal) else _sql(lit_node),
            "constraint_expr": constraint_expr,
            "source": source,
        }
    ]


def _rows_from_guard_and_constraint_any(guard_expr, constraint_expr, global_alias_to_table, subq_lineage, source):
    """
    Like _rows_from_guard_and_constraint, but also allows negated guard:
      - guard: col = lit   -> guard_op '='
      - guard: col != lit  -> guard_op '!='
      - guard: NOT(col = lit) -> guard_op '!='
    Existing function remains unchanged; this is additive.
    """
    guard_expr = _unparen(guard_expr)

    # 1) col = lit
    eq = _guard_eq(guard_expr)
    if eq:
        col_node, lit_node = eq
        base, guard_col = _resolve_col_str(_sql(col_node), global_alias_to_table, subq_lineage)
        if not guard_col:
            return []
        return [{
            "table": base,
            "guard_col": guard_col,
            "guard_op": "=",
            "guard_val": lit_node.this if isinstance(lit_node, exp.Literal) else _sql(lit_node),
            "constraint_expr": constraint_expr,
            "source": source,
        }]

    # 2) col != lit  OR  NOT(col = lit)
    neg = _negate_guard_to_eq(guard_expr)
    if neg:
        col_node, lit_node = neg
        base, guard_col = _resolve_col_str(_sql(col_node), global_alias_to_table, subq_lineage)
        if not guard_col:
            return []
        return [{
            "table": base,
            "guard_col": guard_col,
            "guard_op": "!=",
            "guard_val": lit_node.this if isinstance(lit_node, exp.Literal) else _sql(lit_node),
            "constraint_expr": constraint_expr,
            "source": source,
        }]

    return []


def _extract_case_if_conditionals(pred, global_alias_to_table, subq_lineage, source):
    rows = []

    for case in pred.find_all(exp.Case):
        default = case.args.get("default")
        if default is None or not _is_trueish(default):
            continue
        ifs = case.args.get("ifs") or []
        for ifnode in ifs:
            if not isinstance(ifnode, exp.If):
                continue
            cond = ifnode.args.get("this")
            then_expr = ifnode.args.get("true")
            if cond is None or then_expr is None:
                continue
            if _is_trueish(then_expr):
                continue
            rows += _rows_from_guard_and_constraint(cond, then_expr, global_alias_to_table, subq_lineage, source)

    for ifnode in pred.find_all(exp.If):
        cond = ifnode.args.get("this")
        then_expr = ifnode.args.get("true")
        else_expr = ifnode.args.get("false")
        if cond is None or then_expr is None or else_expr is None:
            continue
        if not _is_trueish(else_expr):
            continue
        if _is_trueish(then_expr):
            continue
        rows += _rows_from_guard_and_constraint(cond, then_expr, global_alias_to_table, subq_lineage, source)

    # =========================
    # NEW (additive): handle inverted CASE/IF patterns
    #  - CASE WHEN guard THEN TRUE ELSE constraint END  -> (NOT guard) -> constraint
    #  - IF(guard, TRUE, constraint)                    -> (NOT guard) -> constraint
    # Existing behavior unchanged.
    # =========================

    for case in pred.find_all(exp.Case):
        default = case.args.get("default")
        if default is None:
            continue

        ifs = case.args.get("ifs") or []
        for ifnode in ifs:
            if not isinstance(ifnode, exp.If):
                continue
            cond = ifnode.args.get("this")
            then_expr = ifnode.args.get("true")
            if cond is None or then_expr is None:
                continue

            # CASE WHEN cond THEN TRUE ELSE <constraint> END
            if _is_trueish(then_expr) and not _is_trueish(default):
                # NOT(cond) is the guard for implication
                rows += _rows_from_guard_and_constraint_any(
                    exp.Not(this=cond),
                    default,
                    global_alias_to_table,
                    subq_lineage,
                    source,
                )

    for ifnode in pred.find_all(exp.If):
        cond = ifnode.args.get("this")
        then_expr = ifnode.args.get("true")
        else_expr = ifnode.args.get("false")
        if cond is None or then_expr is None or else_expr is None:
            continue

        # IF(cond, TRUE, constraint)
        if _is_trueish(then_expr) and not _is_trueish(else_expr):
            rows += _rows_from_guard_and_constraint_any(
                exp.Not(this=cond),
                else_expr,
                global_alias_to_table,
                subq_lineage,
                source,
            )

    return rows


def _extract_conditional_from_predicate(pred, global_alias_to_table, subq_lineage, source="where"):
    rows = []

    rows += _extract_case_if_conditionals(pred, global_alias_to_table, subq_lineage, source)

    for or_node in pred.find_all(exp.Or):
        disj = _flatten_or(or_node)
        for i, g in enumerate(disj):
            neg = _negate_guard_to_eq(g)
            if not neg:
                continue
            col_node, lit_node = neg

            others = [d for j, d in enumerate(disj) if j != i]
            if not others:
                continue
            constraint_expr = others[0]
            for extra in others[1:]:
                constraint_expr = exp.Or(this=constraint_expr, expression=extra)

            base, guard_col = _resolve_col_str(_sql(col_node), global_alias_to_table, subq_lineage)
            if not guard_col:
                continue

            rows.append(
                {
                    "table": base,
                    "guard_col": guard_col,
                    "guard_op": "=",
                    "guard_val": lit_node.this if isinstance(lit_node, exp.Literal) else _sql(lit_node),
                    "constraint_expr": constraint_expr,
                    "source": source,
                }
            )

    for and_node in pred.find_all(exp.And):
        conj = _flatten_and(and_node)
        for i, g in enumerate(conj):
            eq = _guard_eq(g)
            if not eq:
                continue
            col_node, lit_node = eq

            others = [d for j, d in enumerate(conj) if j != i]
            if not others:
                continue
            constraint_expr = others[0]
            for extra in others[1:]:
                constraint_expr = exp.And(this=constraint_expr, expression=extra)

            base, guard_col = _resolve_col_str(_sql(col_node), global_alias_to_table, subq_lineage)
            if not guard_col:
                continue

            rows.append(
                {
                    "table": base,
                    "guard_col": guard_col,
                    "guard_op": "=",
                    "guard_val": lit_node.this if isinstance(lit_node, exp.Literal) else _sql(lit_node),
                    "constraint_expr": constraint_expr,
                    "source": source,
                }
            )

    return rows


def infer_join_and_fd_and_domain(SQL: str, pk_map=None):
    tree = sqlglot.parse_one(SQL, dialect="duckdb")

    subq_lineage = _build_subquery_lineage(tree)
    global_alias_to_table = _build_global_alias_to_table(tree)

    join_pairs = _join_equalities(tree, global_alias_to_table, subq_lineage)

    join_rows = []
    for (lb, lc), (rb, rc) in sorted(join_pairs, key=lambda x: (str(x[0]), str(x[1]))):
        join_rows.append({"lhs": lc, "rhs": rc, "lhs_base": lb, "rhs_base": rb})

    join_df = pd.DataFrame(join_rows, columns=["lhs", "rhs", "lhs_base", "rhs_base"]).drop_duplicates()
    join_df = join_df.sort_values(["lhs_base", "rhs_base", "lhs", "rhs"]).reset_index(drop=True)

    fds = []
    for sel in tree.find_all(exp.Select):
        fds_sel = []
        fds_sel += _detect_group_by_fds(sel, global_alias_to_table, subq_lineage, pk_map=pk_map)
        fds_sel += _detect_window_agg_fds(sel, global_alias_to_table, subq_lineage)
        fds_sel += _detect_top1_fds(sel, global_alias_to_table, subq_lineage)

        existing_pairs = {(tuple(lhs), rhs) for (lhs, rhs, _) in fds_sel}  # NEW

        fds_sel += _detect_simple_deterministic_fds(
            sel, global_alias_to_table, subq_lineage, existing_pairs=existing_pairs
        )

        fds += fds_sel


    cleaned = []
    seen = set()

    for lhs_basecols, rhs, kind in fds:
        lhs_basecols = tuple(lhs_basecols)
        lhs_names = tuple(c for _, c in lhs_basecols)
        # removing trivial rules
        # if rhs in lhs_names:
        #     continue 

        key = (lhs_basecols, rhs, kind)
        if key in seen:
            continue
        seen.add(key)

        lhs_base_str = _lhs_basecols_to_base(lhs_basecols)

        if kind in ("window_agg", "group_by", "top1"):
            rhs_base_col = None
        else:
            rhs_base_col = rhs

        cleaned.append(
            {"lhs": ", ".join(lhs_names), "rhs": rhs, "lhs_base": lhs_base_str, "rhs_base": rhs_base_col, "kind": kind}
        )

    fd_df = pd.DataFrame(cleaned, columns=["lhs", "rhs", "lhs_base", "rhs_base", "kind"])
    fd_df = fd_df.where(pd.notna(fd_df), None)
    fd_df = fd_df.sort_values(["lhs_base", "rhs_base", "lhs", "rhs"], na_position="last").reset_index(drop=True)

    domain_rows = []
    for sel in tree.find_all(exp.Select):
        w = sel.args.get("where")
        if w and w.this:
            domain_rows += _extract_conditional_from_predicate(
                w.this, global_alias_to_table, subq_lineage, source="where"
            )

        for j in sel.args.get("joins") or []:
            on = j.args.get("on")
            if on:
                domain_rows += _extract_conditional_from_predicate(
                    on, global_alias_to_table, subq_lineage, source="join_on"
                )

    domain_df = pd.DataFrame(
        domain_rows,
        columns=["table", "guard_col", "guard_op", "guard_val", "constraint_expr", "source"],
    )
    domain_df = domain_df.where(pd.notna(domain_df), None)
    if len(domain_df) > 0:
        domain_df = domain_df.drop_duplicates().reset_index(drop=True)

    return join_df, fd_df, domain_df, global_alias_to_table, subq_lineage

def infer_join_and_fd_and_domain_selfjoin(SQL: str, pk_map=None):
    tree = sqlglot.parse_one(SQL, dialect="duckdb")

    subq_lineage = _build_subquery_lineage(tree)
    alias_to_base, base_to_aliases = _build_global_alias_to_table_and_base_aliases(tree)

    # JOIN (role-aware)
    join_pairs = _join_equalities_selfjoin(tree, alias_to_base, base_to_aliases, subq_lineage)

    join_rows = []
    for (lb, lc), (rb, rc) in sorted(join_pairs, key=lambda x: (str(x[0]), str(x[1]))):
        join_rows.append({
            "lhs": lc, "rhs": rc,
            "lhs_role": lb, "rhs_role": rb,
            "lhs_base": alias_to_base.get(lb, lb) if lb else None,
            "rhs_base": alias_to_base.get(rb, rb) if rb else None,
        })

    join_df = pd.DataFrame(join_rows).drop_duplicates()
    if len(join_df) > 0:
        join_df = join_df.sort_values(["lhs_base","rhs_base","lhs_role","rhs_role","lhs","rhs"]).reset_index(drop=True)

    # FD (role-aware)
    fds = []
    for sel in tree.find_all(exp.Select):
        fds_sel = []
        fds_sel += _detect_group_by_fds_selfjoin(sel, alias_to_base, base_to_aliases, subq_lineage, pk_map=pk_map)
        fds_sel += _detect_window_agg_fds_selfjoin(sel, alias_to_base, base_to_aliases, subq_lineage)
        fds_sel += _detect_top1_fds_selfjoin(sel, alias_to_base, base_to_aliases, subq_lineage)

        existing_pairs = {(tuple(lhs), rhs) for (lhs, rhs, _) in fds_sel}  # NEW

        fds_sel += _detect_simple_deterministic_fds_selfjoin(
            sel, alias_to_base, base_to_aliases, subq_lineage, existing_pairs=existing_pairs
        )

        fds += fds_sel



    cleaned = []
    seen = set()
    for lhs_rolecols, rhs, kind in fds:
        lhs_rolecols = tuple(lhs_rolecols)

        key = (lhs_rolecols, rhs, kind)
        if key in seen:
            continue
        seen.add(key)

        lhs_names = tuple(f"{role}.{col}" if role else col for role, col in lhs_rolecols)

        cleaned.append({
            "lhs": ", ".join(lhs_names),
            "rhs": rhs,
            "kind": kind,
        })

    fd_df = pd.DataFrame(cleaned, columns=["lhs","rhs","kind"])
    if len(fd_df) > 0:
        fd_df = fd_df.drop_duplicates().sort_values(["kind","lhs","rhs"]).reset_index(drop=True)

    _, _, domain_df, global_alias_to_table_normal, subq_lineage_normal = infer_join_and_fd_and_domain(SQL, pk_map=pk_map)
    # return 6-tuple to match main unpack:
    # global_alias_to_table_normal/subq_lineage_normal: domain/printing
    # alias_to_base: self-join role
    return join_df, fd_df, domain_df, global_alias_to_table_normal, subq_lineage_normal, alias_to_base

def infer_join_and_fd_and_domain_dispatch(SQL: str, pk_map=None):
    tree = sqlglot.parse_one(SQL, dialect="duckdb")
    if _detect_self_join_in_tree(tree) or _detect_role_split_selfjoin(tree):
        return infer_join_and_fd_and_domain_selfjoin(SQL, pk_map), True
    else:
        # normal mode (existing behavior unchanged)
        return infer_join_and_fd_and_domain(SQL, pk_map=pk_map), False

def postprocess_selfjoin_fds(fd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input fd_df columns: ["lhs","rhs","kind"] where lhs like "e.age, m.age"
    Output columns: ["lhs","rhs","kind","lhs_alias","rhs_alias"]

    Policy:
      - Drop single-column projection_fd of form "<role>.<col> -> <outcol>" (rename/projection)
      - Use those dropped rows to build rolecol->outcol map
      - Rewrite remaining rows' lhs from role.col to outcol when mapping exists
      - Keep only rows that are "final-usable": after rewrite, lhs contains output columns (no role prefixes)
    """
    if fd_df is None or len(fd_df) == 0:
        return pd.DataFrame(columns=["lhs","rhs","kind","lhs_alias","rhs_alias"])

    df = fd_df.copy()

    # --- helper parse lhs tokens like "e.age, m.age" -> [("e","age"), ("m","age")]
    def parse_rolecols(lhs_str: str):
        toks = [t.strip() for t in str(lhs_str).split(",") if t.strip()]
        out = []
        for t in toks:
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$", t)
            if m:
                out.append((m.group(1), m.group(2)))
            else:
                out.append((None, t))  # fallback
        return out

    # 1) Build rolecol -> outcol map from single-column projection
    rolecol_to_outcol = {}
    mask_single_proj = df["kind"].isin(["projection", "projection_expression"]) & df["lhs"].astype(str).str.match(
        r"^\s*[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\s*$"
    )
    for _, r in df[mask_single_proj].iterrows():
        role, col = parse_rolecols(r["lhs"])[0]
        if role is None:
            continue
        outcol = str(r["rhs"]).strip()
        rolecol_to_outcol[(role, col)] = outcol

    # 2) Drop those single-column projection rows from the exported set
    df = df[~mask_single_proj].reset_index(drop=True)

    # 3) Rewrite remaining rows
    out_rows = []
    for _, r in df.iterrows():
        kind = str(r["kind"]).strip()
        rhs = str(r["rhs"]).strip()

        rolecols = parse_rolecols(r["lhs"])
        lhs_aliases = [role for role, _ in rolecols if role is not None]
        rhs_aliases = sorted(set(lhs_aliases))

        # replace role.col -> output col if mapped, else keep raw token (will be filtered out later)
        lhs_out_tokens = []
        lhs_out_aliases = []

        roles_in_lhs = [role for role, _ in rolecols if role is not None]
        unique_roles = sorted(set(roles_in_lhs))

        for role, col in rolecols:
            if role is not None and (role, col) in rolecol_to_outcol:
                lhs_out_tokens.append(rolecol_to_outcol[(role, col)])
                lhs_out_aliases.append(role)
            else:
                if role is not None and len(unique_roles) == 1:
                    lhs_out_tokens.append(col)
                    lhs_out_aliases.append(role)
                else:
                    lhs_out_tokens.append(f"{role}.{col}" if role else col)
                    if role:
                        lhs_out_aliases.append(role)

        # de-dup but keep stable order
        seen = set()
        lhs_out_tokens2, lhs_out_aliases2 = [], []
        for tok, al in zip(lhs_out_tokens, lhs_out_aliases + [None] * (len(lhs_out_tokens) - len(lhs_out_aliases))):
            if tok in seen:
                continue
            seen.add(tok)
            lhs_out_tokens2.append(tok)
        # aliases: just unique roles used
        lhs_alias_str = ",".join(sorted(set(lhs_out_aliases))) if lhs_out_aliases else None
        rhs_alias_str = ",".join(rhs_aliases) if rhs_aliases else None

        lhs_out = ", ".join(lhs_out_tokens2)

        out_rows.append({
            "lhs": lhs_out,
            "rhs": rhs,
            "kind": kind,
            "lhs_alias": lhs_alias_str,
            "rhs_alias": rhs_alias_str,
        })

    out = pd.DataFrame(out_rows, columns=["lhs","rhs","kind","lhs_alias","rhs_alias"])

    # 4) Keep only "final usable" rows: lhs tokens should NOT contain role prefixes anymore
    #    (i.e. discared e.age)
    out = out[~out["lhs"].astype(str).str.contains(r"\b[A-Za-z_][A-Za-z0-9_]*\.", regex=True)]

    out = out.drop_duplicates().reset_index(drop=True)
    return out

def add_alias_cols_for_print(fd_df: pd.DataFrame) -> pd.DataFrame:
    """
    fd_df: columns = [lhs, rhs, kind]  (lhs 예: "e.age, m.age")
    returns: + [lhs_alias, rhs_alias] 추가된 copy
    """
    out = fd_df.copy()

    def parse_roles(lhs: str):
        if lhs is None:
            return []
        roles = []
        for tok in str(lhs).split(","):
            tok = tok.strip()
            if "." in tok:
                role = tok.split(".", 1)[0].strip()
                if role:
                    roles.append(role)
        # unique preserve order
        seen = set()
        uniq = []
        for r in roles:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        return uniq

    out["lhs_alias"] = out["lhs"].apply(lambda s: ", ".join(parse_roles(s)))
    out["rhs_alias"] = out["lhs_alias"]

    return out


# =========================================================
# NEW: Write-only (overwrite) outputs for this run
# =========================================================
def write_fd_csv_overwrite(fd_df: pd.DataFrame, join_df: pd.DataFrame, csv_path: str):
    """
    Writes a fresh lhs,rhs CSV (overwrite).
    - If sibling "<stem>_add.csv" exists, seed output with that content first.
    - Then combines fd_df + join_df
    - Explodes comma-separated rhs
    - Dedups within this run
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    # --- NEW: seed with *_add.csv if exists ---
    stem, ext = os.path.splitext(csv_path)
    add_path = stem + "_add" + ext  # e.g., fd_query_add.csv

    frames = []

    if os.path.exists(add_path):
        try:
            add_df = pd.read_csv(add_path)
            # normalize to ["lhs","rhs"] if possible
            if "lhs" in add_df.columns and "rhs" in add_df.columns:
                frames.append(add_df[["lhs", "rhs"]].copy())
            else:
                # best effort: take first two columns
                if add_df.shape[1] >= 2:
                    tmp = add_df.iloc[:, :2].copy()
                    tmp.columns = ["lhs", "rhs"]
                    frames.append(tmp)
        except Exception:
            pass
    # --- NEW end ---

    if fd_df is not None and len(fd_df) > 0:
        frames.append(fd_df[["lhs", "rhs"]].copy())
    if join_df is not None and len(join_df) > 0:
        frames.append(join_df[["lhs", "rhs"]].copy())

    # if nothing at all (no add, no generated)
    if not frames:
        pd.DataFrame(columns=["lhs", "rhs"]).to_csv(csv_path, index=False)
        print(f"Wrote EMPTY FD/JOIN CSV -> {csv_path}")
        return

    out = pd.concat(frames, ignore_index=True)

    # explode rhs lists if "a,b,c"
    out["rhs"] = out["rhs"].apply(lambda s: [x.strip() for x in str(s).split(",") if x.strip()])
    out = out.explode("rhs", ignore_index=True)

    out["lhs"] = out["lhs"].astype(str).str.strip().str.strip('"')
    out["rhs"] = out["rhs"].astype(str).str.strip().str.strip('"')
    out["lhs"] = out["lhs"].str.replace(r"\s*,\s*", ",", regex=True)

    out = out[(out["lhs"] != "") & (out["rhs"] != "")]
    out = out[(out["lhs"].str.lower() != "nan") & (out["rhs"].str.lower() != "nan")]

    out = out.drop_duplicates().reset_index(drop=True)

    # overwrite
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except Exception:
            pass
    out.to_csv(csv_path, index=False)
    print(f"Wrote FD/JOIN CSV (overwrite) -> {csv_path} (rows={len(out)})")

def write_domain_structured_overwrite(domain_df: pd.DataFrame, csv_path: str, global_alias_to_table, subq_lineage):
    """
    Writes structured conditional domain constraints (overwrite).
    Dedups within this run only.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    header = ["cid", "group_id", "lhs", "op", "value", "rhs", "domain_op", "domain_value"]

    if domain_df is None or len(domain_df) == 0:
        pd.DataFrame(columns=header).to_csv(csv_path, index=False)
        print(f"Wrote EMPTY structured domain CSV -> {csv_path}")
        return

    rows = []
    cid = 1

    for _, r in domain_df.iterrows():
        lhs = str(r["guard_col"]).strip()
        op = str(r["guard_op"]).strip()
        value = str(r["guard_val"]).strip()

        atoms = _emit_atomic_domain_rows(r["constraint_expr"], global_alias_to_table, subq_lineage)
        if not atoms:
            continue

        group_id = 0
        for rhs, domain_op, domain_value in atoms:
            rows.append(
                {
                    "cid": str(cid),
                    "group_id": str(group_id),
                    "lhs": lhs,
                    "op": op,
                    "value": value,
                    "rhs": str(rhs).strip(),
                    "domain_op": str(domain_op).strip(),
                    "domain_value": str(domain_value).strip(),
                }
            )
        cid += 1

    out = pd.DataFrame(rows, columns=header)

    for c in header:
        out[c] = out[c].astype(str).str.strip()

    out = out.drop_duplicates(
        subset=["group_id", "lhs", "op", "value", "rhs", "domain_op", "domain_value"],
        keep="first",
    ).reset_index(drop=True)

    # overwrite
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except Exception:
            pass
    out.to_csv(csv_path, index=False)
    print(f"Wrote structured domain CSV (overwrite) -> {csv_path} (rows={len(out)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="constraint_query")
    parser.add_argument("--base-dir", type=str, required=True, help="base directory")
    parser.add_argument("--query", type=str, required=True, help="query file")
    parser.add_argument("--fd-q", type=str, required=True, help="fd query csv")
    parser.add_argument("--domain-q", type=str, required=True, help="domain constraint query csv")
    parser.add_argument("--db", type=str, required=True, help="duckdb file (for PK introspection)")

    args = parser.parse_args()
    sql_path = os.path.join(args.base_dir, args.query)
    db_path  = os.path.join(args.base_dir, args.db)
    fd_q_path = os.path.join(args.base_dir, args.fd_q)
    domain_q_path = os.path.join(args.base_dir, args.domain_q)

    os.makedirs(os.path.dirname(fd_q_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(domain_q_path) or ".", exist_ok=True)

    with open(sql_path, "r") as f:
        SQL = f.read()

    pk_map = load_pk_map_from_duckdb(db_path)

    (result, is_selfjoin) = infer_join_and_fd_and_domain_dispatch(SQL, pk_map=pk_map)

    if not is_selfjoin:
        join_df, fd_df, domain_df, global_alias_to_table, subq_lineage = result
    else:
        join_df, fd_df, domain_df, global_alias_to_table, subq_lineage, alias_to_base = result
    
    print("is_selfjoin =", is_selfjoin)

    print("Join")
    if not is_selfjoin:
        print(join_df.to_string(index=False))
    else:
        # self-join은 role/base 같이 보기
        print(join_df.to_string(index=False))

    print("\nFD")
    # fd_df_print = fd_df.loc[fd_df["rhs"] != fd_df["rhs_base"]].reset_index(drop=True)
    # print(fd_df_print.to_string(index=False))
    fd_df_raw_print = add_alias_cols_for_print(fd_df)
    print(fd_df_raw_print.to_string(index=False))

    print("\nDomain (query-conditional)")
    if domain_df is None or len(domain_df) == 0:
        print("(none)")
    else:
        preview_rows = []
        tmp_cid = 1
        for _, r in domain_df.iterrows():
            atoms = _emit_atomic_domain_rows(r["constraint_expr"], global_alias_to_table, subq_lineage)
            for rhs, domain_op, domain_value in atoms:
                preview_rows.append(
                    {
                        "cid": tmp_cid,
                        "group_id": 0,
                        "lhs": r["guard_col"],
                        "op": r["guard_op"],
                        "value": r["guard_val"],
                        "rhs": rhs,
                        "domain_op": domain_op,
                        "domain_value": domain_value,
                    }
                )
            tmp_cid += 1
        print(pd.DataFrame(preview_rows).to_string(index=False))

    print("\n")
    # Join does not guarantee FDs
    # write_fd_csv_overwrite(fd_df, join_df, fd_q_path)
    fd_df = postprocess_selfjoin_fds(fd_df)
    # print(fd_df)
    fd_df_for_csv = fd_df
    if fd_df_for_csv is not None and "kind" in fd_df_for_csv.columns:
        fd_df_for_csv = fd_df_for_csv[fd_df_for_csv["kind"] != "projection"].copy()
        # removes unit->top_unit but not (order+60)->new_order (expression)
        # printed out on .out but not on csv file

    # print(fd_df_for_csv)
    write_fd_csv_overwrite(fd_df_for_csv, None, fd_q_path) # csv level

    write_domain_structured_overwrite(domain_df, domain_q_path, global_alias_to_table, subq_lineage)
