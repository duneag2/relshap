import os
import re
from typing import Optional, Dict, Tuple, List
import itertools
import argparse
import yaml
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split

def load_config(yaml_path: str) -> Dict:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def compile_id_name_regex(tokens: List[str]) -> re.Pattern:
    """
    suffix-only ID detection:
      id, user_id, movie_uuid, product_code
    """
    toks = [t.strip() for t in (tokens or []) if t.strip()]
    if not toks:
        toks = ["id", "uuid", "code"]

    alts = "|".join(re.escape(t) for t in toks)

    # suffix ONLY
    pattern = rf"(?:^|_)({alts})$"
    return re.compile(pattern, re.IGNORECASE)


def apply_config(cfg: Dict) -> None:
    global CONFIG
    global LABEL_COL, SAMPLE_N, SAMPLE_RANDOM_SEED
    global MAX_KEY_SIZE, MAX_LHS_SIZE, FD_EPS
    global CAT_MAX_UNIQUE, ID_UNIQUE_FRAC, MAX_RHS_DOMAIN, MAX_ALLOWED_SET, MIN_GROUP_SUPPORT
    global USE_QUANTILES_FOR_CONTI_CATE, Q_LO, Q_HI
    global ID_NAME_RE

    CONFIG = cfg

    LABEL_COL = set(cfg.get("LABEL_COL", []))
    SAMPLE_N = cfg.get("SAMPLE_N", None)
    SAMPLE_RANDOM_SEED = cfg.get("SAMPLE_RANDOM_SEED", 0)

    MAX_KEY_SIZE = cfg.get("MAX_KEY_SIZE", 1)
    MAX_LHS_SIZE = cfg.get("MAX_LHS_SIZE", 2)
    FD_EPS = cfg.get("FD_EPS", 0.01)

    CAT_MAX_UNIQUE = cfg.get("CAT_MAX_UNIQUE", 30)
    ID_UNIQUE_FRAC = cfg.get("ID_UNIQUE_FRAC", 0.2)
    MAX_RHS_DOMAIN = cfg.get("MAX_RHS_DOMAIN", 20)
    MAX_ALLOWED_SET = cfg.get("MAX_ALLOWED_SET", 20)
    MIN_GROUP_SUPPORT = cfg.get("MIN_GROUP_SUPPORT", 30)

    USE_QUANTILES_FOR_CONTI_CATE = cfg.get("USE_QUANTILES_FOR_CONTI_CATE", False)
    Q_LO = cfg.get("Q_LO", 0.005)
    Q_HI = cfg.get("Q_HI", 0.995)

    ID_NAME_RE = compile_id_name_regex(cfg.get("ID_NAME_TOKENS", ["id", "uuid", "code"]))


def _normalize_lhs_str(lhs) -> str:
    s = str(lhs).strip()
    s = s.strip("()")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return ",".join(sorted(parts))

def load_excluded_fd_pairs(fd_exclude_path: str) -> set:
    try:
        fd0 = pd.read_csv(fd_exclude_path)
    except Exception:
        return set()

    if fd0.empty or ("lhs" not in fd0.columns) or ("rhs" not in fd0.columns):
        return set()

    ex = set()
    for lhs, rhs in fd0[["lhs", "rhs"]].itertuples(index=False, name=None):
        lhs_s = _normalize_lhs_str(lhs)
        rhs_s = str(rhs).strip()
        if lhs_s and rhs_s:
            ex.add((lhs_s, rhs_s))
    return ex

def maybe_sample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed)

def _lhs_str(cols: Tuple[str, ...]) -> str:
    return ",".join(sorted(cols))

def _load_csv(csv_path: str, lut: str, config_path: str, seed) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if lut == 'full':
        pass
    elif lut == 'train':
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        label_col = list(config.get("LABEL_COL", []))
        if not label_col:
            raise ValueError("LABEL_COL is missing in config YAML.")

        drop_cols = list(config.get("DROP_COLS") or [])
        
        X = df.drop(columns=label_col+drop_cols, errors="raise")
        y = df[label_col].to_numpy().ravel()
        SEED = seed

        X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        
        df = X_train

    for c in LABEL_COL:
        if c in df.columns:
            df = df.drop(columns=[c])
    df = maybe_sample(df, SAMPLE_N, SAMPLE_RANDOM_SEED).reset_index(drop=True)

    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="ignore")

    return df

def _nunique(series: pd.Series) -> int:
    return int(series.nunique(dropna=False))

def _is_numeric_col(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_id_like(df: pd.DataFrame, col: str) -> bool:
    if ID_NAME_RE.search(col):
        return True
    n = len(df)
    if n == 0:
        return False
    d = _nunique(df[col])
    return (d / n) >= ID_UNIQUE_FRAC

def _is_categorical(df: pd.DataFrame, col: str) -> bool:
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
        return True
    if _is_numeric_col(s):
        if _is_id_like(df, col):
            return False
        return _nunique(s) <= CAT_MAX_UNIQUE
    return False

def _is_continuous(df: pd.DataFrame, col: str) -> bool:
    s = df[col]
    if not _is_numeric_col(s):
        return False
    if _is_id_like(df, col):
        return False
    return _nunique(s) > CAT_MAX_UNIQUE

def _fmt_val(v) -> str:
    if pd.isna(v):
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if isinstance(v, float) and float(v).is_integer():
            return str(int(v))
        return str(v)
    return str(v)

def _cond_eq(col: str, v) -> str:
    return f"{col} = {_fmt_val(v)}" if pd.notna(v) else f"{col} IS NULL"

def _domain_expr_allowed(col: str, allowed_list: List) -> str:
    has_null = any(pd.isna(x) for x in allowed_list)
    allowed_nonnull = [x for x in allowed_list if not pd.isna(x)]

    if has_null and not allowed_nonnull:
        return f"{col} IS NULL"
    if not has_null and allowed_nonnull:
        return f"{col} IN ({', '.join(_fmt_val(x) for x in allowed_nonnull)})"
    if has_null and allowed_nonnull:
        return f"({col} IN ({', '.join(_fmt_val(x) for x in allowed_nonnull)}) OR {col} IS NULL)"
    return f"{col} IN ({', '.join(_fmt_val(x) for x in allowed_list)})"

def load_excluded_fd_pairs_multi(paths: List[str], base_dir: str) -> set:
    ex = set()
    for p in (paths or []):
        p = str(p).strip()
        if not p:
            continue
        full = p if os.path.isabs(p) else os.path.join(base_dir, p)
        ex |= load_excluded_fd_pairs(full)
    return ex


# =========================
# (A) Unique keys (exact / near)
# =========================
def profile_unique_keys(df: pd.DataFrame, max_key_size: int, near_threshold: float = 0.99) -> pd.DataFrame:
    cols = list(df.columns)
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["lhs", "is_unique", "ratio_distinct", "distinct", "n"])

    rows = []
    for k in range(1, max_key_size + 1):
        for comb in itertools.combinations(cols, k):
            d = int(df.loc[:, comb].drop_duplicates().shape[0])
            ratio = d / n if n else 0.0
            rows.append(
                {
                    "lhs": _lhs_str(comb),
                    "is_unique": bool(d == n),
                    "ratio_distinct": float(ratio),
                    "distinct": d,
                    "n": n,
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["is_unique", "ratio_distinct", "lhs"], ascending=[False, False, True]).reset_index(drop=True)
    return out

def unique_keys_to_fd_all(df: pd.DataFrame, unique_keys_df: pd.DataFrame, keep: str = "unique", exclude_pairs: Optional[set] = None) -> pd.DataFrame:
    exclude_pairs = exclude_pairs or set()

    cols = list(df.columns)
    if unique_keys_df.empty:
        return pd.DataFrame(columns=["lhs", "rhs"])

    if keep == "unique":
        keys = unique_keys_df.loc[unique_keys_df["is_unique"] == True, "lhs"].tolist()
    else:
        keys = unique_keys_df.loc[unique_keys_df["ratio_distinct"] >= 0.99, "lhs"].tolist()

    rows = []
    for lhs in keys:
        lhs_cols = [c.strip() for c in lhs.split(",") if c.strip()]
        for rhs in cols:
                if rhs in lhs_cols:
                    continue
                if (_normalize_lhs_str(lhs), str(rhs).strip()) in exclude_pairs:
                    continue
                rows.append({"lhs": lhs, "rhs": rhs})

    out = pd.DataFrame(rows, columns=["lhs", "rhs"]).drop_duplicates().sort_values(["lhs", "rhs"]).reset_index(drop=True)
    return out


# =========================
# (B) FD / near-FD (group-violation rate)
# =========================
def profile_fd(df: pd.DataFrame, max_lhs_size: int, eps: float, exclude_pairs: Optional[set] = None) -> Dict[str, pd.DataFrame]:
    exact_min_lhs_sets = defaultdict(list)  # rhs -> [frozenset(lhs), ...]
    near_min_lhs_sets = defaultdict(list)

    def _has_subset(store: Dict[str, List[frozenset]], rhs: str, lhs_set: frozenset) -> bool:
        for s in store.get(rhs, []):
            if s.issubset(lhs_set):
                return True
        return False

    def _register_minimal(store: Dict[str, List[frozenset]], rhs: str, lhs_set: frozenset) -> None:
        # keep only minimal: remove supersets of lhs_set, keep others
        kept = []
        for s in store.get(rhs, []):
            if not s.issuperset(lhs_set):
                kept.append(s)
        kept.append(lhs_set)
        store[rhs] = kept
    
    cols = list(df.columns)
    n = len(df)
    exclude_pairs = exclude_pairs or set()

    if n == 0:
        return {
            "fd_exact": pd.DataFrame(columns=["lhs", "rhs"]),
            "fd_near": pd.DataFrame(
                columns=[
                    "lhs", "rhs", "violating_rows", "violating_ratio",
                    "violating_groups", "total_groups", "approx_error"
                ]
            ),
        }

    rows_exact = []
    rows_near = []


    for lhs_size in range(1, max_lhs_size + 1):
        exact_active_rhs = cols
        near_active_rhs = cols

        if not exact_active_rhs and not near_active_rhs:
            break


        for lhs in itertools.combinations(cols, lhs_size):
            lhs_list = list(lhs)
            lhs_s = _lhs_str(lhs)

            grp = df.groupby(lhs_list, dropna=False)
            grp_sizes = grp.size()
            total_groups = int(grp_sizes.shape[0])
            if total_groups == 0:
                continue

            lhs_set = frozenset(lhs_list)

            exact_rhs_candidates = []
            for rhs in exact_active_rhs:
                if rhs in lhs_list:
                    continue
                if (lhs_s, str(rhs).strip()) in exclude_pairs:
                    continue
                if _has_subset(exact_min_lhs_sets, rhs, lhs_set):
                    continue
                exact_rhs_candidates.append(rhs)

            near_rhs_candidates = []
            if eps is not None and eps >= 0:
                for rhs in near_active_rhs:
                    if rhs in lhs_list:
                        continue
                    if (lhs_s, str(rhs).strip()) in exclude_pairs:
                        continue
                    # NEW: if a smaller near-LHS already exists for this rhs, skip this superset LHS
                    if _has_subset(near_min_lhs_sets, rhs, lhs_set):
                        continue
                    near_rhs_candidates.append(rhs)


            rhs_union = list(dict.fromkeys(exact_rhs_candidates + near_rhs_candidates))
            if not rhs_union:
                continue

            nunique_sub = grp[rhs_union].nunique(dropna=False)

            # ---------- EXACT ----------
            for rhs in exact_rhs_candidates:
                violating_groups = int((nunique_sub[rhs] > 1).sum())
                if violating_groups == 0:
                    rows_exact.append({"lhs": lhs_s, "rhs": rhs})
                    _register_minimal(exact_min_lhs_sets, rhs, lhs_set)

            # ---------- NEAR ----------
            if eps is not None and eps >= 0:
                for rhs in near_rhs_candidates:
                    violating_mask = (nunique_sub[rhs] > 1)
                    violating_groups = int(violating_mask.sum())
                    if violating_groups == 0:
                        continue
                    approx_error = violating_groups / total_groups
                    if approx_error <= eps:
                        violating_rows = int(grp_sizes[violating_mask].sum()) if len(grp_sizes) else 0
                        violating_ratio = (violating_rows / n) if n else 0.0
                        rows_near.append(
                            {
                                "lhs": lhs_s,
                                "rhs": rhs,
                                "violating_rows": int(violating_rows),
                                "violating_ratio": float(violating_ratio),
                                "violating_groups": int(violating_groups),
                                "total_groups": int(total_groups),
                                "approx_error": float(approx_error),
                            }
                        )
                        _register_minimal(near_min_lhs_sets, rhs, lhs_set)


    df_exact = (
        pd.DataFrame(rows_exact, columns=["lhs", "rhs"])
        .drop_duplicates()
        .sort_values(["rhs", "lhs"])
        .reset_index(drop=True)
    )

    df_near = (
        pd.DataFrame(
            rows_near,
            columns=[
                "lhs", "rhs", "violating_rows", "violating_ratio",
                "violating_groups", "total_groups", "approx_error"
            ],
        )
        .drop_duplicates()
        .sort_values(["rhs", "approx_error", "violating_ratio", "lhs"], ascending=[True, True, True, True])
        .reset_index(drop=True)
    )

    return {"fd_exact": df_exact, "fd_near": df_near}


# =========================
# (C) Domain constraints generation from FD (inverse)
#   - cate-cate: (B=v) => A IN allowed(A|B=v)
#   - conti-cate: (B=v) => lo <= A <= hi
#   - cate-conti: (B_bin=b) => A IN allowed(A|B_bin=b)
# =========================
def _fd_pairs_from_dfs(fd_exact: pd.DataFrame, fd_near: pd.DataFrame) -> List[Tuple[str, str]]:
    pairs = []
    if fd_exact is not None and not fd_exact.empty:
        pairs += list(fd_exact[["lhs", "rhs"]].itertuples(index=False, name=None))

    seen = set()
    out = []
    for a, b in pairs:
        key = (str(a), str(b))
        if key not in seen:
            seen.add(key)
            out.append((str(a), str(b)))
    return out

def build_domain_constraints(df: pd.DataFrame, fd_exact: pd.DataFrame, fd_near: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["lhs", "rhs", "domain"])

    cat_cols = {c for c in df.columns if _is_categorical(df, c)}
    cont_cols = {c for c in df.columns if _is_continuous(df, c)}
    id_like = {c for c in df.columns if _is_id_like(df, c)}

    fd_pairs = _fd_pairs_from_dfs(fd_exact, fd_near)

    rows = []

    for A, B in fd_pairs:
        if A not in df.columns or B not in df.columns:
            continue

        if A in id_like:
            continue

        A_is_cat = A in cat_cols
        A_is_cont = A in cont_cols
        B_is_cat = B in cat_cols
        B_is_cont = B in cont_cols

        if B_is_cat:
            B_dom = df[B].drop_duplicates()
            B_dom_size = int(B_dom.shape[0])
            if B_dom_size > MAX_RHS_DOMAIN:
                continue

            vc = df[B].value_counts(dropna=False)

            if A_is_cat:
                for bval in B_dom.tolist():
                    support = int(vc.get(bval, 0))
                    if support < MIN_GROUP_SUPPORT:
                        continue
                    sub = df.loc[df[B].eq(bval) if pd.notna(bval) else df[B].isna(), A]
                    allowed = sub.drop_duplicates().tolist()
                    allowed_size = len(allowed)
                    if allowed_size == 0:
                        continue
                    if allowed_size > MAX_ALLOWED_SET:
                        continue

                    b_str = _cond_eq(B, bval)
                    a_dom = _domain_expr_allowed(A, allowed)
                    domain = f"({b_str}) => ({a_dom})"

                    rows.append({"lhs": A, "rhs": B, "domain": domain})

            elif A_is_cont:
                for bval in B_dom.tolist():
                    support = int(vc.get(bval, 0))
                    if support < MIN_GROUP_SUPPORT:
                        continue
                    sub = df.loc[df[B].eq(bval) if pd.notna(bval) else df[B].isna(), A]
                    sub = pd.to_numeric(sub, errors="coerce").dropna()
                    if sub.empty:
                        continue

                    if USE_QUANTILES_FOR_CONTI_CATE:
                        lo = float(sub.quantile(Q_LO, interpolation="linear"))
                        hi = float(sub.quantile(Q_HI, interpolation="linear"))
                    else:
                        lo = float(sub.min())
                        hi = float(sub.max())

                    b_str = _cond_eq(B, bval)
                    lo_s = str(int(lo)) if float(lo).is_integer() else str(lo)
                    hi_s = str(int(hi)) if float(hi).is_integer() else str(hi)

                    domain = f"({b_str}) => ({A} >= {lo_s} AND {A} <= {hi_s})"
                    rows.append({"lhs": A, "rhs": B, "domain": domain})

            else:
                continue

        elif B_is_cont and A_is_cat:
            bnum = pd.to_numeric(df[B], errors="coerce")
            if bnum.dropna().empty:
                continue

            q = max(2, min(MAX_RHS_DOMAIN, 10))
            try:
                bins = pd.qcut(bnum, q=q, duplicates="drop")
            except Exception:
                continue

            Bbin = f"{B}__bin"
            tmp = df[[A]].copy()
            tmp[Bbin] = bins.astype(str)

            B_dom = tmp[Bbin].drop_duplicates()
            B_dom_size = int(B_dom.shape[0])
            if B_dom_size > MAX_RHS_DOMAIN:
                continue

            vc = tmp[Bbin].value_counts(dropna=False)

            for bval in B_dom.tolist():
                support = int(vc.get(bval, 0))
                if support < MIN_GROUP_SUPPORT:
                    continue

                sub = tmp.loc[tmp[Bbin].eq(bval) if pd.notna(bval) else tmp[Bbin].isna(), A]
                allowed = sub.drop_duplicates().tolist()
                allowed_size = len(allowed)
                if allowed_size == 0:
                    continue
                if allowed_size > MAX_ALLOWED_SET:
                    continue

                b_str = _cond_eq(Bbin, str(bval))
                a_dom = _domain_expr_allowed(A, allowed)
                domain = f"({b_str}) => ({a_dom})"

                rows.append({"lhs": A, "rhs": B, "domain": domain})

        else:
            continue

    out = pd.DataFrame(rows, columns=["lhs", "rhs", "domain"])
    out = out.drop_duplicates().sort_values(["rhs", "lhs"], ascending=[True, True]).reset_index(drop=True)
    return out

# =========================
# (D) Integrity constraints from cyclic FDs (one-hot / parity), emitted as both
#     - integrity expression (readable)
#     - denial form (NOT(violation))
# =========================

def _parse_lhs(lhs_s: str) -> Tuple[str, ...]:
    lhs_s = str(lhs_s).strip()
    if lhs_s == "":
        return tuple()
    return tuple(c.strip() for c in lhs_s.split(",") if c.strip())

def _is_integer_series(s: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(s):
        return True
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        x = x.dropna()
        if x.empty:
            return False
        return bool((x % 1 == 0).all())
    return False

def _as_int_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.astype("Int64")

def _detect_full_cycles(fd_exact: pd.DataFrame) -> List[Tuple[str, ...]]:
    """
    Find sets S such that for every c in S, (S\{c}) -> c exists in fd_exact.
    Returns list of tuples of columns (sorted).
    """
    if fd_exact is None or fd_exact.empty:
        return []

    edges = set()
    for lhs_s, rhs in fd_exact[["lhs", "rhs"]].itertuples(index=False, name=None):
        lhs = _parse_lhs(lhs_s)
        rhs = str(rhs).strip()
        edges.add((frozenset(lhs), rhs))

    candidates: Dict[frozenset, set] = {}

    # Build candidate set S = lhs ∪ {rhs} for each FD
    for lhs_set, rhs in edges:
        S = frozenset(set(lhs_set) | {rhs})
        if len(S) < 2:
            continue
        candidates.setdefault(S, set()).add(rhs)

    out = []
    for S in candidates.keys():
        ok = True
        for c in S:
            lhs_needed = frozenset(set(S) - {c})
            if (lhs_needed, c) not in edges:
                ok = False
                break
        if ok:
            out.append(tuple(sorted(S)))
    out.sort(key=lambda t: (len(t), t))
    return out

def _infer_onehot_k(X: pd.DataFrame) -> Optional[int]:
    """
    X: integer-only dataframe of shape (n, m).
    Returns k if rows look like scaled one-hot: values in {0,k}, at most one nonzero per row,
    and nonzero values are constant k. Otherwise None.
    """
    if X.empty:
        return None

    vals = X.to_numpy()
    # Treat <NA> as missing: if any NA exists, reject for integrity inference
    if pd.isna(vals).any():
        return None

    # compute nonzero counts per row
    nz = (vals != 0)
    nz_cnt = nz.sum(axis=1)

    # must be <= 1 nonzero per row (allow all-zero rows)
    if (nz_cnt > 1).any():
        return None

    nonzero_vals = vals[nz]
    if nonzero_vals.size == 0:
        return None

    # all nonzero must be same positive integer k (allow negative? usually no; enforce >0)
    uniq = pd.unique(nonzero_vals.ravel())
    if len(uniq) != 1:
        return None
    k = int(uniq[0])
    if k <= 0:
        return None

    # all entries must be 0 or k
    if not ((vals == 0) | (vals == k)).all():
        return None

    return k

def _infer_parity_bit(X: pd.DataFrame) -> Optional[int]:
    """
    X integer-only. Detect parity constraint for binary 0/1 columns:
      sum(cols) % 2 == b for b in {0,1}
    Returns b if holds exactly, else None.
    """
    if X.empty:
        return None
    vals = X.to_numpy()
    if pd.isna(vals).any():
        return None

    # must be binary 0/1 only
    if not ((vals == 0) | (vals == 1)).all():
        return None

    s = vals.sum(axis=1) % 2
    if (s == 0).all():
        return 0
    if (s == 1).all():
        return 1
    return None

def _cols_domain_clause(cols: List[str], k: int) -> str:
    # (c IN (0,k)) AND ... for all cols
    return " AND ".join([f"({c} IN (0,{k}))" for c in cols])

def _pairwise_zero_product_clause(cols: List[str]) -> str:
    parts = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            parts.append(f"({cols[i]} * {cols[j]} = 0)")
    if not parts:
        return "TRUE"
    return " AND ".join(parts)

def build_integrity_constraints(df: pd.DataFrame, fd_exact: pd.DataFrame) -> pd.DataFrame:
    """
    Output columns:
      - cols: comma-joined columns involved
      - kind: onehot_2, onehot_m, parity
      - k / parity_bit
      - integrity: human-readable constraint
      - denial: NOT(violation) form
    """
    if df is None or df.empty or fd_exact is None or fd_exact.empty:
        return pd.DataFrame(columns=["cols", "kind", "k", "parity_bit", "integrity", "denial"])

    cycles = _detect_full_cycles(fd_exact)
    rows = []

    for S in cycles:
        cols = list(S)

        # integer-only restriction
        if not all(c in df.columns for c in cols):
            continue
        if not all(_is_integer_series(df[c]) for c in cols):
            continue

        X = pd.DataFrame({c: _as_int_series(df[c]) for c in cols})
        if X.isna().any().any():
            continue

        m = len(cols)

        # 1) one-hot (scaled k)
        k = _infer_onehot_k(X)
        if k is not None:
            sum_expr = " + ".join(cols)
            domain_clause = _cols_domain_clause(cols, k)
            prod_clause = _pairwise_zero_product_clause(cols)

            # decide whether all-zero appears; if yes, allow sum in {0,k}, else force sum==k
            sums = X.to_numpy().sum(axis=1)
            allow_zero = bool((sums == 0).any())

            if allow_zero:
                integrity = f"({domain_clause}) AND ({prod_clause}) AND (({sum_expr}) IN (0,{k}))"
                violation = f"NOT(({domain_clause}) AND ({prod_clause}) AND (({sum_expr}) IN (0,{k})))"
            else:
                integrity = f"({domain_clause}) AND ({prod_clause}) AND (({sum_expr}) = {k})"
                violation = f"NOT(({domain_clause}) AND ({prod_clause}) AND (({sum_expr}) = {k}))"

            kind = "onehot_2" if m == 2 else "onehot_m"

            rows.append(
                {
                    "cols": ",".join(cols),
                    "kind": kind,
                    "k": int(k),
                    "parity_bit": None,
                    "integrity": integrity
                }
            )
            continue

        # 2) parity (xor)
        b = _infer_parity_bit(X)
        if b is not None:
            sum_expr = " + ".join(cols)
            integrity = f"(({sum_expr}) % 2) = {b}"
            violation = f"(({sum_expr}) % 2) != {b}"
            rows.append(
                {
                    "cols": ",".join(cols),
                    "kind": "parity",
                    "k": None,
                    "parity_bit": int(b),
                    "integrity": integrity,
                }
            )
            continue

    out = pd.DataFrame(rows, columns=["cols", "kind", "k", "parity_bit", "integrity"])
    out = out.drop_duplicates().sort_values(["kind", "cols"]).reset_index(drop=True)
    return out


# =========================
# Main
# =========================
def run_all(csv_path: str, lut: str, config_path: str, seed, exclude_pairs: Optional[set] = None,) -> Dict[str, pd.DataFrame]:
    df = _load_csv(csv_path, lut, config_path, seed)
    # print(df)
    exclude_pairs = exclude_pairs or set()

    unique_keys = profile_unique_keys(df, max_key_size=MAX_KEY_SIZE, near_threshold=0.99)
    unique_fd_all = unique_keys_to_fd_all(df, unique_keys, keep="unique", exclude_pairs=exclude_pairs)

    fd_out = profile_fd(df, max_lhs_size=MAX_LHS_SIZE, eps=FD_EPS, exclude_pairs=exclude_pairs)
    fd_exact = fd_out["fd_exact"]
    fd_near = fd_out["fd_near"]

    if not unique_fd_all.empty and not fd_exact.empty:
        fd_exact_pairs = set(map(tuple, fd_exact[["lhs", "rhs"]].itertuples(index=False, name=None)))
        keep_mask = ~unique_fd_all.apply(lambda r: (r["lhs"], r["rhs"]) in fd_exact_pairs, axis=1)
        unique_fd_all = unique_fd_all.loc[keep_mask].reset_index(drop=True)

    domain_df = build_domain_constraints(df, fd_exact=fd_exact, fd_near=fd_near)

    integrity_df = build_integrity_constraints(df, fd_exact=fd_exact)


    return {
        "fd_exact": fd_exact,
        "fd_near": fd_near,
        "unique_fd_all": unique_fd_all,
        "domain": domain_df,
        "integrity": integrity_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="constraint_data")

    parser.add_argument("--base-dir", type=str, required=True, help="base directory")
    parser.add_argument("--flattened", type=str, required=True, help="flattened csv file")
    parser.add_argument("--config", type=str, required=True, help="yaml config file")
    parser.add_argument("--fd-exclude", type=str, nargs="*", default=[], help="one or more fd exclude csv paths")
    parser.add_argument("--fd-d", type=str, required=True, help="fd data csv")
    parser.add_argument("--approx-fd", type=str, required=True, help="approximate fd csv")
    parser.add_argument("--domain-constraint-d", required=True, help="domain constraint csv")
    parser.add_argument("--denial-constraint-d", required=True, help="denial constraint csv")
    parser.add_argument("--lut", required=True, help="lookup table")
    parser.add_argument("--seed", required=True, help="seed")

    args = parser.parse_args()
    csv_path = os.path.join(args.base_dir, args.flattened)
    config_path = os.path.join(args.base_dir, args.config)
    fd_d_path = os.path.join(args.base_dir, args.fd_d)
    approx_fd_path = os.path.join(args.base_dir, args.approx_fd)
    domain_constraint_path = os.path.join(args.base_dir, args.domain_constraint_d)
    denial_constraint_path = os.path.join(args.base_dir, args.denial_constraint_d)

    cfg = load_config(config_path)
    apply_config(cfg)
    
    excluded = load_excluded_fd_pairs_multi(args.fd_exclude, args.base_dir)
    results = run_all(csv_path, str(args.lut), config_path, int(args.seed), exclude_pairs=excluded,)

    print("\n")
    print("FD")
    if results["fd_exact"].empty:
        print("(empty)")
    else:
        print(results["fd_exact"].to_string(index=False))

    print("\n")
    print("Approximate FD")
    if results["fd_near"].empty:
        print("(empty)")
    else:
        print(results["fd_near"].to_string(index=False))

    print("\n")
    print("UNIQUE (filtered only if in exact FD)")
    if results["unique_fd_all"].empty:
        print("(empty)")
    else:
        print(results["unique_fd_all"].to_string(index=False))

    print("\n")
    print("Domain Constraint")
    if results["domain"].empty:
        print("(empty)")
    else:
        print(results["domain"].to_string(index=False))

    print("\n")
    print("Denial Constraint from cyclic FDs")
    if results["integrity"].empty:
        print("(empty)")
    else:
        print(results["integrity"].to_string(index=False))

    # -------------------------
    # 1) fd_d_path: exact FD + UNIQUE  -> columns: [lhs, rhs]
    # -------------------------
    fd_parts = []

    if not results["fd_exact"].empty:
        fd_parts.append(results["fd_exact"][["lhs", "rhs"]].copy())

    if not results["unique_fd_all"].empty:
        fd_parts.append(results["unique_fd_all"][["lhs", "rhs"]].copy())

    if fd_parts:
        fd_d_df = pd.concat(fd_parts, ignore_index=True).drop_duplicates(subset=["lhs", "rhs"]).reset_index(drop=True)
    else:
        fd_d_df = pd.DataFrame(columns=["lhs", "rhs"])

    os.makedirs(os.path.dirname(fd_d_path) or ".", exist_ok=True)
    fd_d_df.to_csv(fd_d_path, index=False)


    # -------------------------
    # 2) approx_fd_path: approximate FD -> columns: [lhs, rhs, approx_error]
    # -------------------------
    if not results["fd_near"].empty:
        approx_df = results["fd_near"].copy()
        approx_df = approx_df[["lhs", "rhs", "approx_error"]].drop_duplicates(subset=["lhs", "rhs", "approx_error"]).reset_index(drop=True)
    else:
        approx_df = pd.DataFrame(columns=["lhs", "rhs", "approx_error"])

    os.makedirs(os.path.dirname(approx_fd_path) or ".", exist_ok=True)
    approx_df.to_csv(approx_fd_path, index=False)

    # -------------------------
    # 3) Domain Constraints & Denial Constraints
    #   - domain_constraint_path: DNF
    #   - denial_constraint_path: compact generator rules for onehot/parity(xor)
    # -------------------------

    def _strip_outer_parens(s: str) -> str:
        s = str(s).strip()
        if s.startswith("(") and s.endswith(")"):
            return s[1:-1].strip()
        return s

    _RE_EQ = re.compile(r"""^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$""")
    _RE_ISNULL = re.compile(r"""^\s*([A-Za-z_]\w*)\s+IS\s+NULL\s*$""", re.IGNORECASE)
    _RE_IN = re.compile(r"""^\s*([A-Za-z_]\w*)\s+IN\s*\((.*)\)\s*$""", re.IGNORECASE)
    _RE_RANGE = re.compile(
        r"""^\s*([A-Za-z_]\w*)\s*>=\s*([+-]?\d+(?:\.\d+)?)\s+AND\s+\1\s*<=\s*([+-]?\d+(?:\.\d+)?)\s*$""",
        re.IGNORECASE,
    )

    def _parse_sql_list_items(inner: str) -> List:
        """
        Parse items inside IN(...).
        Supports:
          - 'str' (with '' escaping)
          - numbers
          - NULL
        """
        inner = inner.strip()
        if inner == "":
            return []

        items = []
        buf = []
        in_quote = False
        i = 0
        while i < len(inner):
            ch = inner[i]
            if ch == "'":
                if in_quote and i + 1 < len(inner) and inner[i + 1] == "'":
                    # escaped quote inside string
                    buf.append("'")
                    i += 2
                    continue
                in_quote = not in_quote
                # do not keep outer quotes
                i += 1
                continue

            if ch == "," and not in_quote:
                token = "".join(buf).strip()
                buf = []
                if token != "":
                    items.append(token)
                i += 1
                continue

            buf.append(ch)
            i += 1

        token = "".join(buf).strip()
        if token != "":
            items.append(token)

        out = []
        for t in items:
            t0 = t.strip()
            if t0.upper() == "NULL":
                out.append(None)
                continue
            # try numeric
            try:
                if "." in t0:
                    out.append(float(t0))
                else:
                    out.append(int(t0))
                continue
            except Exception:
                pass
            # otherwise string literal already unquoted by parser
            out.append(t0)
        return out

    def _parse_domain_rule_to_dnf_atoms(domain_str: str) -> List[Dict]:
        """
        Returns list of atoms dicts with fields:
          lhs, op, value, rhs, domain_op, domain_value, group_id (local, starting at 0)
        Domain strings are produced by build_domain_constraints() and have limited grammar:
          (B = v) => (A IN (...))
          (B = v) => ((A IN (...) OR A IS NULL))
          (B = v) => (A >= lo AND A <= hi)
          (B IS NULL) => (...)
        """
        s = str(domain_str).strip()
        if "=>" not in s:
            return []

        left_s, right_s = s.split("=>", 1)
        left_s = _strip_outer_parens(left_s.strip())
        right_s = _strip_outer_parens(right_s.strip())

        # ---- parse left: B = v OR B IS NULL
        mnull = _RE_ISNULL.match(left_s)
        if mnull:
            guard_col = mnull.group(1).strip()
            guard_op = "IS_NULL"
            guard_val = None
        else:
            meq = _RE_EQ.match(left_s)
            if not meq:
                return []
            guard_col = meq.group(1).strip()
            raw_v = meq.group(2).strip()
            raw_v = raw_v.strip()
            if raw_v.upper() == "NULL":
                guard_op = "IS_NULL"
                guard_val = None
            else:
                # strip quotes if present
                if raw_v.startswith("'") and raw_v.endswith("'"):
                    guard_val = raw_v[1:-1].replace("''", "'")
                else:
                    # numeric?
                    try:
                        guard_val = int(raw_v) if "." not in raw_v else float(raw_v)
                    except Exception:
                        guard_val = raw_v
                guard_op = "="

        # ---- parse right:
        # 1) (A IN (...) OR A IS NULL)
        # right might still have outer parens removed, but may be "(A IN (...) OR A IS NULL)"
        rs = right_s.strip()
        rs = _strip_outer_parens(rs)

        # detect explicit OR-NULL form
        # expected exact pattern: "(A IN (...) OR A IS NULL)" or "A IN (...) OR A IS NULL"
        # We'll split on OR if present (top-level only; our generator only makes this one OR)
        if re.search(r"\s+OR\s+", rs, flags=re.IGNORECASE):
            parts = re.split(r"\s+OR\s+", rs, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) == 2:
                p0, p1 = parts
                p0 = _strip_outer_parens(p0)
                p1 = _strip_outer_parens(p1)

                m_in = _RE_IN.match(p0)
                m_isnull = _RE_ISNULL.match(p1) or _RE_ISNULL.match(p0)
                # handle either order just in case
                if m_in and (_RE_ISNULL.match(p1) is not None):
                    A = m_in.group(1).strip()
                    items = _parse_sql_list_items(m_in.group(2))
                    items_nonnull = [x for x in items if x is not None]

                    atoms = []
                    # group 0: IN(nonnull)
                    atoms.append(
                        {
                            "group_id": 0,
                            "lhs": guard_col,
                            "op": guard_op,
                            "value": guard_val,
                            "rhs": A,
                            "domain_op": "IN",
                            "domain_value": items_nonnull,
                        }
                    )
                    # group 1: IS_NULL
                    atoms.append(
                        {
                            "group_id": 1,
                            "lhs": guard_col,
                            "op": guard_op,
                            "value": guard_val,
                            "rhs": A,
                            "domain_op": "IS_NULL",
                            "domain_value": None,
                        }
                    )
                    return atoms

        # 2) range: A >= lo AND A <= hi
        m_range = _RE_RANGE.match(rs)
        if m_range:
            A = m_range.group(1).strip()
            lo = float(m_range.group(2))
            hi = float(m_range.group(3))
            # keep ints if integral
            lo_v = int(lo) if lo.is_integer() else lo
            hi_v = int(hi) if hi.is_integer() else hi

            return [
                {
                    "group_id": 0,
                    "lhs": guard_col,
                    "op": guard_op,
                    "value": guard_val,
                    "rhs": A,
                    "domain_op": ">=",
                    "domain_value": lo_v,
                },
                {
                    "group_id": 0,
                    "lhs": guard_col,
                    "op": guard_op,
                    "value": guard_val,
                    "rhs": A,
                    "domain_op": "<=",
                    "domain_value": hi_v,
                },
            ]

        # 3) plain IN(...)
        m_in = _RE_IN.match(rs)
        if m_in:
            A = m_in.group(1).strip()
            items = _parse_sql_list_items(m_in.group(2))
            return [
                {
                    "group_id": 0,
                    "lhs": guard_col,
                    "op": guard_op,
                    "value": guard_val,
                    "rhs": A,
                    "domain_op": "IN",
                    "domain_value": items,
                }
            ]

        # 4) plain IS NULL
        m_isnull = _RE_ISNULL.match(rs)
        if m_isnull:
            A = m_isnull.group(1).strip()
            return [
                {
                    "group_id": 0,
                    "lhs": guard_col,
                    "op": guard_op,
                    "value": guard_val,
                    "rhs": A,
                    "domain_op": "IS_NULL",
                    "domain_value": None,
                }
            ]

        return []

    # ---------- (3a) domain_constraint_path: DNF atoms ----------
    domain_atoms_rows = []
    if results["domain"] is not None and not results["domain"].empty:
        cid = 0
        for dom in results["domain"]["domain"].tolist():
            atoms = _parse_domain_rule_to_dnf_atoms(dom)
            if not atoms:
                continue
            cid += 1
            for a in atoms:
                domain_atoms_rows.append(
                    {
                        "cid": cid,
                        "group_id": int(a["group_id"]),
                        "lhs": a["lhs"],
                        "op": a["op"],
                        "value": a["value"],
                        "rhs": a["rhs"],
                        "domain_op": a["domain_op"],
                        # store list/numbers/null in a machine-readable way
                        "domain_value": yaml.safe_dump(a["domain_value"], default_flow_style=True).strip()
                        if isinstance(a["domain_value"], (list, dict))
                        else a["domain_value"],
                    }
                )

    domain_atoms_df = pd.DataFrame(
        domain_atoms_rows,
        columns=["cid", "group_id", "lhs", "op", "value", "rhs", "domain_op", "domain_value"],
    )

    os.makedirs(os.path.dirname(domain_constraint_path) or ".", exist_ok=True)
    domain_atoms_df.to_csv(domain_constraint_path, index=False)

    # ---------- (3b) denial_constraint_path: minimal (kind + k/parity_bit only) ----------
    denial_rows = []

    if results["integrity"] is not None and not results["integrity"].empty:
        ic_id = 0
        for cols_s, kind, k, parity_bit in results["integrity"][["cols", "kind", "k", "parity_bit"]].itertuples(index=False, name=None):
            

            kind0 = str(kind).strip().lower()
            if kind0.startswith("onehot"):
                kind_norm = "onehot"
            elif kind0 == "parity":
                try:
                    if pd.notna(parity_bit) and int(parity_bit) == 0:
                        continue
                except Exception:
                    pass
                kind_norm = "xor"
            else:
                continue

            ic_id += 1

            denial_rows.append(
                {
                    "ic_id": ic_id,
                    "kind": kind_norm,
                    "cols": str(cols_s).strip(),
                    "k": (int(k) if k is not None and str(k) != "nan" else None),
                    "parity_bit": (int(parity_bit) if parity_bit is not None and str(parity_bit) != "nan" else None),
                }
            )

    denial_df = pd.DataFrame(denial_rows, columns=["ic_id", "kind", "cols", "k", "parity_bit"])
    os.makedirs(os.path.dirname(denial_constraint_path) or ".", exist_ok=True)
    denial_df.to_csv(denial_constraint_path, index=False)