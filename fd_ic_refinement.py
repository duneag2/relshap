import os
import argparse
import pickle
from typing import FrozenSet, Iterable, List, Tuple, Dict, Set, Optional, Any
from collections import defaultdict, deque
import yaml

import pandas as pd

# =========================
# Helpers: parsing / formatting / normalization
# =========================

def load_config(yaml_path: str) -> Dict:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _clean_token(s: str) -> str:
    return str(s).strip().strip('"').strip("'")


def normalize_lhs(lhs) -> str:
    if pd.isna(lhs):
        return ""
    s = str(lhs).strip()

    for left, right in [("(", ")"), ("[", "]"), ("{", "}")]:
        if s.startswith(left) and s.endswith(right):
            s = s[1:-1].strip()

    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    parts = [_clean_token(p) for p in parts]
    return ",".join(parts)


def normalize_rhs(rhs) -> str:
    if pd.isna(rhs):
        return ""
    return _clean_token(rhs)


def parse_attr_set(x) -> FrozenSet[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return frozenset()
    s = str(x).strip()
    if not s:
        return frozenset()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return frozenset(parts)


def format_attr_set(xs: Iterable[str]) -> str:
    return ",".join(sorted(set(xs)))


def read_fd_csv(path: Optional[str], expect_cols=("lhs", "rhs")) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=list(expect_cols))

    df = pd.read_csv(path)
    missing = [c for c in expect_cols if c not in df.columns]
    if missing:
        raise ValueError(f"File {path} missing required columns: {missing}. Found: {list(df.columns)}")

    out = df.loc[:, list(expect_cols)].copy()
    out["lhs"] = out["lhs"].apply(normalize_lhs)
    out["rhs"] = out["rhs"].apply(normalize_rhs)
    out = out[(out["lhs"] != "") & (out["rhs"] != "")]
    return out


def read_csv_if_exists(path: Optional[str]) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_common_constraint_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if "lhs" in out.columns:
        out["lhs"] = out["lhs"].apply(normalize_lhs)
    if "rhs" in out.columns:
        out["rhs"] = out["rhs"].apply(normalize_rhs)

    for c in ["attrs", "attributes", "columns", "cols"]:
        if c in out.columns:
            out[c] = out[c].apply(normalize_lhs)

    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda x: _clean_token(x) if pd.notna(x) else x)

    return out


def _parse_bool(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y")


# =========================
# FD closure engine + cache
# =========================

def ensure_single_rhs(fd_df: pd.DataFrame, rhs_col: str = "rhs") -> None:
    if rhs_col not in fd_df.columns:
        raise ValueError(f"fd_df missing required column '{rhs_col}'")

    bad = []
    for i, v in enumerate(fd_df[rhs_col].tolist()):
        rhs_set = parse_attr_set(v)
        if len(rhs_set) != 1:
            bad.append((i, v))
            if len(bad) >= 10:
                break

    if bad:
        examples = "; ".join([f"row={i} rhs={repr(v)}" for i, v in bad])
        raise ValueError(
            "Expected rhs to contain exactly 1 attribute per row, but found violations. "
            f"Examples: {examples}"
        )


FD = Tuple[FrozenSet[str], str]


def build_fds(fd_df: pd.DataFrame, lhs_col: str = "lhs", rhs_col: str = "rhs") -> List[FD]:
    ensure_single_rhs(fd_df, rhs_col=rhs_col)
    fds: List[FD] = []
    for _, row in fd_df.iterrows():
        lhs_set = parse_attr_set(row[lhs_col])
        rhs_set = parse_attr_set(row[rhs_col])
        rhs_attr = next(iter(rhs_set))
        fds.append((lhs_set, rhs_attr))
    return fds


class FDClosureEngine:
    def __init__(self, fds: List[FD]):
        self.fds = fds
        self.m = len(fds)

        self.rhs: List[str] = []
        self.lhs_size: List[int] = []
        self.occurs_in_lhs: Dict[str, List[int]] = defaultdict(list)

        for i, (lhs_set, rhs_attr) in enumerate(fds):
            lhs_list = list(lhs_set)
            self.rhs.append(rhs_attr)
            self.lhs_size.append(len(lhs_list))
            for a in lhs_list:
                self.occurs_in_lhs[a].append(i)

    def closure(self, seed: FrozenSet[str]) -> FrozenSet[str]:
        known: Set[str] = set(seed)
        need = self.lhs_size.copy()

        for a in known:
            for i in self.occurs_in_lhs.get(a, []):
                need[i] -= 1

        ready = deque([i for i in range(self.m) if need[i] == 0])

        while ready:
            i = ready.popleft()
            r = self.rhs[i]
            if r in known:
                continue
            known.add(r)

            for j in self.occurs_in_lhs.get(r, []):
                need[j] -= 1
                if need[j] == 0:
                    ready.append(j)

        return frozenset(known)


def _set_key(s: FrozenSet[str]) -> Tuple[str, ...]:
    return tuple(sorted(s))


class ClosureIndex:
    def __init__(self, engine: FDClosureEngine):
        self.engine = engine
        self.cache: Dict[Tuple[str, ...], Tuple[str, ...]] = {}

    def get_closure(self, S: FrozenSet[str]) -> FrozenSet[str]:
        k = _set_key(S)
        if k in self.cache:
            return frozenset(self.cache[k])
        S_plus = self.engine.closure(S)
        self.cache[k] = _set_key(S_plus)
        return S_plus


def warm_cache_singletons_and_lhs(fd_df: pd.DataFrame, index: ClosureIndex) -> None:
    attrs: Set[str] = set()
    lhs_sets: List[FrozenSet[str]] = []

    for _, r in fd_df.iterrows():
        lhs = parse_attr_set(r.get("lhs"))
        rhs = parse_attr_set(r.get("rhs"))
        attrs |= set(lhs)
        attrs |= set(rhs)
        if lhs:
            lhs_sets.append(lhs)

    for a in sorted(attrs):
        index.get_closure(frozenset([a]))

    for S in sorted(set(lhs_sets), key=lambda x: (len(x), format_attr_set(x))):
        index.get_closure(S)

def drop_fds_by_config(
    fd_df: pd.DataFrame,
    drop_cols: Set[str],
    mode_provenance: bool,
) -> pd.DataFrame:
    """
    - mode_provenance == True  → drop 안 함
    - mode_provenance == False → DROP_COLS에 걸리는 FD 제거 (lhs OR rhs)
    """
    if mode_provenance or fd_df.empty or not drop_cols:
        return fd_df

    def bad_fd(lhs: str, rhs: str) -> bool:
        lhs_set = parse_attr_set(lhs)
        rhs_set = parse_attr_set(rhs)  # should be singleton (enforced later)
        # lhs에 dropcol 하나라도 있거나, rhs가 dropcol이면 제거
        return (not lhs_set.isdisjoint(drop_cols)) or (not rhs_set.isdisjoint(drop_cols))

    mask_keep = ~fd_df.apply(lambda r: bad_fd(r["lhs"], r["rhs"]), axis=1)
    return fd_df.loc[mask_keep].reset_index(drop=True)



# =========================
# Build merged FD df + IC cache
# =========================

def merge_fds(
    base_dir: str,
    fd_s_name: str,
    fd_q_name: str,
    fd_d_name: str,
    approx_fd_name: str,
    include_fd_s: bool,
    include_fd_q: bool,
    include_fd_d: bool,
    include_approx_fd: bool,
    approx_error_threshold: float,
) -> pd.DataFrame:
    paths = []
    if include_fd_s:
        paths.append(os.path.join(base_dir, fd_s_name))
    if include_fd_q:
        paths.append(os.path.join(base_dir, fd_q_name))
    if include_fd_d:
        paths.append(os.path.join(base_dir, fd_d_name))

    dfs = [read_fd_csv(p, expect_cols=("lhs", "rhs")) for p in paths]
    merged = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["lhs", "rhs"])
    merged = merged.drop_duplicates(subset=["lhs", "rhs"]).reset_index(drop=True)

    if include_approx_fd:
        approx_path = os.path.join(base_dir, approx_fd_name)
        if os.path.exists(approx_path):
            approx_df = pd.read_csv(approx_path)
            required = ["lhs", "rhs", "approx_error"]
            missing = [c for c in required if c not in approx_df.columns]
            if missing:
                raise ValueError(
                    f"File {approx_path} missing required columns: {missing}. Found: {list(approx_df.columns)}"
                )

            keep = approx_df.loc[:, ["lhs", "rhs", "approx_error"]].copy()
            keep["lhs"] = keep["lhs"].apply(normalize_lhs)
            keep["rhs"] = keep["rhs"].apply(normalize_rhs)
            keep["approx_error"] = pd.to_numeric(keep["approx_error"], errors="coerce")
            keep = keep.dropna(subset=["approx_error"])
            keep = keep[keep["approx_error"] <= float(approx_error_threshold)]
            keep = keep.loc[:, ["lhs", "rhs"]].drop_duplicates(subset=["lhs", "rhs"])

            merged = pd.concat([merged, keep], ignore_index=True).drop_duplicates(subset=["lhs", "rhs"])

    merged = merged[(merged["lhs"] != "") & (merged["rhs"] != "")]
    merged = merged.sort_values(["lhs", "rhs"], kind="stable").reset_index(drop=True)
    return merged


def build_ic_cache(
    base_dir: str,
    domain_s_name: str,
    domain_q_name: str,
    domain_d_name: str,
    denial_d_name: str,
    include_domain_s: bool,
    include_domain_q: bool,
    include_domain_d: bool,
    include_denial_d: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - ic_cache dict (list[dict])
      - domain_df (normalized)
      - denial_df (normalized)
    """
    domain_dfs: List[pd.DataFrame] = []

    if include_domain_s:
        domain_dfs.append(read_csv_if_exists(os.path.join(base_dir, domain_s_name)))
    if include_domain_q:
        domain_dfs.append(read_csv_if_exists(os.path.join(base_dir, domain_q_name)))
    if include_domain_d:
        domain_dfs.append(read_csv_if_exists(os.path.join(base_dir, domain_d_name)))

    domain_df = pd.concat(domain_dfs, ignore_index=True) if any(not d.empty for d in domain_dfs) else pd.DataFrame()
    domain_df = normalize_common_constraint_columns(domain_df)

    if not domain_df.empty:
        domain_df = domain_df.drop_duplicates().reset_index(drop=True)

    denial_df = read_csv_if_exists(os.path.join(base_dir, denial_d_name)) if include_denial_d else pd.DataFrame()
    denial_df = normalize_common_constraint_columns(denial_df)

    if not denial_df.empty:
        denial_df = denial_df.drop_duplicates().reset_index(drop=True)

    ic_cache = {
        "domain_constraints": domain_df.to_dict(orient="records") if not domain_df.empty else [],
        "denial_constraints": denial_df.to_dict(orient="records") if not denial_df.empty else [],
    }
    return ic_cache, domain_df, denial_df


def _constraints_to_long_csv(
    fd_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    denial_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a single "long" CSV:
      type, lhs, rhs, approx_error, ...extra (json-like string) ...
    Since IC schemas vary, we preserve full rows in 'raw' column.
    """
    rows: List[Dict[str, Any]] = []

    if not fd_df.empty:
        for _, r in fd_df.iterrows():
            rows.append(
                {
                    "constraint_type": "fd",
                    "lhs": r.get("lhs", ""),
                    "rhs": r.get("rhs", ""),
                    "raw": "",
                }
            )

    if not domain_df.empty:
        for _, r in domain_df.iterrows():
            rows.append(
                {
                    "constraint_type": "domain",
                    "lhs": r.get("lhs", "") if "lhs" in domain_df.columns else "",
                    "rhs": r.get("rhs", "") if "rhs" in domain_df.columns else "",
                    "raw": r.to_dict(),
                }
            )

    if not denial_df.empty:
        for _, r in denial_df.iterrows():
            rows.append(
                {
                    "constraint_type": "denial",
                    "lhs": r.get("lhs", "") if "lhs" in denial_df.columns else "",
                    "rhs": r.get("rhs", "") if "rhs" in denial_df.columns else "",
                    "raw": r.to_dict(),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # make raw deterministic-ish
    if "raw" in out.columns:
        out["raw"] = out["raw"].map(lambda x: "" if x == "" else str(x))

    return out


def build_constraints_cache(
    base_dir: str,
    out_cache_pkl: str,
    out_csv: str,
    fd_s_name: str,
    fd_q_name: str,
    fd_d_name: str,
    approx_fd_name: str,
    domain_s_name: str,
    domain_q_name: str,
    domain_d_name: str,
    denial_d_name: str,
    include_fd_s: bool,
    include_fd_q: bool,
    include_fd_d: bool,
    include_approx_fd: bool,
    approx_error_threshold: float,
    include_domain_s: bool,
    include_domain_q: bool,
    include_domain_d: bool,
    include_denial_d: bool,
    warm_cache: bool,
    config: Dict[str, Any],
    mode_provenance: bool,
) -> None:
    fd_df = merge_fds(
        base_dir=base_dir,
        fd_s_name=fd_s_name,
        fd_q_name=fd_q_name,
        fd_d_name=fd_d_name,
        approx_fd_name=approx_fd_name,
        include_fd_s=include_fd_s,
        include_fd_q=include_fd_q,
        include_fd_d=include_fd_d,
        include_approx_fd=include_approx_fd,
        approx_error_threshold=approx_error_threshold,
    )
    label_col = set(config.get("LABEL_COL") or [])
    drop_cols = set(config.get("DROP_COLS") or [])
    drop_cols = label_col.union(drop_cols)
    
    fd_df = drop_fds_by_config(
        fd_df=fd_df,
        drop_cols=drop_cols,
        mode_provenance=mode_provenance,
    )

    fds = build_fds(fd_df, lhs_col="lhs", rhs_col="rhs")
    engine = FDClosureEngine(fds)
    index = ClosureIndex(engine)

    if warm_cache and not fd_df.empty:
        warm_cache_singletons_and_lhs(fd_df, index)

    ic_cache, domain_df, denial_df = build_ic_cache(
        base_dir=base_dir,
        domain_s_name=domain_s_name,
        domain_q_name=domain_q_name,
        domain_d_name=domain_d_name,
        denial_d_name=denial_d_name,
        include_domain_s=include_domain_s,
        include_domain_q=include_domain_q,
        include_domain_d=include_domain_d,
        include_denial_d=include_denial_d,
    )

    payload = {
        # "version": 1,
        "base_dir": os.path.abspath(base_dir),
        "settings": {
            "include_fd_s": include_fd_s,
            "include_fd_q": include_fd_q,
            "include_fd_d": include_fd_d,
            "include_approx_fd": include_approx_fd,
            "approx_error_threshold": float(approx_error_threshold),
            "include_domain_s": include_domain_s,
            "include_domain_q": include_domain_q,
            "include_domain_d": include_domain_d,
            "include_denial_d": include_denial_d,
            "warm_cache": warm_cache,
        },
        # "constraints": {
        #     "fd_rows": fd_df.to_dict(orient="records"),
        #     "domain_rows": domain_df.to_dict(orient="records") if not domain_df.empty else [],
        #     "denial_rows": denial_df.to_dict(orient="records") if not denial_df.empty else [],
        # }, # raw constraints
        "fd": {
            "closure_cache": index.cache,
        },
        "ic": ic_cache,
    }

    # --- write cache pkl ---
    out_cache_path = out_cache_pkl if os.path.isabs(out_cache_pkl) else os.path.join(base_dir, out_cache_pkl)
    os.makedirs(os.path.dirname(out_cache_path) or ".", exist_ok=True)
    with open(out_cache_path, "wb") as f:
        pickle.dump(payload, f)

    # --- write constraints csv (ALL constraints) ---
    out_csv_path = out_csv if os.path.isabs(out_csv) else os.path.join(base_dir, out_csv)
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)

    final_df = _constraints_to_long_csv(fd_df=fd_df, domain_df=domain_df, denial_df=denial_df)
    final_df.to_csv(out_csv_path, index=False)


def main():
    p = argparse.ArgumentParser()

    # ===== required =====
    p.add_argument("--base-dir", required=True)
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--out", required=True)         # now cache path (CONSTRAINTS_CACHE)
    p.add_argument("--out-csv", required=True)     # new: constraints final csv (CONSTRAINTS_FINAL)
    p.add_argument("--approx-error-threshold", required=True)

    # ===== required: FD / IC file names =====
    p.add_argument("--fd-s", required=True)
    p.add_argument("--fd-q", required=True)
    p.add_argument("--fd-d", required=True)
    p.add_argument("--approx-fd", required=True)

    p.add_argument("--domain-s", required=True)
    p.add_argument("--domain-q", required=True)
    p.add_argument("--domain-d", required=True)
    p.add_argument("--denial-d", required=True)

    # ===== switches (default) =====
    p.add_argument("--include-fd-s", default="True")
    p.add_argument("--include-fd-q", default="True")
    p.add_argument("--include-fd-d", default="True")
    p.add_argument("--include-approx-fd", default="True")

    p.add_argument("--include-domain-s", default="True")
    p.add_argument("--include-domain-q", default="True")
    p.add_argument("--include-domain-d", default="True")
    p.add_argument("--include-denial-d", default="True")

    # ===== mode flags =====
    p.add_argument(
        "--mode-provenance",
        action="store_true",
        help="If set, keep FDs in drop_cols; otherwise drop them"
    )

    p.add_argument("--warm-cache", default="True")

    args = p.parse_args()
    cfg = load_config(os.path.join(args.base_dir, args.config))

    build_constraints_cache(
        base_dir=args.base_dir,
        out_cache_pkl=args.out,
        out_csv=args.out_csv,

        fd_s_name=args.fd_s,
        fd_q_name=args.fd_q,
        fd_d_name=args.fd_d,
        approx_fd_name=args.approx_fd,

        domain_s_name=args.domain_s,
        domain_q_name=args.domain_q,
        domain_d_name=args.domain_d,
        denial_d_name=args.denial_d,

        include_fd_s=_parse_bool(args.include_fd_s),
        include_fd_q=_parse_bool(args.include_fd_q),
        include_fd_d=_parse_bool(args.include_fd_d),
        include_approx_fd=_parse_bool(args.include_approx_fd),

        include_domain_s=_parse_bool(args.include_domain_s),
        include_domain_q=_parse_bool(args.include_domain_q),
        include_domain_d=_parse_bool(args.include_domain_d),
        include_denial_d=_parse_bool(args.include_denial_d),

        approx_error_threshold=float(args.approx_error_threshold),
        warm_cache=_parse_bool(args.warm_cache),
        config=cfg,
        mode_provenance=args.mode_provenance,
    )


if __name__ == "__main__":
    main()
