#!/usr/bin/env python3
import os
import time
import random
import numpy as np

import argparse
from pathlib import Path
import pickle
from typing import Dict, Tuple, Iterable, Optional, Any, List, Set
from collections import defaultdict, Counter

import yaml
from tqdm import tqdm
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import joblib
import json
from collections import deque, defaultdict
from shap.explainers._kernel import KernelExplainer
import math
from scipy.special import comb as _binom

import torch
import torch.backends.cudnn as cudnn

# NOTE JF: added for caching and parallelization.
import functools
import multiprocessing
from parallel_utils import parallel_loop

# Module-level slot used by _fork_mc_worker (set before pool creation via fork)
_MC_ROW_FN = None

def _fork_mc_worker(args):
    """Top-level picklable worker: inherited via fork, not pickled itself."""
    row_i, seed = args
    return _MC_ROW_FN(row_i, seed)

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["SEGMENT_DISABLE"] = "1"
    os.environ["POSTHOG_DISABLED"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_DISABLED"] = "true"

    random.seed(seed)
    np.random.seed(seed)

    # Torch CPU
    torch.manual_seed(seed)

    # Torch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        cudnn.benchmark = False
        cudnn.deterministic = True

    # Torch MPS (Apple)
    if hasattr(torch.backends, "mps"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


# =========================================================
# Helpers
# =========================================================
def _dbg_maker(enabled: bool, lim: int, every: int = 1, prefix: str = "[dbg]"):
    state = {"n": 0}
    def _dbg(msg: str):
        if not enabled:
            return
        state["n"] += 1
        if state["n"] > lim:
            return
        if (state["n"] % max(1, every)) != 0:
            return
        print(f"{prefix} {msg}")
    return _dbg

def _dbg3(dbg, *, before: str, because: str, after: str, tag: str = ""):
    """
      1) before
      2) because (FD / ID / Provenance)
      3) after
    """
    t = f"{tag} " if tag else ""
    dbg(f"{t}BEFORE  {before}")
    dbg(f"{t}BECAUSE {because}")
    dbg(f"{t}AFTER   {after}")


def _summ_mask(feature_names: List[str], m: np.ndarray, max_cols: int = 12) -> str:
    idx = np.where(np.asarray(m).reshape(-1) == 1)[0].tolist()
    cols = [feature_names[i] for i in idx]
    head = cols[:max_cols]
    tail = (" ..." if len(cols) > max_cols else "")
    return f"on={len(cols)}/{len(feature_names)} {head}{tail}"

def _diff_cols(before_cols: List[str], after_cols: List[str], max_cols: int = 12) -> str:
    b = set(before_cols); a = set(after_cols)
    added = sorted(list(a - b))
    removed = sorted(list(b - a))
    ad = added[:max_cols]
    rd = removed[:max_cols]
    return f"added={ad}{' ...' if len(added)>max_cols else ''} | removed={rd}{' ...' if len(removed)>max_cols else ''}"

def _cell_change_examples(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    cols: List[str],
    *,
    k: int = 3,
) -> List[str]:
    """
    Return up to k strings like:
      "row=5 col=oh1: 0 -> 1"
    Only for cols in `cols`.
    """
    ex = []
    if before_df is None or after_df is None or not cols:
        return ex
    
    cols = [c for c in cols if c in before_df.columns and c in after_df.columns]
    if not cols:
        return ex

    b = before_df[cols]
    a = after_df[cols]
    eq = b.eq(a) | (b.isna() & a.isna())
    diff_locs = np.argwhere((~eq).to_numpy())

    for t in range(min(k, diff_locs.shape[0])):
        r, j = diff_locs[t]
        col = cols[j]
        vb = b.iat[r, j]
        va = a.iat[r, j]
        ex.append(f"row={r} col={col}: {vb} -> {va}")
    return ex



def _key_from_cols(cols: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(cols)))


def _copy_path(path: Path) -> Path:
    if path.suffix:
        return path.with_name(path.stem + "_copy" + path.suffix)
    return path.with_name(path.name + "_copy")


def _mask_key(m: np.ndarray) -> bytes:
    m = np.asarray(m)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8, copy=False)
    m = m.reshape(-1)
    return np.packbits(m, bitorder="little").tobytes()

def _is_isnull_token(v: Any) -> bool:
    if v is None:
        return True
    try:
        if isinstance(v, float) and pd.isna(v):
            return True
    except Exception:
        pass
    s = str(v).strip().upper()
    return s in ("IS_NULL", "NULL", "NAN", "")

def _normalize_is_null_op_value(op: Any, value: Any) -> Tuple[str, Any]:
    """
    Convert ("=", "IS_NULL") or ("=", None) into ("IS_NULL", None).
    Keep everything else as-is.
    """
    op_u = str(op or "").strip().upper()
    if op_u in ("=", "==") and _is_isnull_token(value):
        return ("IS_NULL", None)
    if op_u == "IS_NULL":
        return ("IS_NULL", None)
    return (op_u, value)


class ProvenanceIndex:
    """
    Build row-wise provenance lookup:
      given (idcol, key_cols subset, key_vals) -> (candidate_ids tuple, rhs_const_map)
    rhs_const_map[rhs_feature] = True if rhs_feature is constant within that key group.
    """

    def __init__(self, *, idcol_to_rhs, idcol_to_rules, feature_cols, df_ref, drop_cols):
        self.idcol_to_rhs = {k: list(v) for k, v in idcol_to_rhs.items()}
        self.idcol_to_rules = {k: list(v) for k, v in idcol_to_rules.items()}
        self.feature_cols = list(feature_cols)
        self.df_ref = df_ref
        self.drop_set = set(drop_cols)
        self.feat_set = set(feature_cols)

        self.det_cols = {}
        self.maps = {}

        # col -> nunique
        self._nunique = {c: int(self.df_ref[c].nunique(dropna=False)) for c in self.feature_cols if c in self.df_ref.columns}
        self._build_det_cols()

        # support composite id key for up to 2 drop_cols
        self.drop_cols = list(drop_cols)

        self.pair_idcol = None
        if len(self.drop_cols) == 2:
            a, b = self.drop_cols[0], self.drop_cols[1]
            if (a in self.df_ref.columns) and (b in self.df_ref.columns):
                pair_col = f"__pair__{a}__{b}"
                self.pair_idcol = pair_col

                # materialize composite id column as tuple(a,b)
                if pair_col not in self.df_ref.columns:
                    self.df_ref[pair_col] = list(zip(self.df_ref[a].tolist(), self.df_ref[b].tolist()))

                # treat composite id as an additional drop identifier
                self.drop_set.add(pair_col)

                # union RHS and RULES from both ids so pair can expand masks too
                rhs_union = []
                rhs_union.extend(self.idcol_to_rhs.get(a, []))
                rhs_union.extend(self.idcol_to_rhs.get(b, []))
                # keep order but unique
                seen = set()
                rhs_union_unique = []
                for x in rhs_union:
                    if x not in seen:
                        seen.add(x)
                        rhs_union_unique.append(x)
                self.idcol_to_rhs[pair_col] = rhs_union_unique

                rules_union = []
                rules_union.extend(self.idcol_to_rules.get(a, []))
                rules_union.extend(self.idcol_to_rules.get(b, []))
                self.idcol_to_rules[pair_col] = rules_union

        # Pre-build row-index sets for fast multi-intersect (avoids pandas isin in hot path).
        # Use positional indices (matching _col_arr) rather than label-based df.index values.
        self._id_row_sets: Dict[str, Dict[Any, frozenset]] = {}
        for idc in self.drop_set:
            if idc in self.df_ref.columns:
                arr = self.df_ref[idc].values
                id_map: Dict[Any, list] = {}
                for pos, val in enumerate(arr):
                    if val not in id_map:
                        id_map[val] = []
                    id_map[val].append(pos)
                self._id_row_sets[idc] = {k: frozenset(v) for k, v in id_map.items()}

        # Pre-store column arrays for fast nunique on row subsets
        self._col_arr: Dict[str, Any] = {c: self.df_ref[c].values for c in self.df_ref.columns}

        # Inverted index: eliminates per-coalition pandas groupby in get_map.
        # Built once at init; lookups are O(|key_cols|) frozenset intersections.
        self._inv: Dict[str, Dict[str, Dict[Any, frozenset]]] = {}
        self._entity_feat_vals: Dict[str, Dict[Any, Dict[str, frozenset]]] = {}
        self._build_inv_index()


    def _build_inv_index(self):
        """Build per-feature inverted index for O(|key_cols|) candidate-ID lookup.

        self._inv[idc][feat][canon_feat_val]  -> frozenset of raw idc values
        self._entity_feat_vals[idc][idc_val][feat] -> frozenset of canon feat values
        """
        for idc in self.drop_set:
            if idc not in self.df_ref.columns:
                continue
            idc_arr = self.df_ref[idc].values
            rhs_set = {c for c in self.idcol_to_rhs.get(idc, [])
                       if c in self.feat_set and c in self.df_ref.columns}

            feat_inv: Dict[str, Dict[Any, set]] = {}
            entity_rhs: Dict[Any, Dict[str, set]] = {}

            for feat in self.feature_cols:
                if feat not in self.df_ref.columns:
                    continue
                feat_arr = self.df_ref[feat].values
                f_to_ids: Dict[Any, set] = {}
                is_rhs = feat in rhs_set

                for pos in range(len(idc_arr)):
                    iv = idc_arr[pos]
                    try:
                        if pd.isna(iv):
                            continue
                    except Exception:
                        pass
                    fv = _canon_scalar(feat_arr[pos])
                    if fv not in f_to_ids:
                        f_to_ids[fv] = set()
                    f_to_ids[fv].add(iv)

                    if is_rhs:
                        if iv not in entity_rhs:
                            entity_rhs[iv] = {}
                        if feat not in entity_rhs[iv]:
                            entity_rhs[iv][feat] = set()
                        entity_rhs[iv][feat].add(fv)

                feat_inv[feat] = {k: frozenset(v) for k, v in f_to_ids.items()}

            self._inv[idc] = feat_inv
            self._entity_feat_vals[idc] = {
                iv: {f: frozenset(vs) for f, vs in feats.items()}
                for iv, feats in entity_rhs.items()
            }

    def _compute_const_rhs(self, idcol: str, candidate_ids: set) -> set:
        """Return RHS features that are constant across all candidate entities."""
        entity_vals = self._entity_feat_vals.get(idcol, {})
        rhs_list = [c for c in self.idcol_to_rhs.get(idcol, [])
                    if c in self.feat_set and c in self.df_ref.columns]
        const_rhs = set()
        for feat in rhs_list:
            all_vals: set = set()
            for cid in candidate_ids:
                all_vals |= entity_vals.get(cid, {}).get(feat, set())
                if len(all_vals) > 1:
                    break
            if len(all_vals) == 1:
                const_rhs.add(feat)
        return const_rhs

    def _build_det_cols(self):
        for idc, rhs_list in self.idcol_to_rhs.items():
            if idc not in self.drop_set:
                continue
            det = [c for c in rhs_list if c in self.feat_set and c in self.df_ref.columns]
            det = sorted(set(det))
            # if det:
            #     self.det_cols[idc] = det
            if det:
                # --- order by selectivity (nunique desc) to reduce ambiguity early ---
                det_sorted = sorted(det, key=lambda c: (-self._nunique.get(c, 0), c))
                self.det_cols[idc] = det_sorted

    def get_map(self, idc: str, key_cols: Tuple[str, ...]):
        key = (idc, tuple(key_cols))
        if key in self.maps:
            return self.maps[key]

        if idc not in self.df_ref.columns:
            self.maps[key] = {}
            return self.maps[key]

        rhs_list = [c for c in self.idcol_to_rhs.get(idc, []) if c in self.feat_set and c in self.df_ref.columns]
        use_cols = list(key_cols) + [idc] + [c for c in rhs_list if c != idc and c not in set(key_cols)]
        use_cols = list(dict.fromkeys(use_cols))

        sub = self.df_ref[use_cols].copy()
        g = sub.groupby(list(key_cols), dropna=False)

        ids_series = g[idc].apply(lambda s: tuple(sorted(pd.unique(s.dropna()))))
        rhs_nunique = {rhs: g[rhs].nunique(dropna=False) for rhs in rhs_list}

        m = {}
        for key_vals, ids_tuple in ids_series.items():
            key_vals_t = (key_vals,) if len(key_cols) == 1 else tuple(key_vals)
            const_rhs = set()
            for rhs in rhs_list:
                try:
                    nu = rhs_nunique[rhs].loc[key_vals]
                except Exception:
                    nu = None
                if nu == 1:
                    const_rhs.add(rhs)
            m[key_vals_t] = (ids_tuple, const_rhs)

        self.maps[key] = m
        return m


    def infer_ids(self, *, idcol: str, key_cols: Tuple[str, ...], x_row: dict, threshold: int):
        mp = self.get_map(idcol, key_cols)
        key_vals = tuple(_canon_scalar(x_row.get(c, None)) for c in key_cols)
        hit = mp.get(key_vals)
        if hit is None:
            return tuple(), set()
        ids_tuple, const_rhs = hit
        if 0 < len(ids_tuple) <= int(threshold):
            return ids_tuple, set(const_rhs)
        return tuple(), set()
    
    def infer_ids_and_const(self, *, idcol: str, key_cols: Tuple[str, ...], x_row: dict):
        """
        Always returns (ids_tuple, const_rhs) if group exists, regardless of threshold.
        - ids_tuple: all candidate ids in that group (can be > threshold)
        - const_rhs: rhs features that are constant within that group
        """
        mp = self.get_map(idcol, key_cols)
        key_vals = tuple(_canon_scalar(x_row.get(c, None)) for c in key_cols)
        hit = mp.get(key_vals)
        if hit is None:
            return tuple(), set()
        ids_tuple, const_rhs = hit
        return tuple(ids_tuple), set(const_rhs)
    
    def infer_ids_from_canon(self, *, idcol: str, key_cols: Tuple[str, ...], x_canon: List[Any], col2i: Dict[str,int], threshold: int):
        inv = self._inv.get(idcol)
        if not inv:
            return tuple(), set()
        candidate_set = None
        for feat in key_cols:
            fv = x_canon[col2i[feat]] if feat in col2i else None
            cands = inv.get(feat, {}).get(fv, frozenset())
            if candidate_set is None:
                candidate_set = set(cands)
            else:
                candidate_set &= cands
            if not candidate_set:
                return tuple(), set()
        if not candidate_set:
            return tuple(), set()
        if not (0 < len(candidate_set) <= int(threshold)):
            return tuple(), set()
        return tuple(sorted(candidate_set, key=str)), set()

    def infer_ids_and_const_from_canon(self, *, idcol: str, key_cols: Tuple[str, ...], x_canon: List[Any], col2i: Dict[str,int]):
        inv = self._inv.get(idcol)
        if not inv:
            return tuple(), set()
        candidate_set = None
        for feat in key_cols:
            fv = x_canon[col2i[feat]] if feat in col2i else None
            cands = inv.get(feat, {}).get(fv, frozenset())
            if candidate_set is None:
                candidate_set = set(cands)
            else:
                candidate_set &= cands
            if not candidate_set:
                return tuple(), set()
        if not candidate_set:
            return tuple(), set()
        const_rhs = self._compute_const_rhs(idcol, candidate_set)
        return tuple(sorted(candidate_set, key=str)), const_rhs


# =========================================================
# IC (Domain / Denial) utilities
# =========================================================

def _build_ic_repair_fn(
    ic_payload: Dict[str, Any],
    *,
    mode_domain: bool,
    mode_denial: bool,
    debug_enabled: bool = False,
    domain_numerical_mode: str = "minimal_edit",
    seed: int,
) -> Optional[Any]:
    """
    Return a function repair_df(df: pd.DataFrame) -> pd.DataFrame that
    repairs rows to satisfy IC as much as possible.
    Applied at *evaluation time* (i.e., inside f_proba_pos on synth samples).
    """
    if not (mode_domain or mode_denial):
        return None

    ic = (ic_payload or {}).get("ic", {})
    if not isinstance(ic, dict):
        ic = {}

    domain_atoms = ic.get("domain_constraints", []) if mode_domain else []
    denial_rules = ic.get("denial_constraints", []) if mode_denial else []

    # -------- Parse domain: group by cid -> group_id -> list of atoms --------
    domain_by_cid_gid: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    domain_guard_by_cid: Dict[str, Tuple[str, str, Any]] = {}  # cid -> (lhs, op, value)

    def _nan_to_none(v):
        try:
            if isinstance(v, float) and pd.isna(v):
                return None
            if pd.isna(v):
                return None
        except Exception:
            pass
        return v

    def _pick(*vals):
        for v in vals:
            v = _nan_to_none(v)
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            return v
        return None

    if mode_domain:
        for a in (domain_atoms or []):
            if not isinstance(a, dict):
                continue

            # --- legacy schema support ---
            lhs = _pick(a.get("lhs"), a.get("lhs_col"), a.get("guard_lhs"))
            rhs = _pick(a.get("rhs"), a.get("rhs_col"), a.get("target_col"))

            # guard op/value: prefer (op,value) else (lhs_op,lhs_value)
            op_raw  = _pick(a.get("op"), a.get("lhs_op"))
            val_raw = _pick(a.get("value"), a.get("lhs_value"))

            # domain op/value: prefer (domain_op,domain_value) else (rhs_op,rhs_value)
            dop_raw  = _pick(a.get("domain_op"), a.get("rhs_op"))
            dval_raw = _pick(a.get("domain_value"), a.get("rhs_value"))

            # normalize null tokens ("= IS_NULL" etc.)
            op_u,  val_n  = _normalize_is_null_op_value(op_raw,  val_raw)
            dop_u, dval_n = _normalize_is_null_op_value(dop_raw, _ic_parse_domain_value(dval_raw))

            a2 = dict(a)
            a2["lhs"] = lhs
            a2["rhs"] = rhs
            a2["op"] = op_u
            a2["value"] = val_n
            a2["domain_op"] = str(dop_u or "").strip().upper()
            a2["domain_value"] = dval_n

            cid = str(_pick(a2.get("cid"), a2.get("constraint_id"), ""))
            gid = int(_pick(a2.get("group_id"), 0) or 0)

            domain_by_cid_gid[cid][gid].append(a2)
            if cid not in domain_guard_by_cid:
                domain_guard_by_cid[cid] = (a2["lhs"], a2["op"], a2["value"])


    # -------- Parse denial rules --------
    denial_norm: List[Dict[str, Any]] = []
    if mode_denial:
        for r in (denial_rules or []):
            if not isinstance(r, dict):
                continue
            ic_id = r.get("ic_id")
            kind = str(r.get("kind", "")).strip().lower()
            cols_s = str(r.get("cols", "")).strip()
            cols = [c.strip() for c in cols_s.split(",") if c.strip()]
            k = r.get("k", None)
            parity_bit = r.get("parity_bit", None)

            if kind == "onehot":
                kind_norm = "onehot"
            elif kind in ("parity", "xor"):
                kind_norm = "parity"
            else:
                kind_norm = kind  # unknown -> no-op (or strict). We'll no-op to avoid breaking.

            denial_norm.append(
                {"ic_id": ic_id, "kind": kind_norm, "cols": cols, "k": k, "parity_bit": parity_bit}
            )

    # -------- Debug-only WHY helper (to avoid runtime cost when --debug is off) --------
    denial_by_col = None
    domain_by_rhs = None
    _why_for_cell = None

    if debug_enabled:
        # denial rule: col -> list of rules that mention it
        denial_by_col = defaultdict(list)
        for rule in denial_norm:
            for c in (rule.get("cols", []) or []):
                denial_by_col[str(c)].append(rule)

        # domain atom: rhs -> list of (cid, gid, atom)
        domain_by_rhs = defaultdict(list)
        if mode_domain:
            for cid, gid_map in domain_by_cid_gid.items():
                for gid, atoms in gid_map.items():
                    for a in atoms:
                        rhs = a.get("rhs")
                        if rhs is not None:
                            domain_by_rhs[str(rhs)].append((cid, int(gid), a))

        def _fmt(v):
            # small pretty formatter
            if v is None:
                return "None"
            try:
                if isinstance(v, float) and pd.isna(v):
                    return "NaN"
            except Exception:
                pass
            return str(v)

        def _why_for_cell_impl(
            *,
            before_row: pd.Series,
            after_row: pd.Series,
            col: str,
            fixed_cols: Set[str],
        ) -> str:
            col = str(col)

            # --- Denial: onehot/parity ---
            rules = (denial_by_col.get(col, []) if denial_by_col is not None else [])
            for r in rules:
                kind = r.get("kind")
                cols = [c for c in (r.get("cols", []) or []) if isinstance(c, str)]
                cols = [c for c in cols if c in before_row.index]

                if kind == "onehot":
                    k = r.get("k", None)
                    k0 = int(k) if (k is not None and str(k) != "nan") else 1

                    # cols present in row
                    cols = [c for c in cols if c in before_row.index]

                    def to01(v):
                        vv = _coerce_num(v)
                        if vv is None:
                            b01 = _coerce_01(v)
                            if b01 is None:
                                return 0
                            return k0 if b01 == 1 else 0
                        return k0 if float(vv) >= (k0 / 2.0) else 0

                    # group context (before/after) + mask
                    mask_map = {c: (1 if c in fixed_cols else 0) for c in cols}
                    fixed = [c for c in cols if mask_map[c] == 1]
                    free  = [c for c in cols if mask_map[c] == 0]

                    before_map = {c: to01(before_row.get(c, None)) for c in cols}
                    after_map  = {c: to01(after_row.get(c, None)) for c in cols}

                    ones_before = [c for c in cols if before_map[c] != 0]
                    ones_after  = [c for c in cols if after_map[c] != 0]

                    fixed_ones_before = [c for c in fixed if before_map[c] != 0]
                    fixed_ones_after  = [c for c in fixed if after_map[c] != 0]

                    # what changed in this group?
                    changed_in_group = []
                    for c in cols:
                        if before_map[c] != after_map[c]:
                            changed_in_group.append(f"{c}:{before_map[c]}->{after_map[c]}")

                    # infer keep (post-state nonzero; if multiple, list them)
                    keep_after = [c for c in cols if after_map[c] != 0]

                    # build a human-readable WHY
                    # note: "exactly-one" means nonzero count must be exactly 1
                    why_parts = []
                    why_parts.append("denial(onehot): exactly-one")
                    why_parts.append(f"cols={cols}")
                    why_parts.append(f"mask(1=fixed)={mask_map}")
                    why_parts.append(f"fixed={fixed} free={free}")
                    why_parts.append(f"before_vals={before_map} (ones={ones_before})")
                    why_parts.append(f"after_vals={after_map} (ones={ones_after})")

                    # decision summary
                    if len(fixed_ones_before) > 1:
                        why_parts.append(f"NOTE: infeasible (fixed ones >1): fixed_ones_before={fixed_ones_before} -> cannot repair without touching fixed")
                    elif len(fixed_ones_before) == 1:
                        # fixed has the 1, so free must be 0
                        why_parts.append(f"decision: fixed already 1 at {fixed_ones_before[0]} -> forced all free to 0")
                    else:
                        # no fixed ones, so choose/keep a free one (or create one if 000)
                        if len(ones_before) == 0:
                            why_parts.append(f"decision: before was all-zero -> created a 1 in free (chosen={keep_after})")
                        else:
                            why_parts.append(f"decision: before had multiple ones -> kept/selected one in free (chosen={keep_after}) and zeroed others (free only)")

                    if changed_in_group:
                        why_parts.append(f"changed_in_group={changed_in_group}")

                    # also explain THIS cell explicitly (col argument)
                    c0 = str(col)
                    if c0 in cols:
                        why_parts.append(
                            f"this_cell: {c0} mask={mask_map[c0]} value {before_map[c0]}->{after_map[c0]} "
                            + ("(free adjusted)" if mask_map[c0] == 0 else "(fixed; should not change)")
                        )

                    return " | ".join(why_parts)

                if kind == "parity":
                    b = r.get("parity_bit", None)
                    pb = None if b is None or str(b) == "nan" else int(b)
                    if pb not in (0, 1):
                        continue

                    cols = [c for c in cols if c in before_row.index]

                    mask_map = {c: (1 if c in fixed_cols else 0) for c in cols}
                    fixed = [c for c in cols if mask_map[c] == 1]
                    free  = [c for c in cols if mask_map[c] == 0]

                    before_bits = {c: (_coerce_01(before_row.get(c, None)) or 0) for c in cols}
                    after_bits  = {c: (_coerce_01(after_row.get(c, None)) or 0) for c in cols}

                    s_before = sum(before_bits.values()) % 2
                    s_after  = sum(after_bits.values()) % 2

                    # infer flipped col (among free) by comparing before/after
                    flipped = [c for c in free if before_bits.get(c, 0) != after_bits.get(c, 0)]
                    changed_in_group = []
                    for c in cols:
                        if before_bits[c] != after_bits[c]:
                            changed_in_group.append(f"{c}:{before_bits[c]}->{after_bits[c]}")

                    why_parts = []
                    why_parts.append("denial(parity): parity_bit")
                    why_parts.append(f"cols={cols}")
                    why_parts.append(f"mask(1=fixed)={mask_map}")
                    why_parts.append(f"fixed={fixed} free={free}")
                    why_parts.append(f"before_bits={before_bits} parity={s_before}")
                    why_parts.append(f"after_bits={after_bits} parity={s_after}")
                    why_parts.append(f"target_parity={pb}")

                    if s_before == pb:
                        why_parts.append("decision: already satisfied -> no change expected")
                    else:
                        if not free:
                            why_parts.append("NOTE: infeasible (no free bits) -> cannot repair without touching fixed")
                        else:
                            why_parts.append(f"decision: flipped free bit(s)={flipped if flipped else '(unknown)'} to satisfy parity")

                    if changed_in_group:
                        why_parts.append(f"changed_in_group={changed_in_group}")

                    c0 = str(col)
                    if c0 in cols:
                        why_parts.append(
                            f"this_cell: {c0} mask={mask_map[c0]} bit {before_bits[c0]}->{after_bits[c0]} "
                            + ("(free adjusted)" if mask_map[c0] == 0 else "(fixed; should not change)")
                        )

                    return " | ".join(why_parts)

            # --- Domain: guard + RHS domain ---
            atoms = (domain_by_rhs.get(col, []) if domain_by_rhs is not None else [])
            for (cid, gid, atom) in atoms:
                # check if guard holds on BEFORE row (repair decision is based on BEFORE)
                lhs = atom.get("lhs")
                op = str(atom.get("op", "")).strip().upper()
                gv = atom.get("value", None)

                lv = before_row.get(lhs, None)
                guard = False
                if op == "=":
                    guard = (_canon_scalar(lv) == _canon_scalar(gv))
                elif op == "IS_NULL":
                    guard = pd.isna(lv)

                if not guard:
                    continue

                dop = str(atom.get("domain_op", "")).strip().upper()
                dval = atom.get("domain_value")
                return f"domain(cid={cid}, gid={gid}): guard({lhs} {op} {gv}) held -> enforced {col} {dop} {dval}"

            return "unknown (no matched domain/denial rule for this col)"


    # -------- Vectorized domain repair helpers --------
    def _repair_domain_inplace(df: pd.DataFrame, rng: np.random.RandomState) -> None:

        def _is_numeric_series(s: pd.Series) -> bool:
            return is_integer_dtype(s) or is_float_dtype(s)
        
        def _sample_from_column_pool(orig: Any, pool: np.ndarray, *, mode: str, rng: np.random.RandomState) -> Any:
            """
            pool: candidate values that already exist in df[col] and satisfy constraints.
            """
            if pool is None or len(pool) == 0:
                return orig

            if mode == "random":
                return pool[int(rng.randint(0, len(pool)))]

            if mode == "weighted_random":
                o = _coerce_num(orig)
                if o is None:
                    return pool[int(rng.randint(0, len(pool)))]

                pool_num = np.array([_coerce_num(v) for v in pool], dtype=float)
                ok = np.isfinite(pool_num)
                if not ok.any():
                    return pool[int(rng.randint(0, len(pool)))]

                pool2 = pool[ok]
                pool_num2 = pool_num[ok]

                d = np.abs(pool_num2 - float(o))
                w = 1.0 / (1.0 + d)
                w = w / w.sum()
                idx = int(rng.choice(len(pool2), p=w))
                return pool2[idx]

            return pool[int(rng.randint(0, len(pool)))]


        def _sample_numeric(orig: float, lo: float, hi: float, *, mode: str, rng: np.random.RandomState) -> float:
            # sanitize
            if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
                # fallback: can't do much, return orig
                return orig

            # clamp orig into [lo,hi] for mode
            try:
                o = float(orig)
            except Exception:
                o = (lo + hi) / 2.0
            if not np.isfinite(o):
                o = (lo + hi) / 2.0
            o_clipped = min(max(o, lo), hi)

            if mode == "minimal_edit":
                return o_clipped

            if mode == "random":
                return float(rng.uniform(lo, hi))

            if mode == "weighted_random":
                # triangular: more mass near original (distance-weighted 느낌)
                if lo == hi:
                    return float(lo)
                return float(rng.triangular(lo, o_clipped, hi))

            # unknown -> minimal
            return o_clipped

        def _sample_from_allowed(orig: Any, allowed: List[Any], *, mode: str, rng: np.random.RandomState) -> Any:
            if not allowed:
                return orig

            # if not numeric allowed -> uniform (or keep if already valid)
            o_num = _coerce_num(orig)
            allowed_num = [(_coerce_num(a), a) for a in allowed]

            # if orig is already allowed (canon-wise), keep it for minimal_edit
            if mode == "minimal_edit":
                oc = _canon_scalar(orig)
                for a in allowed:
                    if _canon_scalar(a) == oc:
                        return orig

            # numeric allowed & orig numeric -> weighted by distance
            if mode == "weighted_random" and o_num is not None and any(an is not None for an, _ in allowed_num):
                weights = []
                items = []
                for an, a_raw in allowed_num:
                    if an is None:
                        continue
                    d = abs(float(an) - float(o_num))
                    w = 1.0 / (1.0 + d)   # distance 기반 가중치 (가까울수록 큼)
                    weights.append(w)
                    items.append(a_raw)
                if items:
                    w = np.asarray(weights, dtype=float)
                    w = w / w.sum()
                    idx = int(rng.choice(len(items), p=w))
                    return items[idx]

            # random / fallback: uniform
            return allowed[int(rng.randint(0, len(allowed)))]


        # For each cid: if guard holds, enforce at least one group in DNF.
        # Strategy:
        #   - If any group is already satisfied, do nothing.
        #   - Else, pick the smallest gid and "project" RHS values to satisfy that group's atoms.
        for cid, guard in domain_guard_by_cid.items():
            lhs, op, gv = guard
            if lhs not in df.columns:
                continue

            if op == "IS_NULL":
                guard_mask = df[lhs].isna()
            elif op == "=":
                # normal equality (gv is not null)
                lv = df[lhs]
                guard_mask = (lv.map(_canon_scalar) == _canon_scalar(gv))
            else:
                continue

            if not guard_mask.any():
                continue

            groups = domain_by_cid_gid.get(cid, {})
            if not groups:
                continue

            # build "satisfied already" mask per group, then combine
            satisfied_any = pd.Series(False, index=df.index)
            group_satisfied: Dict[int, pd.Series] = {}
            
            for gid, atoms in groups.items():
                ok = pd.Series(True, index=df.index)
                for atom in atoms:
                    rhs = atom.get("rhs")
                    if rhs not in df.columns:
                        ok &= False
                        continue
                    dop = str(atom.get("domain_op", "")).strip().upper()
                    dval = _ic_parse_domain_value(atom.get("domain_value"))

                    if dop == ">=":
                        dvn = _coerce_num(dval)
                        if dvn is None:
                            ok &= False
                        else:
                            rvn = pd.to_numeric(df[rhs], errors="coerce")
                            ok &= (rvn >= dvn)
                    elif dop == "<=":
                        dvn = _coerce_num(dval)
                        if dvn is None:
                            ok &= False
                        else:
                            rvn = pd.to_numeric(df[rhs], errors="coerce")
                            ok &= (rvn <= dvn)
                    elif dop == "IN":
                        allowed = _ic_parse_domain_value(dval)
                        if not isinstance(allowed, list):
                            allowed = [allowed]
                        allowed_canon = [a for a in allowed if a is not None]
                        if len(allowed_canon) == 0:
                            ok &= False
                            continue
                        allowed_set = set(_canon_scalar(a) for a in allowed_canon)
                        rv_c = df[rhs].map(_canon_scalar)
                        ok &= rv_c.isin(list(allowed_set))


                    elif dop == "IS_NULL":
                        ok &= df[rhs].isna()
                    else:
                        ok &= False

                group_satisfied[gid] = ok
                satisfied_any |= ok

            # rows where guard holds but no group satisfied
            need_repair = guard_mask & (~satisfied_any)
            if not need_repair.any():
                continue
            # # guard_mask, need_repair 계산 직후
            if debug_enabled:
                print(
                    f"[dbg] [IC-domain] cid={cid} guard({lhs} {op} {gv}) "
                    f"guard_rows={int(guard_mask.sum())} need_repair_rows={int(need_repair.sum())} "
                    f"target_gid={sorted(groups.keys())[0]}"
                )

            # choose target group (smallest gid) and project rhs values to satisfy that group's atoms
            target_gid = sorted(groups.keys())[0]
            atoms = groups[target_gid]

            # --- Aggregate constraints per rhs in this group ---
            rhs_specs = defaultdict(lambda: {"lo": None, "hi": None, "allowed": None, "is_null": False, "eq": None, "pool": None,})

            for atom in atoms:
                rhs = atom.get("rhs")
                if rhs not in df.columns:
                    continue
                dop = str(atom.get("domain_op", "")).strip().upper()
                dval = _ic_parse_domain_value(atom.get("domain_value"))

                spec = rhs_specs[rhs]

                if dop == ">=":
                    dvn = _coerce_num(dval)
                    if dvn is not None:
                        spec["lo"] = dvn if spec["lo"] is None else max(spec["lo"], dvn)

                elif dop == "<=":
                    dvn = _coerce_num(dval)
                    if dvn is not None:
                        spec["hi"] = dvn if spec["hi"] is None else min(spec["hi"], dvn)

                elif dop == "IN":
                    allowed = _ic_parse_domain_value(dval)
                    if not isinstance(allowed, list):
                        allowed = [allowed]
                    allowed = [a for a in allowed if a is not None]
                    if allowed:
                        # intersect if multiple IN atoms exist
                        if spec["allowed"] is None:
                            spec["allowed"] = list(allowed)
                        else:
                            prev = set(_canon_scalar(a) for a in spec["allowed"])
                            now  = set(_canon_scalar(a) for a in allowed)
                            inter = prev & now
                            spec["allowed"] = [a for a in spec["allowed"] if _canon_scalar(a) in inter]

                elif dop == "IS_NULL":
                    spec["is_null"] = True

                elif dop == "=":
                    # treat "= IS_NULL" as null
                    if isinstance(dval, str) and dval.strip().upper() == "IS_NULL":
                        spec["is_null"] = True
                    else:
                        spec["eq"] = dval

            # --- NEW: build per-rhs value pool from already-valid rows (optional) ---
            # pool is used only for random/weighted_random to sample realistic values.
            if domain_numerical_mode in ("random", "weighted_random"):
                # rows where guard holds AND at least one group is satisfied already
                valid_rows = guard_mask & satisfied_any
                if valid_rows.any():
                    for rhs, spec in rhs_specs.items():
                        if rhs not in df.columns:
                            continue

                        # start from values observed in valid rows
                        pool = df.loc[valid_rows, rhs].dropna().to_numpy()

                        # filter pool by constraints collected in spec
                        # priority: IS_NULL / EQ -> pool not meaningful
                        if spec.get("is_null", False):
                            spec["pool"] = np.array([], dtype=object)
                            continue
                        if spec.get("eq", None) is not None:
                            spec["pool"] = np.array([spec["eq"]], dtype=object)
                            continue

                        allowed = spec.get("allowed", None)
                        if allowed is not None:
                            allowed_set = set(_canon_scalar(a) for a in allowed if a is not None)
                            pool2 = []
                            for v in pool:
                                if _canon_scalar(v) in allowed_set:
                                    pool2.append(v)
                            pool = np.asarray(pool2, dtype=object)

                        lo = spec.get("lo", None)
                        hi = spec.get("hi", None)
                        if lo is not None or hi is not None:
                            pool_num = np.array([_coerce_num(v) for v in pool], dtype=float)
                            ok = np.isfinite(pool_num)
                            if lo is not None:
                                ok &= (pool_num >= float(lo))
                            if hi is not None:
                                ok &= (pool_num <= float(hi))
                            pool = pool[ok] if hasattr(pool, "__len__") else np.array([], dtype=object)

                        # finally store
                        spec["pool"] = pool
                else:
                    # no valid rows to form pool
                    for rhs, spec in rhs_specs.items():
                        spec["pool"] = np.array([], dtype=object)

            
            
            # --- Apply repair per rhs, only on rows needing repair ---
            for rhs, spec in rhs_specs.items():
                if rhs not in df.columns:
                    continue

                # rows we may touch (guard holds & no group satisfied)
                idx = df.index[need_repair].to_numpy()
                if idx.size == 0:
                    continue

                # Highest priority: IS_NULL / EQ
                if spec["is_null"]:
                    if debug_enabled:
                        sample_idx = idx[:3]
                        before = [df.at[i, rhs] for i in sample_idx]
                    df.loc[idx, rhs] = np.nan
                    if debug_enabled:
                        after = [df.at[i, rhs] for i in sample_idx]
                        print(f"[dbg] [IC-domain] SET-NULL rhs={rhs} rows={len(idx)} ex={list(zip(before, after))}")
                    continue

                if spec["eq"] is not None:
                    if debug_enabled:
                        sample_idx = idx[:3]
                        before = [df.at[i, rhs] for i in sample_idx]
                    df.loc[idx, rhs] = spec["eq"]
                    if debug_enabled:
                        after = [df.at[i, rhs] for i in sample_idx]
                        print(f"[dbg] [IC-domain] SET-EQ rhs={rhs} val={spec['eq']} rows={len(idx)} ex={list(zip(before, after))}")
                    continue


                # IN constraint
                allowed = spec["allowed"]
                if allowed is not None:
                    rv = df.loc[idx, rhs]
                    # if already valid, keep for minimal_edit; else sample
                    # validity by canon membership
                    allowed_set = set(_canon_scalar(a) for a in allowed)
                    rv_c = rv.map(_canon_scalar)
                    bad = ~rv_c.isin(list(allowed_set))

                    if bad.any():
                        bad_idx = rv.index[bad].to_numpy()
                        # sample per-row (allowed could be categorical)
                        for rix in bad_idx:
                            orig = df.at[rix, rhs]

                            pool = spec.get("pool", None)
                            if pool is not None and len(pool) > 0 and domain_numerical_mode in ("random", "weighted_random"):
                                newv = _sample_from_column_pool(orig, pool, mode=domain_numerical_mode, rng=rng)
                            else:
                                newv = _sample_from_allowed(orig, allowed, mode=domain_numerical_mode, rng=rng)

                            df.at[rix, rhs] = newv

                            if debug_enabled and (orig != newv):
                                print(f"[dbg] [IC-domain] IN rhs={rhs} row={rix}: {orig} -> {newv} allowed(head)={allowed[:5]}")
                    continue

                # Numeric range constraint (lo/hi could be one-sided or two-sided)
                lo = spec["lo"]
                hi = spec["hi"]

                # If no numeric bounds -> nothing to do
                if lo is None and hi is None:
                    continue

                # Determine effective lo/hi
                # If one-sided, make it a "clamp only" in minimal_edit,
                # and for random/weighted_random we still need a finite interval:
                # We'll infer the missing side from observed data range (safe fallback).
                s = df[rhs]
                is_num = _is_numeric_series(s)
                if not is_num:
                    # if not numeric dtype, fallback to minimal_edit behavior via coercion
                    # (still try clamp with numeric coercion)
                    pass

                s_num = pd.to_numeric(s, errors="coerce")
                obs_min = float(np.nanmin(s_num.to_numpy())) if np.isfinite(np.nanmin(s_num.to_numpy())) else None
                obs_max = float(np.nanmax(s_num.to_numpy())) if np.isfinite(np.nanmax(s_num.to_numpy())) else None

                eff_lo = lo if lo is not None else obs_min
                eff_hi = hi if hi is not None else obs_max

                # If still missing (all NaN), fallback: just boundary set
                if eff_lo is None and hi is not None:
                    eff_lo = hi
                if eff_hi is None and lo is not None:
                    eff_hi = lo

                if eff_lo is None or eff_hi is None:
                    # can't define interval -> minimal boundary set if possible
                    if lo is not None:
                        df.loc[idx, rhs] = lo
                    elif hi is not None:
                        df.loc[idx, rhs] = hi
                    continue

                # enforce order
                eff_lo, eff_hi = (eff_lo, eff_hi) if eff_lo <= eff_hi else (eff_hi, eff_lo)

                # find rows out of range
                rvn = pd.to_numeric(df.loc[idx, rhs], errors="coerce").to_numpy()
                bad_mask = ~np.isfinite(rvn)
                if lo is not None:
                    bad_mask |= (rvn < float(lo))
                if hi is not None:
                    bad_mask |= (rvn > float(hi))

                if not bad_mask.any():
                    continue

                bad_idx = idx[bad_mask]

                # sample/correct per-row (vectorized sampling with triangular is possible, but loop is fine here)
                for rix in bad_idx:
                    orig = df.at[rix, rhs]
                    o = _coerce_num(orig)
                    if o is None:
                        o = (eff_lo + eff_hi) / 2.0

                    # newv = _sample_numeric(o, float(eff_lo), float(eff_hi), mode=domain_numerical_mode, rng=rng)
                    pool = spec.get("pool", None)
                    if pool is not None and len(pool) > 0 and domain_numerical_mode in ("random", "weighted_random"):
                        newv = _sample_from_column_pool(orig, pool, mode=domain_numerical_mode, rng=rng)
                        on = _coerce_num(newv)
                        if on is not None:
                            newv = _sample_numeric(on, float(eff_lo), float(eff_hi), mode="minimal_edit", rng=rng)
                    else:
                        newv = _sample_numeric(o, float(eff_lo), float(eff_hi), mode=domain_numerical_mode, rng=rng)


                    # keep integer-ness if the column is integer-ish
                    if is_integer_dtype(df[rhs]) or (
                        is_float_dtype(df[rhs]) and float(newv).is_integer() and float(eff_lo).is_integer() and float(eff_hi).is_integer()
                    ):
                        newv = int(round(newv))

                    orig0 = df.at[rix, rhs]
                    df.at[rix, rhs] = newv
                    if debug_enabled:
                        print(f"[dbg] [IC-domain] RANGE rhs={rhs} row={rix}: {orig0} -> {newv} (lo={lo}, hi={hi}, eff=[{eff_lo},{eff_hi}], mode={domain_numerical_mode})")


    def _repair_denial_inplace(df: pd.DataFrame, *, fixed_cols: Set[str], rng: np.random.RandomState) -> None:

        for rule in denial_norm:
            kind = rule.get("kind")
            cols = rule.get("cols", [])
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue

            if kind == "onehot":
                k = rule.get("k", None)
                k0 = int(k) if (k is not None and str(k) != "nan") else 1

                # work row-wise (safe; denial rows are usually small)
                infeasible = 0

                if not any(c in fixed_cols for c in cols):
                    continue
                for ridx in range(len(df)):
                    # ---- helpers ----
                    def to01(v):
                        vv = _coerce_num(v)
                        if vv is None:
                            b = _coerce_01(v)
                            if b is None:
                                return 0
                            return k0 if b == 1 else 0
                        return k0 if float(vv) >= (k0/2.0) else 0

                    cur = {c: to01(df.iloc[ridx][c]) for c in cols if c in df.columns}
                    ones = [c for c, v in cur.items() if v != 0]
                    total_ones = len(ones)

                    if total_ones == 1:
                        continue

                    fixed = [c for c in cols if c in fixed_cols and c in df.columns]
                    free  = [c for c in cols if c not in fixed_cols and c in df.columns]

                    fixed_ones = [c for c in ones if c in fixed_cols]

                    if len(fixed_ones) > 1:
                        infeasible += 1
                        continue

                    # --- now enforce EXACTLY-ONE, without touching fixed-cols unless necessary ---
                    if len(fixed_ones) == 1:
                        keep = fixed_ones[0]
                        for c in free:
                            df.iat[ridx, df.columns.get_loc(c)] = 0
                        
                        continue

                    if len(free) == 0:
                        infeasible += 1
                        continue

                    cand_keep = [c for c in ones if c in free]
                    if cand_keep:
                        keep = cand_keep[0]
                    else:
                        keep = free[int(rng.randint(0, len(free)))]  # 000

                    for c in free:
                        df.iat[ridx, df.columns.get_loc(c)] = (k0 if c == keep else 0)


                df.attrs["_denial_onehot_infeasible"] = int(df.attrs.get("_denial_onehot_infeasible", 0)) + infeasible



            elif kind == "parity":
                parity_bit = rule.get("parity_bit", None)
                b = None if parity_bit is None or str(parity_bit) == "nan" else int(parity_bit)
                if b not in (0, 1):
                    continue

                if not any(c in fixed_cols for c in cols):
                    continue

                fixed = [c for c in cols if c in fixed_cols]
                free  = [c for c in cols if c not in fixed_cols]

                infeasible = 0
                for ridx in range(len(df)):

                    bits_fixed = []
                    bits_free  = []

                    for c in fixed:
                        bi = _coerce_01(df.iloc[ridx][c])
                        bits_fixed.append(0 if bi is None else bi)
                    for c in free:
                        bi = _coerce_01(df.iloc[ridx][c])
                        bits_free.append(0 if bi is None else bi)

                    s = (sum(bits_fixed) + sum(bits_free)) % 2
                    if s == b:
                        continue

                    if len(free) == 0:
                        infeasible += 1
                        continue

                    # flip last free bit
                    c_flip = free[-1]
                    cur = _coerce_01(df.iloc[ridx][c_flip]) or 0
                    df.iat[ridx, df.columns.get_loc(c_flip)] = 1 - cur

                df.attrs["_denial_parity_infeasible"] = int(df.attrs.get("_denial_parity_infeasible", 0)) + infeasible

    rng = np.random.RandomState(seed + 123)  # deterministic
    def repair_df(df: pd.DataFrame, *, fixed_cols: Optional[Set[str]] = None) -> pd.DataFrame:
        fixed_cols = set(fixed_cols or [])
        out = df.copy()
        # before = out.copy()

        if mode_domain:
            _repair_domain_inplace(out, rng=rng)
        if mode_denial:
            _repair_denial_inplace(out, fixed_cols=fixed_cols, rng=rng)


        # eq = before.eq(out) | (before.isna() & out.isna())
        # changed_cells = int((~eq).to_numpy().sum())


        return out

    if debug_enabled:
        repair_df._why_for_cell = _why_for_cell_impl
    else:
        repair_df._why_for_cell = None

    return repair_df



def _ic_parse_domain_value(v: Any) -> Any:
    """
    domain_value in cache is often a string:
      - "[TWD, KRW, JPY]"  (yaml-safe)
      - "['False']"       (yaml-safe)
      - "18" / 18
    Return parsed python object if possible.
    """
    if isinstance(v, (list, dict)) or v is None:
        return v
    if isinstance(v, (int, float, bool)):
        return v
    if _is_isnull_token(v):
        return None
    s = str(v).strip()
    if s == "":
        return s
    # Try YAML first (handles [A, B] and ['False'])
    try:
        out = yaml.safe_load(s)
        return out
    except Exception:
        return s


@functools.lru_cache(maxsize=32768)
def _canon_scalar(x: Any) -> Any:
    """Canonicalize scalar for equality / membership comparisons."""
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    if pd.isna(x):
        return None
    # pandas types
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        # keep int-like floats as int for stable membership
        xf = float(x)
        if xf.is_integer():
            return int(xf)
        return xf
    # strings
    s = str(x).strip()
    # normalize common bool strings
    if s.lower() in ("true", "t", "1", "yes", "y"):
        return True
    if s.lower() in ("false", "f", "0", "no", "n"):
        return False
    return s


def _coerce_01(x: Any) -> Optional[int]:
    """Coerce a value to 0/1 for parity/onehot checks; return None if impossible."""
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    if pd.isna(x):
        return None
    if isinstance(x, (np.bool_, bool)):
        return 1 if bool(x) else 0
    if isinstance(x, (np.integer, int)):
        xi = int(x)
        if xi in (0, 1):
            return xi
        return None
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        if xf.is_integer():
            xi = int(xf)
            if xi in (0, 1):
                return xi
        return None
    s = str(x).strip()
    if s.lower() in ("true", "t", "1", "yes", "y"):
        return 1
    if s.lower() in ("false", "f", "0", "no", "n"):
        return 0
    return None


def _coerce_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer, int)):
        return float(int(x))
    if isinstance(x, (np.floating, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None



# =========================================================
# ClosureCache
# =========================================================
class ClosureCache:
    """
    Reads constraints cache pkl payload.
      - base rules from payload['fd']['closure_cache'] (original)
      - optionally uses payload['fd']['closure_cache_bg'] (copy)
      - optionally uses payload['fd']['coalition_class_cache'] (copy)

    Derives FD rules (lhs -> rhs) from base closure_cache ONLY.
    Assumption: max |lhs|=2, rhs size=1 sufficient.

    Provides Armstrong closure + provenance.
    Can append new entries into copy sections and flush.
    """

    def __init__(
        self,
        cache_pkl_path: str,
        *,
        enable_copy_write: bool,
        copy_out_path: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ):
        self.src_path = Path(cache_pkl_path)

        if payload is None:
            with open(self.src_path, "rb") as f:
                payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError("constraints cache pkl must be a dict payload")

        fd = payload.get("fd", {})
        if not isinstance(fd, dict):
            raise ValueError("payload['fd'] must be a dict")

        base_cc = fd.get("closure_cache", {})
        if not isinstance(base_cc, dict):
            raise ValueError("payload['fd']['closure_cache'] must be a dict")

        bg_cc = fd.get("closure_cache_bg", {}) or {}
        if not isinstance(bg_cc, dict):
            raise ValueError("payload['fd']['closure_cache_bg'] must be a dict if present")

        class_cc = fd.get("coalition_class_cache", {}) or {}
        if not isinstance(class_cc, dict):
            raise ValueError("payload['fd']['coalition_class_cache'] must be a dict if present")

        def _norm_map(d: dict) -> Dict[Tuple[str, ...], Tuple[str, ...]]:
            def _norm_tuple(t):
                return tuple(x.lower() if isinstance(x, str) else x for x in t)

            out: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
            for k, v in d.items():
                out[_norm_tuple(k)] = _norm_tuple(v)
            return out


        self.base_cache = _norm_map(base_cc)
        self.bg_cache_disk = _norm_map(bg_cc)
        self.class_cache_disk = _norm_map(class_cc)

        self.bg_appended_runtime: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
        self.class_appended_runtime: Dict[Tuple[str, ...], Tuple[str, ...]] = {}

        self.payload = payload
        self.enable_copy_write = bool(enable_copy_write)
        self.copy_out_path = Path(copy_out_path) if copy_out_path else _copy_path(self.src_path)
        self._ic_eval_ctx = {"in_synth": False, "total_n": None}
        # (debug) quick summary
        self._dbg = None



        # derive FD rules from base closure_cache only
        rules_set: Set[Tuple[Tuple[str, ...], str]] = set()
        for lhs, clo in self.base_cache.items():
            lhs_set = set(lhs)
            for a in clo:
                if a not in lhs_set:
                    rules_set.add((tuple(lhs), str(a)))

        # # --- derive FD rules from base closure_cache only (same as before) ---
        # rules_set: Set[Tuple[Tuple[str, ...], str]] = set()
        # for lhs, clo in self.base_cache.items():
        #     lhs_set = set(lhs)
        #     for a in clo:
        #         if a not in lhs_set:
        #             rules_set.add((tuple(lhs), str(a)))

        self.rules: List[Tuple[Tuple[str, ...], str]] = sorted(
            rules_set, key=lambda t: (len(t[0]), t[0], t[1])
        )

        # --- NEW: horn-forward indexing ---
        # rule ids 0..R-1
        self._rule_lhs: List[Tuple[str, ...]] = []
        self._rule_rhs: List[str] = []
        self._watch: Dict[str, List[int]] = defaultdict(list)  # attr -> [rule_id,...]

        for rid, (lhs, rhs) in enumerate(self.rules):
            if len(lhs) > 2:
                continue
            self._rule_lhs.append(tuple(lhs))
            self._rule_rhs.append(rhs)
            for a in lhs:
                self._watch[a].append(rid)

        self._rule_len = np.asarray([len(lhs) for lhs in self._rule_lhs], dtype=np.uint8)

    def lookup_bg(self, S_key: Tuple[str, ...]) -> Optional[Tuple[str, ...]]:
        S_key = tuple(a.lower() if isinstance(a, str) else a for a in S_key)
        hit = self.base_cache.get(S_key)
        if hit is not None:
            return hit
        hit = self.bg_cache_disk.get(S_key)
        if hit is not None:
            return hit
        return self.bg_appended_runtime.get(S_key)

    def lookup_class(self, canon_key: Tuple[str, ...]) -> Optional[Tuple[str, ...]]:
        canon_key = tuple(a.lower() if isinstance(a, str) else a for a in canon_key)
        hit = self.class_cache_disk.get(canon_key)
        if hit is not None:
            return hit
        return self.class_appended_runtime.get(canon_key)

    def closure_and_provenance(
        self,
        S_key: Tuple[int, ...],
        collect_prov: bool = False,
    ) -> Tuple[Tuple[int, ...], Optional[Dict[str, List[Tuple[str, ...]]]]]:
        """
        Horn-forward chaining (queue).
        Equivalent to repeated Armstrong scan, but faster when |lhs| can be >=3 or
        when rule set is large.
        """
        # optional debug hook (very light)
        # if hasattr(self, "_dbg") and callable(self._dbg):
        #     self._dbg(f"[closure] input |S|={len(S_key)} S={list(S_key)[:10]}{'...' if len(S_key)>10 else ''}")
        
        # current: Set[str] = set(S_key)
        current: Set[str] = {
            a.lower() if isinstance(a, str) else a
            for a in S_key
        }
        prov = defaultdict(list) if collect_prov else None

        R = len(self._rule_rhs)
        if R == 0:
            return tuple(sorted(current)), prov

        remaining = self._rule_len.copy()  # uint8 array
        q = deque()

        # lhs 원소가 current에 있으면 remaining 감소
        # (lhs 길이 최대 2라서 아래처럼 단순 루프로도 빠름)
        for rid, lhs in enumerate(self._rule_lhs):
            rem = remaining[rid]
            if rem == 0:
                continue
            for a in lhs:
                if a in current:
                    rem -= 1
            remaining[rid] = rem
            if rem == 0:
                rhs = self._rule_rhs[rid]
                if rhs not in current:
                    current.add(rhs)
                    if collect_prov:
                        prov[rhs].append(lhs)
                    q.append(rhs)


        # BFS-style propagation: when a becomes true, it reduces remaining for rules that watch a
        while q:
            a = q.popleft()
            for rid in self._watch.get(a, []):
                if remaining[rid] <= 0:
                    continue
                remaining[rid] -= 1
                if remaining[rid] == 0:
                    rhs = self._rule_rhs[rid]
                    if rhs in current:
                        continue
                    current.add(rhs)
                    lhs = self._rule_lhs[rid]
                    if collect_prov:
                        prov[rhs].append(lhs)
                    q.append(rhs)

        out = tuple(sorted(current))
        # if hasattr(self, "_dbg") and callable(self._dbg):
        #     self._dbg(f"[closure] output |clo|={len(out)} added={len(set(out)-set(S_key))}")

        if self._dbg is not None:
            added = sorted(list(set(out) - set(S_key)))

            # reasons from prov dict
            rule_reasons = []
            for rhs in added[:8]:
                lhs_list = (prov or {}).get(rhs, [])
                if lhs_list:
                    lhs = list(lhs_list[0])
                    rule_reasons.append(f"{lhs} -> {rhs}")
                else:
                    rule_reasons.append(f"(unknown_lhs) -> {rhs}")

            before = f"|S|={len(S_key)} S={list(S_key)[:12]}{'...' if len(S_key)>12 else ''}"
            because = " ; ".join(rule_reasons) if rule_reasons else "(no rule applied)"
            after = f"|clo|={len(out)} added={len(added)} clo={list(out)[:12]}{'...' if len(out)>12 else ''}"

            _dbg3(self._dbg, before=before, because=because, after=after, tag="[CLOSURE]")

        return out, (dict(prov) if collect_prov else None)


    def remember_bg(self, raw_key: Tuple[str, ...], clo: Tuple[str, ...]) -> None:
        if not self.enable_copy_write:
            return
        if raw_key in self.base_cache or raw_key in self.bg_cache_disk:
            return
        self.bg_appended_runtime.setdefault(raw_key, clo)

    def remember_class(self, canon_key: Tuple[str, ...]) -> None:
        if not self.enable_copy_write:
            return
        if canon_key in self.class_cache_disk:
            return
        self.class_appended_runtime.setdefault(canon_key, canon_key)

    def flush_copy(self) -> Optional[Path]:
        if not self.enable_copy_write:
            return None

        fd = self.payload.get("fd", {})
        if not isinstance(fd, dict):
            fd = {}
            self.payload["fd"] = fd

        fd.setdefault("closure_cache_bg", {})
        fd.setdefault("coalition_class_cache", {})
        if not isinstance(fd["closure_cache_bg"], dict):
            fd["closure_cache_bg"] = {}
        if not isinstance(fd["coalition_class_cache"], dict):
            fd["coalition_class_cache"] = {}

        for k, v in self.bg_appended_runtime.items():
            fd["closure_cache_bg"][tuple(k)] = tuple(v)
        for k, v in self.class_appended_runtime.items():
            fd["coalition_class_cache"][tuple(k)] = tuple(v)

        fd.setdefault("meta", {})
        if isinstance(fd["meta"], dict):
            fd["meta"]["src_cache"] = str(self.src_path)
            fd["meta"]["bg_appended_n_this_run"] = len(self.bg_appended_runtime)
            fd["meta"]["coalition_appended_n_this_run"] = len(self.class_appended_runtime)

        self.copy_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.copy_out_path, "wb") as f:
            pickle.dump(self.payload, f)
        return self.copy_out_path


# =========================================================
# RelShapKernelExplainer
#   - mode-bg: apply closure to mask
#   - coalition-canon
#   - coalition-budget
# =========================================================
class RelShapKernelExplainer(KernelExplainer):
    def __init__(
        self,
        model,
        data,
        *,
        feature_names: List[str],
        mode_bg: bool,
        mode_coal_canon: bool,
        mode_coal_budget: bool,
        budget_max_skips: int,
        closure_cache: Optional[ClosureCache],
        provenance_mode: Optional[str],  # None or one of choices
        provenance_threshold: int,
        provenance_index: Optional[ProvenanceIndex],
        prov_strength: str = "strong",
        debug: int = 0,
        debug_lim: int = 0,
        debug_coal: bool = False,
        debug_coal_limit: int = 5000,
        seed: int,
        n_jobs: int = -1, # -1 uses all threads; 1 is single-threaded
        **kwargs,
    ):
        super().__init__(model, data, **kwargs)
        self.feature_names = list(feature_names)
        self.col2i = {c: i for i, c in enumerate(self.feature_names)}

        self.debug_coal = bool(debug_coal)
        self.debug_coal_limit = int(debug_coal_limit)
        self.seed = int(seed)
        self.n_jobs = n_jobs

        self._coal_keys_by_mode = defaultdict(list)

        self._coal_size_hist_by_mode = defaultdict(Counter)


        self.mode_bg = bool(mode_bg)
        self.mode_coal_canon = bool(mode_coal_canon)
        self.mode_coal_budget = bool(mode_coal_budget)
        self.budget_max_skips = int(budget_max_skips)
        self.closure_cache = closure_cache

        self.debug = int(debug)
        self._dbg_lim = int(debug_lim)
        self._dbg_every = 1
        self._dbg = _dbg_maker(bool(self.debug), self._dbg_lim, self._dbg_every, prefix="[dbg]")

        self._ic_repair_fn = None
        self._ic_feature_cols = list(feature_names)

        self.provenance_mode = provenance_mode  # None / "bg_only" / "bg-coalition-canon" / "bg-coalition-budget"
        self.mode_provenance = (provenance_mode is not None)
        self.prov_threshold = int(provenance_threshold)

        # key: mask_key(before provenance), val: packed mask bytes(after provenance)
        self._prov_cache: Dict[bytes, bytes] = {}
        self._x_canon = None  # optional (for step 3)
        self._x_canon_key: Optional[bytes] = None   # x.tobytes() of last seen x
        self._x_canon_val: Optional[List[Any]] = None  # cached x_canon_full list

        self.prov_index = provenance_index
        self.prov_strength = str(prov_strength).strip().lower()
        if self.prov_strength not in ("strong", "weak"):
            raise ValueError(f"prov_strength must be 'strong' or 'weak', got {self.prov_strength}")

        if (self.mode_coal_canon or self.mode_coal_budget) and not self.mode_bg:
            raise ValueError("Coalition modes require mode_bg=True.")
        if self.mode_coal_canon and self.mode_coal_budget:
            raise ValueError("Choose only one: canon OR budget.")


        # stats
        self._canon_count: Dict[bytes, int] = defaultdict(int)  # keyed by canon mask key
        self._budget_skipped = 0

        # used for canon sampling
        self._seen_canon_mask_keys: Set[bytes] = set()

        self._eval_cache: Dict[bytes, np.ndarray] = {}
        self._eval_cache_hits = 0
        self._eval_cache_misses = 0

        # desired #samples to actually ADD in budget mode (not attempts)
        self._budget_desired_nsamples: Optional[int] = None

        self._feat_arr = np.asarray(self.feature_names, dtype=object)
        self._col_to_i_cache = {}

    # def _dbg(self, msg: str):
    #     if self.debug and self._dbg_n < self._dbg_lim:
    #         print(msg)
    #         self._dbg_n += 1

    def _rand_mask_like(self, rng: np.random.RandomState) -> np.ndarray:
        """Random 0/1 mask over M features; avoids all-zero and all-one."""
        M = len(self.feature_names)
        while True:
            m = (rng.rand(M) < 0.5).astype(np.int8)
            s = int(m.sum())
            if 0 < s < M:
                return m

    def explain(self, *args, **kwargs):
        # IMPORTANT: reset per instance, not per shap_values call
        self._prov_cache = {}
        self._x_canon = None
        self._x_canon_key = None
        self._x_canon_val = None

        if self.mode_coal_budget:
            self._seen_canon_mask_keys = set()
            self._budget_skipped = 0

            # --- Option A: keep trying until nsamplesAdded reaches user's nsamples ---
            ns = kwargs.get("nsamples", None)
            if ns is None:
                raise ValueError("[mode-coalition-budget] requires an explicit nsamples int (not None).")

            # shap sometimes uses "auto"; we enforce int for safety
            if not isinstance(ns, int):
                raise ValueError(f"[mode-coalition-budget] nsamples must be int, got {type(ns)}: {ns}")

            self._budget_desired_nsamples = int(ns)

            # Inflate attempts so that even if we skip up to budget_max_skips,
            # we can still ADD desired_nsamples.
            M = len(self.feature_names)
            max_masks = (1 << M) - 2              # 2^M - 2
            attempts = int(ns) + int(self.budget_max_skips)
            kwargs["nsamples"] = min(attempts, max_masks - 1)  # 전수조사 트리거 피하려고 -1

        try:
            return super().explain(*args, **kwargs)
        finally:
            if self.mode_coal_budget:
                self._budget_desired_nsamples = None

    def shap_values(self, X, *args, **kwargs):
        # remove the reset here (or leave it, but per-instance reset is what matters)
        return super().shap_values(X, *args, **kwargs)

    def shap_values_mc(self, X, *args, **kwargs):
        """Monte Carlo (coalition-sampling) Shapley estimator (KernelSHAP drop-in).

        Keeps existing RelShap behaviors (FD closure, provenance expansion,
        IC repair, coalition canon/budget debug stats) but replaces the
        KernelSHAP weighted-regression solver with a coalition-sampling MC estimator.

        Interpretation of `nsamples`: number of coalition samples per row.
        Each sample draws:
        - pivot feature i ~ Uniform({0..M-1})
        - coalition size k ~ Uniform({0..M-1}) over predecessors of i
        - S ⊆ N\{i} uniformly among subsets of size k
        then uses delta = v(SU{i}) - v(S) (with FD closure + provenance expansion),
        and distributes delta equally over newly-activated features (closure bundles).
        """

        # --- parse nsamples like SHAP, but require an int for reproducibility ---
        ns = kwargs.get("nsamples", None)
        if ns is None:
            ns = 100
        if isinstance(ns, str):
            if ns.strip().lower() == "auto":
                ns = 2 * max(1, len(self.feature_names))
            else:
                raise ValueError(f"nsamples must be int or 'auto', got: {ns!r}")
        ns = int(ns)
        if ns <= 0:
            raise ValueError(f"nsamples must be positive, got {ns}")

        # normalize X to a 2D numpy array with correct column order
        if isinstance(X, pd.DataFrame):
            X_df = X[self.feature_names]
            X_np = X_df.to_numpy()
        else:
            X_np = np.asarray(X)
            if X_np.ndim == 1:
                X_np = X_np.reshape(1, -1)

        M = len(self.feature_names)
        if X_np.shape[1] != M:
            raise ValueError(
                f"X must have M={M} features (in explainer.feature_names order); got shape {X_np.shape}"
            )

        # background (as provided at init)
        bg_obj = getattr(getattr(self, "data", None), "data", None)
        if bg_obj is None:
            bg_obj = getattr(self, "data", None)
        bg_np = np.asarray(bg_obj)
        if bg_np.ndim == 1:
            bg_np = bg_np.reshape(1, -1)
        if bg_np.shape[1] != M:
            raise ValueError(f"Background must have M={M} features; got shape {bg_np.shape}")

        # optional IC repair
        repair_fn = getattr(self, "_ic_repair_fn", None)
        feature_cols_for_ic = getattr(self, "_ic_feature_cols", self.feature_names)

        # deterministic RNG; vary per row to avoid identical streams
        base_rng = np.random.RandomState(self.seed)

        # --- resolve a callable model function (KernelExplainer wraps models) ---
        model_fn = None

        # 1) if self.model is directly callable
        if callable(getattr(self, "model", None)):
            model_fn = self.model

        # 2) shap KernelExplainer often stores a Model wrapper with .f
        elif callable(getattr(getattr(self, "model", None), "f", None)):
            model_fn = self.model.f

        # 3) sometimes wrapper has .predict
        elif callable(getattr(getattr(self, "model", None), "predict", None)):
            model_fn = self.model.predict

        # 4) (rare) wrapper has .model which might be callable or have predict
        else:
            inner = getattr(getattr(self, "model", None), "model", None)
            if callable(inner):
                model_fn = inner
            elif callable(getattr(inner, "predict", None)):
                model_fn = inner.predict

        if model_fn is None:
            raise TypeError(
                "Could not resolve a callable model function from explainer.model. "
                "Tried: model, model.f, model.predict, model.model, model.model.predict"
            )


        def _canonize_mask_from_cols(on_cols: Tuple[str, ...], x_full: np.ndarray) -> np.ndarray:
            """Return canonical 0/1 mask over full M features.
            - If mode_bg: apply FD closure via closure_cache (and provenance reasons for debug).
            - If mode_provenance: apply provenance mask expansion (independent of mode_bg).
            """
            raw_key = tuple(sorted(set(on_cols)))

            # ---- 1) raw mask always ----
            m = np.zeros(M, dtype=np.int8)
            for c in raw_key:
                j = self.col2i.get(c)
                if j is not None:
                    m[j] = 1

            clo = raw_key
            prov = None

            # ---- 2) FD closure only when mode_bg ----
            if self.mode_bg:
                if self.closure_cache is None:
                    raise ValueError("--mode-bg requires --constraints-cache")

                collect_prov = bool(self.debug)
                clo, prov = self.closure_cache.closure_and_provenance(raw_key, collect_prov=collect_prov)

                # rebuild mask from closure
                m[:] = 0
                for col in clo:
                    j = self._col_to_i_cache.get(col)
                    if j is None:
                        j = self.col2i.get(col)
                        if j is None:
                            continue
                        self._col_to_i_cache[col] = j
                    m[j] = 1

            # ---- 3) provenance expansion independent of mode_bg ----
            if self.mode_provenance and self.prov_index is not None:
                m = self._apply_provenance_mask_expansion(m, x_full)

            # ---- debug (keep yours; but prov might be None when mode_bg off) ----
            if self.debug:
                # RAW mask summary based on raw_key
                m_raw = np.zeros(M, dtype=np.int8)
                for c in raw_key:
                    j = self.col2i.get(c)
                    if j is not None:
                        m_raw[j] = 1

                before_fd = _summ_mask(self.feature_names, m_raw)
                after_all = _summ_mask(self.feature_names, m)

                self._dbg(f"[addsample] raw {before_fd} raw_key={list(raw_key)}")

                if self.mode_bg:
                    self._dbg(
                        f"[addsample] clo added_n={len(set(clo)-set(raw_key))} "
                        f"clo_head={list(clo)[:12]}{'...' if len(clo)>12 else ''}"
                    )

                if after_all != before_fd and self.mode_bg:
                    added = sorted(set(clo) - set(raw_key))
                    if added:
                        rule_reasons = []
                        for rhs in added[:8]:
                            lhs_list = (prov or {}).get(rhs, [])
                            if lhs_list:
                                lhs = list(lhs_list[0])
                                rule_reasons.append(f"{lhs} -> {rhs}")
                            else:
                                rule_reasons.append(f"(unknown_lhs) -> {rhs}")
                        because = "FD: " + " ; ".join(rule_reasons)
                        if len(added) > 8:
                            because += " ..."
                        _dbg3(self._dbg, before=before_fd, because=because, after=after_all, tag="[FD]")

            return m


        def _model_eval_mean_over_bg(mask: np.ndarray, x_full: np.ndarray) -> float:
            """v(S): mean_{z in BG} f(z with S from x_full), with optional IC repair.
            canonical mask key.
            """
            key = _mask_key(mask)

            if self.mode_coal_canon:
                if key in self._eval_cache:
                    self._eval_cache_hits += 1
                    return float(self._eval_cache[key])
                self._eval_cache_misses += 1

            on_idx = np.flatnonzero(mask)
            batch = np.array(bg_np, copy=True)
            if on_idx.size:
                batch[:, on_idx] = x_full[on_idx]

            if repair_fn is not None:
                fixed_cols = set(feature_cols_for_ic[j] for j in on_idx.tolist())
                dfb = pd.DataFrame(batch, columns=feature_cols_for_ic)
                dfb = repair_fn(dfb, fixed_cols=fixed_cols)
                batch = dfb.to_numpy()

            try:
                pred = model_fn(pd.DataFrame(batch, columns=self.feature_names))
            except Exception:
                pred = model_fn(batch)


            pred = np.asarray(pred).reshape(-1)
            v = float(np.mean(pred))

            if self.mode_coal_canon:
                self._eval_cache[key] = v
            return v

        feat_idx_all = np.arange(M, dtype=int)

        def _iter_row(row_i: int, seed: int):
            # reset per-instance caches (matches original intent)
            self._prov_cache = {}
            self._x_canon = None
            if self.mode_coal_canon:
                self._eval_cache = {}
                self._eval_cache_hits = 0
                self._eval_cache_misses = 0
            if self.mode_coal_budget:
                self._seen_canon_mask_keys = set()
                self._budget_skipped = 0

            x_full = np.asarray(X_np[row_i], dtype=bg_np.dtype, order="C")
            rng = np.random.RandomState(seed)

            phi = np.zeros(M, dtype=float)

            desired_add = ns
            max_skips = int(self.budget_max_skips) if self.mode_coal_budget else 0
            max_attempts = desired_add + max_skips if self.mode_coal_budget else desired_add

            added = 0
            attempts = 0

            while (added < desired_add) and (attempts < max_attempts):
                attempts += 1

                # 1) pivot i ~ Uniform({0..M-1})
                pivot = int(rng.randint(0, M))

                # 2) k ~ Uniform({0..M-1}) among remaining M-1 features
                k = int(rng.randint(0, M))  # 0..M-1

                # 3) sample S of size k from N\{pivot}
                others = np.delete(feat_idx_all, pivot)
                if k > 0:
                    S_idx = rng.choice(others, size=k, replace=False)
                else:
                    S_idx = np.empty((0,), dtype=int)

                on_cols = tuple(self.feature_names[j] for j in S_idx.tolist())
                mask_S = _canonize_mask_from_cols(on_cols, x_full)
                self._coal_record("mc", mask_S)

                on_cols_plus = on_cols + (self.feature_names[pivot],)
                mask_Si = _canonize_mask_from_cols(on_cols_plus, x_full)
                # self._coal_record("mc_SUi", mask_Si)

                key_S  = _mask_key(mask_S)
                key_Si = _mask_key(mask_Si)

                newly_on = np.flatnonzero((mask_Si == 1) & (mask_S == 0))
                if newly_on.size == 0:
                    continue   # do NOT count toward added

                if self.mode_coal_budget:
                    pair_key = (key_S, key_Si)
                    if pair_key in self._seen_canon_mask_keys:
                        self._budget_skipped += 1
                        continue
                    self._seen_canon_mask_keys.add(pair_key)

                v_S  = _model_eval_mean_over_bg(mask_S, x_full)
                v_Si = _model_eval_mean_over_bg(mask_Si, x_full)

                delta = v_Si - v_S
                if newly_on.size:
                    phi[newly_on] += delta / float(newly_on.size)

                added += 1

            denom = float(max(1, added))
            row_phi = phi / denom

            if self.debug and self.mode_coal_canon:
                self._dbg(
                    f"[mc-coal] row={row_i} canon hits={self._eval_cache_hits} misses={self._eval_cache_misses} "
                    f"unique_masks={len(self._eval_cache)} added={added}/{ns}"
                )
            if self.debug and self.mode_coal_budget:
                self._dbg(
                    f"[mc-coal] row={row_i} budget skipped={self._budget_skipped} "
                    f"seen={len(self._seen_canon_mask_keys)} added={added}/{ns}"
                )
            return row_phi
            # end of _iter_row

        out = np.zeros((X_np.shape[0], M), dtype=float)

        # pre-draw per-row seeds before any forking
        seeds = [int(base_rng.randint(0, 2**31 - 1)) for _ in range(X_np.shape[0])]

        n_jobs = getattr(self, "n_jobs", 1) or 1
        use_parallel = (n_jobs != 1) and (X_np.shape[0] > 1)

        if use_parallel:
            # fork-based pool: workers inherit the closure (incl. model_fn) via os.fork()
            # without any pickling. Only the lightweight (row_i, seed) args are sent per task.
            global _MC_ROW_FN
            _MC_ROW_FN = _iter_row
            ctx = multiprocessing.get_context("fork")
            actual_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
            with ctx.Pool(actual_jobs) as pool:
                results = list(tqdm(
                    pool.imap(_fork_mc_worker, [(i, seeds[i]) for i in range(X_np.shape[0])]),
                    total=X_np.shape[0], leave=True,
                ))
        else:
            pbar = tqdm(total=X_np.shape[0], leave=True)
            results = []
            for i in range(X_np.shape[0]):
                results.append(_iter_row(i, seeds[i]))
                pbar.update(1)
            pbar.close()

        for row_i, row_phi in enumerate(results):
            out[row_i, :] = row_phi

        return out

    def shap_values_leverage(self, X, *args, **kwargs):
        """LeverageSHAP WLS + BG-distribution expectation, RelShap-compatible knobs.

        What this implements (matches your requirements):

        - ALWAYS uses BG expectation:
            v(S) = mean_{z in BG} f( z with S from x )
        (So even when NO modes are on, it is "LeverageSHAP + bg dist expectation".)

        - mode semantics (ONLY when those modes are enabled):
        1) --mode-bg:
            - do NOT reduce by coalition equivalence class
            - ONLY change coalition mask via FD closure / provenance expansion hooks
            - top-up sampling if canonicalization makes unusable edge masks (|S| in {0,M})

        2) --mode-bg + --mode-coalition-canon:
            - same canonicalization
            - reduce by canonical key (collapse); sum kernel weights per key
            - evaluate v(S) once per key (plus eval cache)

        3) --mode-bg + --mode-coalition-budget:
            - same canonicalization
            - enforce UNIQUE canonical keys; top-up until exactly nsamples unique usable keys

        - Other modes (provenance expansion, IC repair) are applied during v(S) evaluation
        (and/or during canonicalization, as in your MC implementation).
        """
        # ----------------------------
        # Parse nsamples
        # ----------------------------
        ns = kwargs.get("nsamples", None)
        if ns is None:
            ns = 100
        if isinstance(ns, str):
            if ns.strip().lower() == "auto":
                ns = 2 * max(1, len(self.feature_names))
            else:
                raise ValueError(f"nsamples must be int or 'auto', got: {ns!r}")
        ns = int(ns)
        if ns < 6:
            ns = 6  # LeverageSHAP code: enforce >= 6

        # ----------------------------
        # Normalize X
        # ----------------------------
        if isinstance(X, pd.DataFrame):
            X_df = X[self.feature_names]
            X_np = X_df.to_numpy()
        else:
            X_np = np.asarray(X)
            if X_np.ndim == 1:
                X_np = X_np.reshape(1, -1)

        M = len(self.feature_names)
        if X_np.shape[1] != M:
            raise ValueError(f"X must have M={M} features; got shape {X_np.shape}")

        # ----------------------------
        # Background data (distribution support)
        # ----------------------------
        bg_obj = getattr(getattr(self, "data", None), "data", None)
        if bg_obj is None:
            bg_obj = getattr(self, "data", None)
        bg_np = np.asarray(bg_obj)
        if bg_np.ndim == 1:
            bg_np = bg_np.reshape(1, -1)
        if bg_np.shape[1] != M:
            raise ValueError(f"background must have M={M} features; got shape {bg_np.shape}")

        # ----------------------------
        # Resolve model callable like KernelExplainer does
        # ----------------------------
        if callable(getattr(self, "model", None)):
            model_fn = self.model
        elif callable(getattr(getattr(self, "model", None), "f", None)):
            model_fn = self.model.f
        elif callable(getattr(getattr(self, "model", None), "predict", None)):
            model_fn = self.model.predict
        else:
            raise ValueError("Cannot resolve a callable model function from explainer.model")

        def _pred_batch(X_batch_np: np.ndarray) -> np.ndarray:
            """Predict 1D array for a batch."""
            X_batch_np = np.asarray(X_batch_np)
            try:
                y = model_fn(pd.DataFrame(X_batch_np, columns=self.feature_names))
            except Exception:
                y = model_fn(X_batch_np)
            y = np.asarray(y)
            if y.ndim > 1:
                y = y[:, 0]
            return y.reshape(-1)

        # ----------------------------
        # IC repair hook (same pattern as your MC code)
        # ----------------------------
        repair_fn = getattr(self, "_ic_repair_fn", None)
        feature_cols_for_ic = getattr(self, "_ic_feature_cols", self.feature_names)

        # ----------------------------
        # Canonicalize mask from selected columns (FD closure + provenance expansion)
        # ----------------------------
        def _canonize_mask_from_cols(on_cols: tuple, x_full: np.ndarray) -> np.ndarray:

            m0 = np.zeros(M, dtype=np.int8)
            for c in on_cols:
                j = self.col2i.get(c, None)
                if j is not None:
                    m0[j] = 1

            # mode_bg: FD closure (mask changes)
            if self.mode_bg and (self.closure_cache is not None):
                before_cols = [self.feature_names[j] for j in np.flatnonzero(m0)]
                after_cols, _prov = self.closure_cache.closure_and_provenance(
                    tuple(sorted(before_cols)),
                    collect_prov=bool(getattr(self, "debug", False)),
                )
                m1 = np.zeros(M, dtype=np.int8)
                for c in after_cols:
                    j = self.col2i.get(c, None)
                    if j is not None:
                        m1[j] = 1
            else:
                m1 = m0

            # provenance expansion (RelShap behavior)
            if getattr(self, "mode_provenance", False):
                m2 = self._apply_provenance_mask_expansion(m1, x=np.asarray(x_full).reshape(-1))
            else:
                m2 = m1

            # ---------------- DEBUG ONLY (kernel-style FD print) ----------------
            if getattr(self, "debug", False) and self.mode_bg and (self.closure_cache is not None):

                # raw (before FD)
                before_fd = _summ_mask(self.feature_names, m0)
                self._dbg(f"[addsample] raw {before_fd} raw_key={list(on_cols)}")

                # after FD (m1)
                after_fd = _summ_mask(self.feature_names, m1)

                if after_fd != before_fd:
                    before_cols = [self.feature_names[j] for j in np.flatnonzero(m0)]
                    after_cols = [self.feature_names[j] for j in np.flatnonzero(m1)]

                    added = sorted(set(after_cols) - set(before_cols))
                    if added:
                        # try to re-get provenance only for explanation
                        _, prov = self.closure_cache.closure_and_provenance(
                            tuple(sorted(before_cols)), collect_prov=True
                        )

                        rule_reasons = []
                        for rhs in added[:8]:
                            lhs_list = (prov or {}).get(rhs, [])
                            if lhs_list:
                                lhs = list(lhs_list[0])
                                rule_reasons.append(f"{lhs} -> {rhs}")
                            else:
                                rule_reasons.append(f"(unknown_lhs) -> {rhs}")

                        because = "FD: " + " ; ".join(rule_reasons)
                        if len(added) > 8:
                            because += " ..."

                        if getattr(self, "mode_coal_canon", False):
                            fd_tag = "[FD|COAL-CANON]"
                        elif getattr(self, "mode_coal_budget", False):
                            fd_tag = "[FD|COAL-BUDGET]"
                        else:
                            fd_tag = "[FD|BG-ONLY]"

                        _dbg3(self._dbg, before=before_fd, because=because, after=after_fd, tag=fd_tag)

            return np.asarray(m2, dtype=np.int8)

        def _mask_key(m: np.ndarray) -> bytes:
            mb = np.asarray(m, dtype=np.uint8).reshape(-1)
            return np.packbits(mb, bitorder="little").tobytes()

        # ----------------------------
        # CoalitionSampler (keep your in-function version for parity)
        # ----------------------------
        class _CoalitionSampler:
            def __init__(self, n_players: int, sampling_weights: np.ndarray, *,
                        pairing_trick: bool = True,
                        seed: int,
                        sample_with_replacement: bool = False,):

                self._n_players = n_players

                if len(sampling_weights) == n_players + 1:
                    sampling_weights = sampling_weights[1:-1]
                elif len(sampling_weights) == n_players:
                    sampling_weights = sampling_weights[1:]
                elif len(sampling_weights) != n_players - 1:
                    raise ValueError(f"sampling_weights should be of length n_players-1, but got length {len(sampling_weights)}.")

                self._distribution = sampling_weights / np.min(sampling_weights)
                self._distribution = np.concatenate(([0.0], self._distribution, [0.0]))

                self._pairing_trick = pairing_trick
                self.seed = seed
                self._rng = np.random.default_rng(seed=seed)
                self._sample_with_replacement = sample_with_replacement
                self._sampled = False

            def _symmetric_round_even(self, x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, float); n = x.size
                tgt = int(np.round(x.sum()/2)*2)
                out = np.floor(x).astype(int)
                rem = tgt - out.sum()
                frac = x - np.floor(x)

                pairs = [(i, n-1-i, frac[i]+frac[n-1-i]) for i in range(n//2)]
                pairs.sort(key=lambda t: t[2], reverse=True)
                for i, j, _ in pairs:
                    if rem < 2: break
                    out[i] += 1; out[j] += 1; rem -= 2
                if n % 2 == 1 and rem == 1:
                    out[n//2] += 1; rem -= 1
                return out

            def _sampling_probs(self, sizes: np.ndarray) -> np.ndarray:
                return np.minimum(
                    self._constant * self._distribution[sizes] / _binom(self._n_players, sizes), 1
                )

            def _get_sampling_probs(self, budget: int):
                n = self._n_players
                sizes = np.arange(1, n)
                comb_counts = _binom(n, sizes).astype(float)
                weights = self._distribution[sizes].astype(float)

                target_total = float(np.clip(budget, 0, np.sum(comb_counts)))
                if target_total == 0.0:
                    self._constant = 0.0
                    return

                saturation_thresholds = comb_counts / weights
                order = np.argsort(saturation_thresholds)
                comb_counts_sorted = comb_counts[order]
                weights_sorted = weights[order]
                thresholds_sorted = saturation_thresholds[order]

                saturated_prefix = np.concatenate(([0.0], np.cumsum(comb_counts_sorted[:-1])))
                weights_prefix = np.concatenate(([0.0], np.cumsum(weights_sorted[:-1])))
                remaining_weight = np.sum(weights_sorted) - weights_prefix

                expected_at_threshold = saturated_prefix + thresholds_sorted * remaining_weight
                segment_idx = np.searchsorted(expected_at_threshold, target_total, side="left")

                if segment_idx >= len(thresholds_sorted):
                    scale = float(thresholds_sorted[-1])
                else:
                    denom = remaining_weight[segment_idx]
                    scale = thresholds_sorted[segment_idx] if denom == 0 else \
                            min((target_total - saturated_prefix[segment_idx]) / denom,
                                thresholds_sorted[segment_idx])
                self._constant = float(scale)

            def _index_th_combination(self, pool, size: int, index: int):
                n = len(pool)
                k = size
                combo = []
                for i in range(n):
                    if k == 0:
                        break
                    if n - i == k:
                        combo.extend(pool[i:i+k])
                        k = 0
                        break
                    c = math.comb(n - i - 1, k - 1)
                    if index < c:
                        combo.append(pool[i]); k -= 1
                    else:
                        index -= c
                return tuple(combo)

            def _combination_generator(self, n: int, s: int, num_samples: int):
                num_combos = math.comb(n, s)
                try:
                    assert not self._sample_with_replacement
                    indices = self._rng.choice(num_combos, num_samples, replace=False)
                    for i in indices:
                        yield self._index_th_combination(range(n), s, int(i))
                except (OverflowError, AssertionError):
                    for _ in range(num_samples):
                        yield self._rng.choice(n, s, replace=False)

            def _add_one_sample(self, indices):
                self._sampled_coalitions_matrix[self._coalition_idx, list(indices)] = 1
                self._coalition_idx += 1

            def sample(self, budget: int):
                assert budget >= 2
                budget = min(budget, 2**self._n_players)
                budget += budget % 2

                self._get_sampling_probs(budget - 2)
                sizes = np.arange(1, self._n_players)

                samples_per_size = self._symmetric_round_even(
                    self._sampling_probs(sizes) * _binom(self._n_players, sizes)
                )
                sampling_probs = samples_per_size / _binom(self._n_players, sizes)

                self._sampled_coalitions_matrix = np.zeros((budget, self._n_players), dtype=np.int8)
                self._coalition_idx = 0

                # empty and full
                self._add_one_sample([])
                self._add_one_sample(range(self._n_players))

                for idx, size in enumerate(sizes):
                    if idx >= self._n_players // 2 and self._pairing_trick:
                        break
                    if self._pairing_trick and size == self._n_players // 2 and self._n_players % 2 == 0:
                        combo_gen = self._combination_generator(self._n_players - 1, size - 1, int(samples_per_size[idx] // 2))
                        for indices in combo_gen:
                            indices = list(indices)
                            self._add_one_sample(indices + [self._n_players - 1])
                            self._add_one_sample(list(set(range(self._n_players - 1)) - set(indices)))
                    else:
                        combo_gen = self._combination_generator(self._n_players, int(size), int(samples_per_size[idx]))
                        for indices in combo_gen:
                            indices = list(indices)
                            self._add_one_sample(indices)
                            if self._pairing_trick:
                                self._add_one_sample(list(set(range(self._n_players)) - set(indices)))

                coalition_sizes = np.sum(self._sampled_coalitions_matrix, axis=1).astype(int)

                self._sampled_coalitions_probability = np.ones(self._sampled_coalitions_matrix.shape[0], dtype=float)
                filter_idx = (coalition_sizes > 0) & (coalition_sizes < self._n_players)
                self._sampled_coalitions_probability[filter_idx] = sampling_probs[coalition_sizes[filter_idx] - 1]
                self._sampled = True

        # ----------------------------
        # v(S) = mean over BG of f(z with S from x_full)
        # ----------------------------
        def _v_mean_bg(mask: np.ndarray, x_full: np.ndarray) -> float:
            key = _mask_key(mask)

            if getattr(self, "mode_coal_canon", False):
                if key in self._eval_cache:
                    self._eval_cache_hits += 1
                    return float(self._eval_cache[key])
                self._eval_cache_misses += 1

            on_idx = np.flatnonzero(mask)
            batch = np.array(bg_np, copy=True)
            if on_idx.size:
                batch[:, on_idx] = x_full[on_idx]

            if repair_fn is not None:
                fixed_cols = set(feature_cols_for_ic[j] for j in on_idx.tolist())
                dfb = pd.DataFrame(batch, columns=feature_cols_for_ic)
                dfb = repair_fn(dfb, fixed_cols=fixed_cols)
                batch = dfb.to_numpy()

            v = float(np.mean(_pred_batch(batch)))

            if getattr(self, "mode_coal_canon", False):
                self._eval_cache[key] = v
            return v

        # ----------------------------
        # LeverageSHAP WLS per-row
        # ----------------------------
        out = np.zeros((X_np.shape[0], M), dtype=float)
        pbar = tqdm(total=X_np.shape[0]) # desc="LeverageSHAP (rows)"

        # sampling weights (official: ones)
        sampling_weights = np.ones(M - 1, dtype=float)

        def _lev_w(prob: float, s: int) -> float:
            reg = 1.0 / (_binom(M, s) * s * (M - s))
            return float(reg / float(prob))

        def _iter_row(row_i: int):
            self._prov_cache = {}
            self._x_canon = None
            if getattr(self, "mode_coal_canon", False):
                self._eval_cache = {}
                self._eval_cache_hits = 0
                self._eval_cache_misses = 0
            if getattr(self, "mode_coal_budget", False):
                self._seen_canon_mask_keys = set()
                self._budget_skipped = 0

            x_full = np.asarray(X_np[row_i]).reshape(-1)

            # Edge cases under BG expectation:
            # v0 = mean_{z in BG} f(z)
            # v1 = f(x_full)  (mean over BG of all-features-replaced == f(x_full))
            mask_empty = np.zeros(M, dtype=np.int8)
            mask_full = np.ones(M, dtype=np.int8)
            v0 = _v_mean_bg(mask_empty, x_full)
            v1 = _v_mean_bg(mask_full, x_full)

            want_bg = bool(getattr(self, "mode_bg", False))
            want_canon = bool(getattr(self, "mode_coal_canon", False))
            want_budget = bool(getattr(self, "mode_coal_budget", False))

            # ------------------------------------------------------------
            # Step 1: collect coalitions
            # ------------------------------------------------------------
            masks = []
            sizes_can = []
            probs_raw = []
            keys = []

            def _push(mask, s_can, prob, key):
                self._coal_record("leverage", mask)
                masks.append(mask)
                sizes_can.append(int(s_can))
                probs_raw.append(float(prob))
                keys.append(key)

            # FAST PATH: if NO mode-bg / NO canon / NO budget / NO provenance
            # -> still BG expectation, but sampling is one-shot (no top-up)
            if (not want_bg) and (not want_canon) and (not want_budget) and (not getattr(self, "mode_provenance", False)):
                sampler = _CoalitionSampler(
                    n_players=M,
                    sampling_weights=sampling_weights,
                    pairing_trick=True,
                    seed=self.seed,
                    sample_with_replacement=False,
                )
                sampler.sample(ns)

                Z = np.asarray(sampler._sampled_coalitions_matrix, dtype=np.int8)
                probs = np.asarray(sampler._sampled_coalitions_probability, dtype=float)
                sizes = np.sum(Z, axis=1).astype(int)
                
                keep = np.where((sizes > 0) & (sizes < M))[0]
                Z = Z[keep]
                probs = probs[keep]
                sizes = sizes[keep]

                for t in range(Z.shape[0]):
                    mask = np.asarray(Z[t], dtype=np.int8)
                    s = int(sizes[t])
                    key = _mask_key(mask)
                    _push(mask, s, probs[t], key)

            else:
                # GENERAL PATH: supports mode-bg canonicalization + top-up for edgecases,
                seen = set()
                n_dup_skipped = 0
                n_edge_skipped = 0
                max_skips = int(getattr(self, "budget_max_skips", 0)) if want_budget else 0

                total_drawn = 0
                max_total_draws = max(10 * ns, ns + 20000)

                it = 0
                while len(masks) < ns and total_drawn < max_total_draws:
                    need = ns - len(masks)
                    draw = max(200, 8 * need)
                    it += 1
                    total_drawn += draw

                    sampler = _CoalitionSampler(
                        n_players=M,
                        sampling_weights=sampling_weights,
                        pairing_trick=True,
                        seed=self.seed + (self.seed+123) * it,
                        sample_with_replacement=True,
                    )
                    sampler.sample(draw)

                    Z = np.asarray(sampler._sampled_coalitions_matrix, dtype=np.int8)
                    probs = np.asarray(sampler._sampled_coalitions_probability, dtype=float)
                    sizes = np.sum(Z, axis=1).astype(int)

                    
                    keep = np.where((sizes > 0) & (sizes < M))[0]
                    Z = Z[keep]
                    probs = probs[keep]

                    for t in range(Z.shape[0]):
                        on_idx = np.flatnonzero(Z[t]).tolist()
                        on_cols = tuple(self.feature_names[j] for j in on_idx)

                        mask = _canonize_mask_from_cols(on_cols, x_full) if (want_bg or getattr(self, "mode_provenance", False)) else np.asarray(Z[t], dtype=np.int8)
                        s_can = int(np.sum(mask))

                        if s_can <= 0 or s_can >= M:
                            n_edge_skipped += 1
                            continue

                        key = _mask_key(mask)

                        if want_budget:
                            if key in seen:
                                n_dup_skipped += 1
                                if n_dup_skipped > max_skips:
                                    raise RuntimeError(
                                        f"[mode-coalition-budget][leverage] too many duplicates skipped "
                                        f"({n_dup_skipped} > budget_max_skips={max_skips}). "
                                        f"Increase --budget-max-skips or lower nsamples."
                                    )
                                continue
                            seen.add(key)

                        _push(mask, s_can, probs[t], key)

                        if len(masks) >= ns:
                            break

                if len(masks) < ns:
                    raise RuntimeError(
                        f"[leverage] unable to collect enough usable samples: got {len(masks)} / {ns}. "
                        f"(edge_skipped={n_edge_skipped}, want_budget={want_budget}, max_skips={max_skips}, total_drawn={total_drawn})."
                    )

            # ------------------------------------------------------------
            # Step 2: reduction (ONLY for canon mode, and ONLY when budget mode is not already unique)
            # ------------------------------------------------------------
            if want_canon and (not want_budget):
                mask_map = {}
                size_map = {}
                w_total = {}

                for t in range(len(masks)):
                    key = keys[t]
                    mask = masks[t]
                    s = sizes_can[t]
                    prob = probs_raw[t]

                    if key not in mask_map:
                        mask_map[key] = mask.copy()
                        size_map[key] = int(s)
                        w_total[key] = 0.0

                    w_total[key] += _lev_w(prob, s)

                keys_red = list(mask_map.keys())
                K = len(keys_red)
                if K == 0:
                    out[row_i, :] = (v1 - v0) / float(M)
                    pbar.update(1)
                    return

                Z_can = np.zeros((K, M), dtype=np.int8)
                sizes_vec = np.zeros(K, dtype=int)
                kernel_w = np.zeros(K, dtype=float)

                for j, key in enumerate(keys_red):
                    Z_can[j, :] = mask_map[key]
                    sizes_vec[j] = size_map[key]
                    kernel_w[j] = w_total[key]

            else:
                K = len(masks)
                Z_can = np.asarray(masks, dtype=np.int8)
                sizes_vec = np.asarray(sizes_can, dtype=int)
                kernel_w = np.zeros(K, dtype=float)
                for j in range(K):
                    kernel_w[j] = _lev_w(probs_raw[j], int(sizes_vec[j]))

            # ------------------------------------------------------------
            # Step 3: evaluate v(S) with BG expectation
            # ------------------------------------------------------------
            vals = np.zeros(K, dtype=float)
            for j in range(K):
                vals[j] = _v_mean_bg(Z_can[j, :], x_full)

            # ------------------------------------------------------------
            # Step 4: official LeverageSHAP WLS math
            # ------------------------------------------------------------
            values_adjusted = vals - (v1 - v0) * (sizes_vec / float(M))

            P = np.eye(M) - (1.0 / M) * np.ones((M, M))

            Wy = kernel_w * values_adjusted
            WZ = kernel_w[:, None] * Z_can

            Atb = P @ (Z_can.T @ Wy)
            AtA = P @ (Z_can.T @ WZ) @ P

            if np.linalg.cond(AtA) > 1 / np.finfo(AtA.dtype).eps and ns <= 3 * M:
                sqrt_alpha = 1e-3
                AtA = AtA + sqrt_alpha * np.eye(AtA.shape[0])

            sol = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
            out[row_i, :] = sol + (v1 - v0) / float(M)

            pbar.update(1)
            # end of _iter_row

        parallel_loop(
            _iter_row,
            list(range(X_np.shape[0])),
            n_jobs=self.n_jobs
        )

        pbar.close()
        return out

    def _apply_provenance_mask_expansion(self, m: np.ndarray, x: np.ndarray) -> np.ndarray:
        dbg = bool(getattr(self, "debug", False))
        if (not self.mode_provenance) or (self.prov_index is None):
            return m

        m = np.asarray(m, dtype=np.int8).reshape(-1)
        x = np.asarray(x).reshape(-1)

        before_prov = _summ_mask(self.feature_names, m) if dbg else None
        before_cols = [self.feature_names[i] for i in np.flatnonzero(m)] if dbg else None
        reasons = [] if dbg else None

        if dbg:
            self._dbg(f"[prov] before on={int(m.sum())}/{len(m)} {before_cols}")

        k_mask = _mask_key(m)
        on_idx = np.flatnonzero(m)

        x_bytes = x.tobytes()
        if self._x_canon_key != x_bytes:
            self._x_canon_key = x_bytes
            self._x_canon_val = [_canon_scalar(x[i]) for i in range(len(self.feature_names))]
        x_canon_full = self._x_canon_val
        x_key_on = tuple((int(i), x_canon_full[int(i)]) for i in on_idx)
        k0 = (k_mask, x_key_on)

        hit = self._prov_cache.get(k0)
        if hit is not None:
            buf = np.frombuffer(hit, dtype=np.uint8)
            bits = np.unpackbits(buf, bitorder="little", count=len(m))
            m[:] = bits.astype(np.int8, copy=False)
            return m

        thr = int(self.prov_threshold)
        on_set = set(self.feature_names[i] for i in np.flatnonzero(m))
        is_weak = (getattr(self, "prov_strength", "strong") == "weak")

        # ------------------------
        # helpers
        # ------------------------
        def best_key_for_id_strong(idc: str, *, threshold: int, on_set_local: set):
            det = self.prov_index.det_cols.get(idc, [])
            key_cols = tuple([c for c in det if c in on_set_local])
            if not key_cols:
                return None

            cand_ids, const_rhs = self.prov_index.infer_ids_from_canon(
                idcol=idc,
                key_cols=key_cols,
                x_canon=x_canon_full,
                col2i=self.col2i,
                threshold=int(threshold),
            )

            if 0 < len(cand_ids) <= int(threshold):
                if dbg:
                    self._dbg(
                        f"[prov] idc={idc} ACCEPT key_cols(all_on)={key_cols} "
                        f"cand_ids_n={len(cand_ids)} thr={threshold}"
                    )
                return key_cols, cand_ids, const_rhs

            if dbg:
                self._dbg(
                    f"[prov] idc={idc} REJECT key_cols(all_on)={key_cols} "
                    f"cand_ids_n={len(cand_ids)} thr={threshold}"
                )
            return None

        def best_key_for_id_weak(idc: str, *, on_set_local: set):
            det = self.prov_index.det_cols.get(idc, [])
            key_cols = tuple([c for c in det if c in on_set_local])
            if not key_cols:
                return None

            cand_ids, const_rhs = self.prov_index.infer_ids_and_const_from_canon(
                idcol=idc,
                key_cols=key_cols,
                x_canon=x_canon_full,
                col2i=self.col2i,
            )

            if len(cand_ids) == 0 and len(const_rhs) == 0:
                if dbg:
                    self._dbg(f"[prov-weak] idc={idc} MISS key_cols(all_on)={key_cols}")
                return None

            if dbg:
                self._dbg(
                    f"[prov-weak] idc={idc} HIT key_cols(all_on)={key_cols} "
                    f"cand_ids_n={len(cand_ids)}"
                )
            return key_cols, cand_ids, const_rhs

        stage0_hits = []
        for idc in self.prov_index.det_cols.keys():
            if is_weak:
                got = best_key_for_id_weak(idc, on_set_local=on_set)
                if got is None:
                    continue
                key_cols, cand_ids, const_rhs = got
                if len(cand_ids) == 0:
                    continue
                stage0_hits.append((idc, key_cols, tuple(cand_ids), set(const_rhs)))
            else:
                got = best_key_for_id_strong(idc, threshold=thr, on_set_local=on_set)
                if got is None:
                    continue
                key_cols, cand_ids, const_rhs = got
                stage0_hits.append((idc, key_cols, tuple(cand_ids), set(const_rhs)))

        multi_added = []
        if len(stage0_hits) >= 2:
            # Use pre-built row-index sets: intersect row sets for each id's candidate values.
            row_sets = []
            for (idc, _kcols, cand_ids_t, _const_rhs) in stage0_hits:
                id_map = self.prov_index._id_row_sets.get(idc, {})
                s: set = set()
                for cid in cand_ids_t:
                    rows = id_map.get(cid)
                    if rows:
                        s |= rows
                row_sets.append(s)

            intersected_rows: set = row_sets[0]
            for s in row_sets[1:]:
                intersected_rows = intersected_rows & s

            if intersected_rows:
                col_arr = self.prov_index._col_arr
                for (idc, key_cols, _cand_ids_t, const_rhs0) in stage0_hits:
                    if idc not in col_arr:
                        continue

                    if not is_weak:
                        idc_arr = col_arr[idc]
                        ids_n = len({idc_arr[i] for i in intersected_rows if not (isinstance(idc_arr[i], float) and math.isnan(idc_arr[i]))})
                        if not (0 < ids_n <= thr):
                            continue

                        rules = self.prov_index.idcol_to_rules.get(idc, [])
                        for req_feats, _req_ids, rhs in rules:
                            if rhs not in self.col2i:
                                continue

                            # NEW: gate by LHS ⊆ current mask
                            if not req_feats.issubset(on_set):
                                continue

                            j = self.col2i.get(rhs)
                            if j is None or m[j] == 1:
                                continue
                            m[j] = 1
                            multi_added.append((idc, rhs, key_cols))
                            if reasons is not None:
                                reasons.append(
                                    f"[multi-intersect][FD] {idc} -> {rhs} "
                                    f"(identified_by={list(key_cols)}, ids_n<=thr={thr})"
                                )
                    else:
                        rules = self.prov_index.idcol_to_rules.get(idc, [])
                        for req_feats, _req_ids, rhs in rules:
                            if rhs not in self.col2i:
                                continue
                            # LHS gating
                            if not req_feats.issubset(on_set):
                                continue

                            j = self.col2i.get(rhs)
                            if j is None or m[j] == 1:
                                continue
                            # optional consistency check: rhs is constant across intersected rows
                            if rhs in col_arr:
                                rhs_arr = col_arr[rhs]
                                rhs_vals = {rhs_arr[i] for i in intersected_rows}
                                if len(rhs_vals) != 1:
                                    continue
                            m[j] = 1
                            multi_added.append((idc, rhs, key_cols))
                            if reasons is not None:
                                reasons.append(
                                    f"[multi-intersect][weak] {idc} -> {rhs} "
                                    f"(identified_by={list(key_cols)}; const_rhs)"
                                )

                if multi_added:
                    on_set = set(self.feature_names[i] for i in np.flatnonzero(m))
                    if dbg:
                        participants = "; ".join(
                            f"{idc}:key={list(kcols)}" for (idc, kcols, _ids, _cr) in stage0_hits
                        )
                        added_preview = "; ".join(
                            f"{idc}->{rhs}" for (idc, rhs, _kcols) in multi_added[:6]
                        )
                        tail = " ..." if len(multi_added) > 6 else ""
                        self._dbg(f"[prov] [multi-intersect] participants={participants} | added={added_preview}{tail}")

        for idc in self.prov_index.det_cols.keys():
            got = best_key_for_id_strong(idc, threshold=thr, on_set_local=on_set)
            if got is None:
                continue
            key_cols, cand_ids, const_rhs = got
            rules = self.prov_index.idcol_to_rules.get(idc, [])
            for req_feats, req_ids, rhs in rules:
                if rhs not in self.col2i:
                    continue
                # only use the rule if all required feature-determiners are ON in the mask
                lhs_ok = req_feats.issubset(on_set)
                # if dbg:
                #     self._dbg(
                #         f"[prov][lhs-check] {idc}->{rhs} "
                #         f"lhs={list(req_feats)} ok={lhs_ok}"
                #     ) # too much (just for debugging)
                if not lhs_ok:
                    continue

                j = self.col2i.get(rhs)
                if j is None or m[j] == 1:
                    continue
                m[j] = 1
                if reasons is not None:
                    reasons.append(
                        f"{idc} -> {rhs} "
                        f"(identified_by={list(key_cols)}, cand_ids_n={len(cand_ids)})"
                    )

        on_set = set(self.feature_names[i] for i in np.flatnonzero(m))

        do_stage2 = (self.provenance_mode in ("bg-coalition-canon", "bg-coalition-budget"))
        if do_stage2:
            for idc in self.prov_index.det_cols.keys():
                got = best_key_for_id_strong(idc, threshold=thr, on_set_local=on_set)
                if got is None:
                    continue
                key_cols, cand_ids, const_rhs = got
                if len(cand_ids) != 1:
                    continue

                rules = self.prov_index.idcol_to_rules.get(idc, [])
                for req_feats, req_ids, rhs in rules:
                    if rhs not in self.col2i:
                        continue
                    lhs_ok = req_feats.issubset(on_set)
                    # if dbg:
                    #     self._dbg(
                    #         f"[prov][lhs-check] {idc}->{rhs} "
                    #         f"lhs={list(req_feats)} ok={lhs_ok}"
                    #     ) # too much (just for debugging)
                    if not lhs_ok:
                        continue

                    j = self.col2i.get(rhs)
                    if j is None or m[j] == 1:
                        continue
                    m[j] = 1
                    if reasons is not None:
                        reasons.append(
                            f"{idc} -> {rhs} "
                            f"(identified_by={list(key_cols)}, cand_ids_n=1)"
                        )


        # ============================================================
        # weak-extra: only when prov_strength == "weak"
        #   - preserve your original weak semantics
        # ============================================================
        if is_weak:
            before_weak_cols = [self.feature_names[i] for i in np.flatnonzero(m)] if dbg else None
            weak_reasons = [] if dbg else None

            for idc in self.prov_index.det_cols.keys():
                got = best_key_for_id_weak(idc, on_set_local=on_set)
                if got is None:
                    continue
                key_cols, cand_ids, const_rhs = got

                if len(cand_ids) <= thr:
                    continue

                # rule-gated RHS expansion (weak extra)
                rules = self.prov_index.idcol_to_rules.get(idc, [])
                allowed_rhs = set()
                for req_feats, req_ids, rhs in rules:
                    if rhs not in self.col2i:
                        continue
                    lhs_ok = req_feats.issubset(on_set)
                    # if dbg:
                    #     self._dbg(
                    #         f"[prov][lhs-check] {idc}->{rhs} "
                    #         f"lhs={list(req_feats)} ok={lhs_ok}"
                    #     ) # too much (just for debugging)
                    if not lhs_ok:
                        continue
                    allowed_rhs.add(rhs)

                for rhs in const_rhs:
                    if rhs not in allowed_rhs:
                        continue
                    j = self.col2i.get(rhs)
                    if j is None or m[j] == 1:
                        continue
                    m[j] = 1
                    if weak_reasons is not None:
                        weak_reasons.append(
                            f"[weak] {idc} -> {rhs} "
                            f"(identified_by={list(key_cols)}, cand_ids_n={len(cand_ids)}, req_feats_ok=1)"
                        )

            if dbg:
                after_weak_cols = [self.feature_names[i] for i in np.flatnonzero(m)]
                if set(after_weak_cols) != set(before_weak_cols or []):
                    because = " ; ".join((weak_reasons or [])[:6]) + (" ..." if weak_reasons and len(weak_reasons) > 6 else "")
                    _dbg3(
                        self._dbg,
                        before=_summ_mask(
                            self.feature_names,
                            np.array([1 if c in set(before_weak_cols or []) else 0 for c in self.feature_names], dtype=np.int8),
                        ),
                        because=because if because else "[weak-extra] const_rhs expansion",
                        after=_summ_mask(self.feature_names, m),
                        tag="[PROV-WEAK]",
                    )

        # ============================================================
        # Final debug
        # ============================================================
        if dbg:
            self._dbg(f"[prov] after  {_summ_mask(self.feature_names, m)}")
            after_cols = [self.feature_names[i] for i in np.flatnonzero(m)]
            if set(after_cols) != set(before_cols or []):
                because = " ; ".join(reasons[-6:]) + (" ..." if reasons and len(reasons) > 6 else "")
                if not because:
                    because = f"provenance expansion (threshold={thr}, strength={'weak' if is_weak else 'strong'})"
                _dbg3(
                    self._dbg,
                    before=before_prov,
                    because=because,
                    after=_summ_mask(self.feature_names, m),
                    tag="[PROV]",
                )

        # store packed mask
        self._prov_cache[k0] = _mask_key(m)
        return m


    def _get_varying_inds(self):
        vi = getattr(self, "varyingInds", None)
        if vi is None:
            vi = getattr(self, "varying_inds", None)
        if vi is None:
            raise RuntimeError("KernelExplainer did not set varyingInds; cannot map mask.")
        return np.asarray(vi, dtype=int)
    
    def _ensure_x_full(self, x, varying, M_full):
        xv = np.asarray(x).reshape(-1)

        if xv.size == M_full:
            return xv
        
        if xv.size == varying.size:
            bg0 = np.asarray(self.data.data[0]).reshape(-1)
            if bg0.size != M_full: # just in case
                bg0 = np.resize(bg0, M_full)

            x_full = bg0.copy()
            x_full[varying] = xv
            return x_full

        return xv

    def _coal_record(self, mode: str, mask: np.ndarray) -> None:
        """Record coalition mask for debugging: store (capped) + size histogram (uncapped)."""
        if not getattr(self, "debug_coal", False):
            return

        m = np.asarray(mask, dtype=np.int8).reshape(-1)
        s = int(m.sum())
        self._coal_size_hist_by_mode[mode][s] += 1

        if len(self._coal_keys_by_mode[mode]) < self.debug_coal_limit:
            self._coal_keys_by_mode[mode].append(_mask_key(m))

    def dump_coalitions(self) -> None:
        """Print recorded coalitions and size histograms."""
        if not getattr(self, "debug_coal", False):
            return

        print("\n========== [debug-coal] coalition size histograms ==========")
        for mode, ctr in self._coal_size_hist_by_mode.items():
            items = sorted(ctr.items(), key=lambda x: x[0])
            total = sum(ctr.values())
            sizes_str = "{" + ", ".join(f"{k}:{v}" for k, v in items) + "}"
            print(f"[{mode}] total={total}  sizes={sizes_str}")
            if mode == 'kernel':
                print(f"0 not used in the WLS")
            elif mode == 'mc':
                print(f"including 0")
            elif mode == 'leverage':
                print(f"0 not used in the WLS; not even counted")

    def _ensure_full_mask_and_x(self, m, x):
        m = np.asarray(m, dtype=np.int8).reshape(-1)
        varying = self._get_varying_inds()
        M_full = len(self.feature_names)

        if len(m) == M_full:
            return m

        m_full = np.zeros(M_full, dtype=np.int8)
        m_full[varying] = m
        x = self._ensure_x_full(x, varying, M_full)
        return m_full

    def addsample(self, x, m, w, *args, **kwargs):
        """
        Called by SHAP while building the weighted regression dataset.
        We:
          1) compute raw_key from mask
          2) compute closure(raw) -> canon set
          3) apply canon to mask (valid-world)
          4) if budget: drop duplicate canon masks
          5) proceed with super().addsample
        """

        if not self.mode_bg:
            if self.mode_provenance:
                m = self._ensure_full_mask_and_x(m, x)
                if self.mode_provenance and self.prov_index is not None:
                    m = self._apply_provenance_mask_expansion(m, x)

            return super().addsample(x, m, w, *args, **kwargs)


        if self.closure_cache is None:
            raise ValueError("--mode-bg requires --constraints-cache")

        m = np.asarray(m, dtype=np.int8).reshape(-1).copy()

        varying = self._get_varying_inds()
        M_full = len(self.feature_names)

        if len(m) != M_full:
            m_full = np.zeros(M_full, dtype=np.int8)
            m_full[varying] = m
            m = m_full

            x = self._ensure_x_full(x, varying, M_full)

            if self.debug:
                self._dbg(f"[varying->full] K={len(varying)} M={M_full} | replaced m,x with full")


        on_idx = np.flatnonzero(m)  # slightly faster than np.where
        raw_S = self._feat_arr[on_idx]
        raw_key = tuple(sorted(set(raw_S.tolist())))

        if self.debug:
            before_fd = _summ_mask(self.feature_names, m)

        if self.debug:
            self._dbg(f"[addsample] raw {_summ_mask(self.feature_names, m)} raw_key={list(raw_key)}")
        
        collect_prov = bool(self.debug)
        clo, prov = self.closure_cache.closure_and_provenance(raw_key,collect_prov=collect_prov,)

        if self.debug:
            self._dbg(f"[addsample] clo added_n={len(set(clo)-set(raw_key))} clo_head={list(clo)[:12]}{'...' if len(clo)>12 else ''}")


        for col in clo:
            j = self._col_to_i_cache.get(col)
            if j is None:
                j = self.col2i.get(col)
                if j is None:
                    continue
                self._col_to_i_cache[col] = j
            m[j] = 1

        if self.debug:
            after_fd = _summ_mask(self.feature_names, m)
        if self.debug and after_fd != before_fd:
            added = sorted(set(clo) - set(raw_key))

            if self.debug and added:
                # build rule reasons like "age -> life_stage"
                rule_reasons = []
                for rhs in added[:8]:
                    lhs_list = (prov or {}).get(rhs, [])
                    if lhs_list:
                        # prov[rhs] can have multiple lhs; show first (or join)
                        lhs = list(lhs_list[0])
                        rule_reasons.append(f"{lhs} -> {rhs}")
                    else:
                        rule_reasons.append(f"(unknown_lhs) -> {rhs}")

                because = "FD: " + " ; ".join(rule_reasons) + (" ..." if len(added) > 8 else "")
                if self.mode_coal_canon:
                    fd_tag = "[FD|COAL-CANON]"
                elif self.mode_coal_budget:
                    fd_tag = "[FD|COAL-BUDGET]"
                else:
                    fd_tag = "[FD|BG-ONLY]"

                _dbg3(
                    self._dbg,
                    before=before_fd,
                    because=because,
                    after=after_fd,
                    tag=fd_tag,
                )


        m_before_prov = m.copy()
        if self.mode_provenance and self.prov_index is not None and prov is not None:
            m = self._apply_provenance_mask_expansion(m, x)
        canon_mask_key = _mask_key(m)
        if self.debug:
            if _mask_key(m_before_prov) != _mask_key(m):
                self._dbg(f"[addsample] prov expanded {_diff_cols([self.feature_names[i] for i in np.where(m_before_prov==1)[0]], [self.feature_names[i] for i in np.where(m==1)[0]])}")

        if self.mode_coal_budget and self._budget_desired_nsamples is not None:
            # nsamplesAdded is maintained by SHAP's addsample (super().addsample)
            cur_added = int(getattr(self, "nsamplesAdded", 0) or 0)
            if cur_added >= self._budget_desired_nsamples:
                return None


        if self.mode_coal_budget:
            if canon_mask_key in self._seen_canon_mask_keys:
                self._budget_skipped += 1
                if self.debug:
                    self._dbg(f"[budget] DUP skip={self._budget_skipped} key_len={len(canon_mask_key)} cur_added={int(getattr(self,'nsamplesAdded',0) or 0)}")
                if self._budget_skipped > self.budget_max_skips:
                    raise RuntimeError(
                        f"[mode-coalition-budget] too many duplicates skipped "
                        f"({self._budget_skipped} > budget_max_skips={self.budget_max_skips})."
                    )

                # offset (256 trials)
                rng = np.random.RandomState(self.seed + self._budget_skipped)  # deterministic-ish
                for _try in range(256):  # for safety
                    m2 = self._rand_mask_like(rng)

                    # apply FD closure to m2
                    on2 = np.where(m2 == 1)[0]
                    raw2 = [self.feature_names[i] for i in on2]
                    raw2_key = _key_from_cols(raw2)
                    clo2, _ = self.closure_cache.closure_and_provenance(raw2_key,collect_prov=collect_prov,)

                    for col in clo2:
                        j = self.col2i.get(col)
                        if j is not None:
                            m2[j] = 1

                    if self.mode_provenance and self.prov_index is not None:
                        m2 = self._apply_provenance_mask_expansion(m2, x)

                    k2 = _mask_key(m2)

                    if k2 in self._seen_canon_mask_keys:
                        continue  # still duplicate, try again

                    # found a new unique canon mask -> accept it
                    self._seen_canon_mask_keys.add(k2)

                    # (optional) cache append like original
                    canon2 = tuple(sorted(self.feature_names[i] for i in np.where(m2 == 1)[0]))

                    # bg_cache도 "raw -> final canon"으로 통일 (FD+prov 반영)
                    bg_hit2 = self.closure_cache.lookup_bg(raw2_key) is not None
                    if not bg_hit2:
                        self.closure_cache.remember_bg(raw2_key, canon2)

                    # class_cache도 그대로 canon2
                    self.closure_cache.remember_class(canon2)


                    if self.debug:
                        self._dbg(f"[budget] REPLACE try_ok raw2_key={list(raw2_key)} canon2_on={canon2[:12]}{'...' if len(canon2)>12 else ''}")


                    # add the *replacement* sample using same x and w
                    return super().addsample(x, m2, w, *args, **kwargs)

                # 256 trials -> no match then drop
                if self.debug:
                    self._dbg("[budget] REPLACE failed after 256 tries -> drop this sample")

                return None

            # not duplicate -> record and continue normally
            self._seen_canon_mask_keys.add(canon_mask_key)




        # bg cache append (raw -> closure) if new
        bg_hit = self.closure_cache.lookup_bg(raw_key) is not None
        canon_cols = tuple(sorted(self.feature_names[i] for i in np.where(m == 1)[0]))
        if not bg_hit:
            self.closure_cache.remember_bg(raw_key, canon_cols)


        # class cache append (canon class) if coalition mode enabled
        do_stage2 = (self.provenance_mode in ("bg-coalition-canon", "bg-coalition-budget"))
        if do_stage2:
            if self.provenance_mode == "bg-coalition-canon" and not self.mode_coal_canon:
                do_stage2 = False
            if self.provenance_mode == "bg-coalition-budget" and not self.mode_coal_budget:
                do_stage2 = False
            canon_cols = tuple(sorted(self.feature_names[i] for i in np.where(m == 1)[0]))
            self.closure_cache.remember_class(canon_cols)
            
            if self.debug:
                self._dbg(f"[class-cache] remember_class canon_cols={list(canon_cols)[:12]}{'...' if len(canon_cols)>12 else ''}")


        varying = self._get_varying_inds()
        M_full = len(self.feature_names)
        K = varying.size
        x_full = self._ensure_x_full(x, varying, M_full)
        x_out = np.asarray(x_full).reshape(1, -1)   # (1, M_full)

        m = np.asarray(m, dtype=np.int8).reshape(-1)
        if m.size == M_full:
            m = m[varying].astype(np.int8, copy=False)  # (K,)

        self._coal_record("kernel", m)
        if self.debug:
            self._dbg(f"[END->super] x_out.shape={x_out.shape} len(m)={len(m)} K={K} M_full={M_full}")

        return super().addsample(x_out, m, w, *args, **kwargs)


    def run(self):
        """
        Safe composition:
        - optional IC mask-aware repair (protect x-side: mask==1 never modified)
        - optional coalition canon (block-wise cache by canon mask)
        Implemented with a SINGLE model.f swap to avoid recursive self-wrapping.
        """
        # --- grab internals ---
        mask_mat = getattr(self, "maskMatrix", None)
        synth_data = getattr(self, "synth_data", None)
        model_obj = getattr(self, "model", None)
        data_obj = getattr(self, "data", None)

        if mask_mat is None or synth_data is None or model_obj is None or data_obj is None:
            # can't do anything fancy
            return super().run()

        # background size
        bg_data = getattr(data_obj, "data", data_obj)
        bg_data = np.asarray(bg_data)
        n_bg = int(bg_data.shape[0])

        mask_mat = np.asarray(mask_mat, dtype=np.uint8)
        n_masks = int(mask_mat.shape[0])

        Xs = getattr(synth_data, "data", synth_data)
        Xs = np.asarray(Xs)

        expected = n_masks * n_bg
        if Xs.shape[0] != expected:
            # KernelSHAP internal ordering assumption broken -> fall back
            return super().run()

        if self.debug:
            self._dbg(f"[run] n_masks={n_masks} n_bg={n_bg} expected={expected} Xs.shape={Xs.shape}")
            self._dbg(f"[run] mask_mat.shape={mask_mat.shape} synth_data.shape={Xs.shape}")


        # ====== capture the TRUE base f exactly once ======
        base_f = getattr(model_obj, "f", None)
        if base_f is None:
            return super().run()

        # ====== setup IC repair metadata (optional) ======
        repair_fn = getattr(self, "_ic_repair_fn", None)
        feature_cols = getattr(self, "_ic_feature_cols", self.feature_names)

        if repair_fn is not None:
            off_idx_list = [np.where(mask_mat[i] == 0)[0] for i in range(n_masks)]
            on_idx_list  = [np.where(mask_mat[i] == 1)[0] for i in range(n_masks)]
            fixed_cols_set_list = [set(feature_cols[j] for j in on_idx_list[i]) for i in range(n_masks)]
        else:
            off_idx_list = None
            on_idx_list = None
            fixed_cols_set_list = None


        if self.mode_coal_canon:
            self._eval_cache = {}
            self._eval_cache_hits = 0
            self._eval_cache_misses = 0
            keys = [_mask_key(mask_mat[i]) for i in range(n_masks)]
        else:
            keys = None

        # shared ctx (optional)
        ctx = getattr(self, "_ic_eval_ctx", None)

        # ====== composed wrapper ======
        def f_composed(X_in):
            X_in = np.asarray(X_in)

            # Only handle the expected synth batch; otherwise just call base_f
            if X_in.shape[0] != expected:
                return base_f(X_in)

            if ctx is not None:
                ctx["in_synth"] = True
                ctx["total_n"] = int(X_in.shape[0])

            try:
                # Start from original synth batch
                X_work = X_in

                # ---------- IC mask-aware repair (x-side protected) ----------
                if repair_fn is not None:
                    X_rep = X_work.copy()
                    for i in range(n_masks):
                        off_idx = off_idx_list[i]
                        if off_idx.size == 0:
                            continue
                        start = i * n_bg
                        end = (i + 1) * n_bg

                        block_df = pd.DataFrame(X_rep[start:end], columns=feature_cols)
                        # fixed_cols = set(feature_cols[j] for j in np.where(mask_mat[i] == 1)[0].tolist())
                        fixed_cols = fixed_cols_set_list[i]
                        off_idx = off_idx_list[i]
                        fixed_df = repair_fn(block_df, fixed_cols=fixed_cols)
                        

                        # debug: count changes ONLY on off_idx
                        if self.debug:
                            # before/after only on bg-side (mask==0)
                            before_off = block_df.iloc[:, off_idx]
                            after_off  = fixed_df.iloc[:, off_idx]

                            eq = before_off.eq(after_off) | (before_off.isna() & after_off.isna())
                            changed_cells = int((~eq).to_numpy().sum())

                            # which columns changed (on off-side)
                            changed_cols_mask = (~eq).any(axis=0).to_numpy()
                            changed_cols = before_off.columns.to_numpy()[changed_cols_mask].tolist()

                            diff_cols_mask = changed_cols[:12]
                            examples = _cell_change_examples(block_df, fixed_df, diff_cols_mask, k=3)
                            if not examples:
                                examples = ["(no cell changed; IC already satisfied)"]

                            # --- WHY (debug-only): attach reason per example if available ---
                            why_fn = getattr(repair_fn, "_why_for_cell", None)
                            if callable(why_fn):
                                if examples:
                                    examples2 = []
                                    # prepare row slices once (cheap)
                                    for s in examples:
                                        # s format: "row=R col=C: before -> after"
                                        try:
                                            # very small parser
                                            # row=0 col=oh2: 1 -> 0
                                            left = s.split(":", 1)[0].strip()   # "row=0 col=oh2"
                                            r_str = left.split("row=", 1)[1].split(" ", 1)[0]
                                            c_str = left.split("col=", 1)[1].strip()
                                            rr = int(r_str)
                                            cc = c_str

                                            before_row = block_df.iloc[rr]
                                            after_row  = fixed_df.iloc[rr]
                                            why = why_fn(before_row=before_row, after_row=after_row, col=cc, fixed_cols=fixed_cols)
                                            examples2.append(s + f" | why: {why}")
                                        except Exception:
                                            # if parsing fails, keep original
                                            examples2.append(s)
                                    examples = examples2
                                else:
                                    # NEW: 변화 없을 때 WHY 요약
                                    why_summary = []
                                    for c in before_off.columns[:5]:  # 너무 길어지지 않게
                                        try:
                                            why = why_fn(
                                                before_row=block_df.iloc[0],
                                                after_row=fixed_df.iloc[0],
                                                col=c,
                                                fixed_cols=fixed_cols,
                                            )
                                            if why and "unknown" not in why:
                                                why_summary.append(f"{c}: {why}")
                                        except Exception:
                                            pass

                                    if why_summary:
                                        examples = ["WHY(no-change): " + " | ".join(why_summary)]
                                    else:
                                        examples = ["WHY(no-change): IC guard false or already satisfied"]

                            # summarize fixed/protected cols
                            fixed_on = np.where(mask_mat[i] == 1)[0]
                            fixed_n = int(len(fixed_on))
                            off_n = int(len(off_idx))

                            # accurate flags (you set these in main)
                            dom_on = bool(getattr(self, "_ic_mode_domain", False))
                            den_on = bool(getattr(self, "_ic_mode_denial", False))
                            enabled = []
                            if dom_on: enabled.append("domain")
                            if den_on: enabled.append("denial")
                            enabled_s = "+".join(enabled) if enabled else "IC"

                            # infeasible notes (denial)
                            inf1 = int(fixed_df.attrs.get("_denial_onehot_infeasible", 0))
                            inf2 = int(fixed_df.attrs.get("_denial_parity_infeasible", 0))

                            before_msg = f"block i={i} | fixed(mask=1)={fixed_n} cols | editable(mask=0)={off_n} cols"
                            because_msg = f"{enabled_s} repair on mask==0 only; protected fixed_cols(mask==1)={fixed_n}"
                            after_msg = (
                                f"changed_cells(mask0_only)={changed_cells} | "
                                f"changed_cols={diff_cols_mask}"
                                + (f" | examples={examples}" if examples else "")
                            )

                            if inf1 or inf2:
                                after_msg += f" | infeasible(onehot)={inf1} infeasible(parity)={inf2}"

                            _dbg3(
                                self._dbg,
                                before=f"block i={i} | fixed(mask=1)={int(mask_mat[i].sum())} cols | editable(mask=0)={len(off_idx)} cols",
                                because=f"repair on mask==0 only; protected fixed_cols(mask==1)={int(mask_mat[i].sum())}",
                                after=after_msg,
                                tag="[IC]"
                            )



                        # overwrite ONLY bg-side columns (mask==0)
                        X_rep[start:end, off_idx] = fixed_df.iloc[:, off_idx].to_numpy()

                    X_work = X_rep

                # ---------- Memoization: evaluate per mask block ----------
                if self.mode_coal_canon:
                    out_blocks = []
                    for i, k in enumerate(keys):
                        start = i * n_bg
                        end = (i + 1) * n_bg
                        if k in self._eval_cache:
                            yb = self._eval_cache[k]
                            self._eval_cache_hits += 1
                        else:
                            yb = base_f(X_work[start:end])
                            yb = np.asarray(yb)
                            self._eval_cache[k] = yb
                            self._eval_cache_misses += 1
                        out_blocks.append(yb)
                    return np.concatenate(out_blocks, axis=0)

                return base_f(X_work)

            finally:
                if ctx is not None:
                    ctx["in_synth"] = False
                    ctx["total_n"] = None

        # ====== swap once ======
        model_obj.f = f_composed
        try:
            ret = super().run()
            return ret
        finally:
            model_obj.f = base_f
            if self.mode_coal_canon and self.debug:
                self._dbg(f"[canon] hits={self._eval_cache_hits} misses={self._eval_cache_misses} unique_masks={len(self._eval_cache)}")




# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--flattened", required=True)
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--constraints-cache", type=str, default=None)

    parser.add_argument("--background-n", type=int, default=100)
    parser.add_argument("--explain-n", type=int, default=50)
    parser.add_argument("--nsamples", type=int, default=100)

    parser.add_argument("--base-mode", type=str, default="kernel", choices=["kernel", "mc", "leverage"])

    parser.add_argument("--mode-bg", action="store_true")
    parser.add_argument("--bg-lut", type=str, required=True, help="background sample lookup table")

    # coalition options (mutually exclusive; require bg)
    parser.add_argument("--mode-coalition-canon", action="store_true")
    parser.add_argument("--mode-coalition-budget", action="store_true")
    parser.add_argument("--budget-max-skips", type=int, default=500)

    # NEW: IC modes (independent)
    parser.add_argument("--mode-domain", action="store_true")
    parser.add_argument("--mode-denial", action="store_true")

    parser.add_argument(
        "--mode-domain-numerical",
        type=str,
        default=None,
        choices=["minimal_edit", "random", "weighted_random"],
        help="Numerical repair policy for --mode-domain. "
            "minimal_edit=clamp to nearest boundary; "
            "random=uniform sample inside feasible range; "
            "weighted_random=triangular sampling centered near original value (distance-weighted).",
    )


    parser.add_argument(
        "--mode-provenance",
        type=str,
        default=None,
        choices=["bg-only", "bg-coalition-canon", "bg-coalition-budget"],
        help="Enable provenance refinement: bg-only | bg-coalition-canon | bg-coalition-budget",
    )

    parser.add_argument(
        "--provenance-threshold",
        type=int,
        default=None,
        help="If not set, read PROVENANCE_THRESHOLD from config.yaml; default=1.",
    )

    parser.add_argument(
        "--prov-strength",
        type=str,
        default="strong",
        choices=["strong", "weak"],
        help="Provenance strength. strong=exactly current behavior. weak=strong + const-RHS expansion even when |cand_ids| > threshold.",
    )


    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path relative to base-dir, or absolute path to a saved sklearn Pipeline (.joblib).",
    )

    parser.add_argument("--cache-copy", action="store_true")
    parser.add_argument("--cache-copy-path", type=str, default=None)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-lim", type=int, default=1000, help="max number of debug prints per component")
    parser.add_argument("--debug-every", type=int, default=1, help="print every Nth event (e.g., 10)")

    # NEW: coalition debug (dump + size histogram)
    parser.add_argument("--debug-coal", action="store_true")
    parser.add_argument("--debug-coal-limit", type=int, default=5000)

    parser.add_argument("--out", required=True, help="out csv file")
    args = parser.parse_args()

    repair_fn = None

    SEED = int(args.seed)
    seed_everything(SEED)

    # enforce constraints
    if (args.mode_coalition_canon or args.mode_coalition_budget) and not args.mode_bg:
        raise ValueError("Coalition modes require --mode-bg.")
    if args.mode_coalition_canon and args.mode_coalition_budget:
        raise ValueError("Choose only one: --mode-coalition-canon OR --mode-coalition-budget")

    # NEW: if domain/denial enabled, constraints-cache required (independent of mode-bg)
    if (args.mode_domain or args.mode_denial) and not args.constraints_cache:
        raise ValueError("--mode-domain/--mode-denial require --constraints-cache (IC is in the cache).")

    if args.mode_provenance is not None and not args.constraints_cache:
        raise ValueError("--mode-provenance requires --constraints-cache (to read FD rules).")

    if args.mode_domain:
        domain_num_mode = args.mode_domain_numerical or "minimal_edit"
    else:
        domain_num_mode = None

    if args.mode_domain_numerical is not None and not args.mode_domain:
        raise ValueError("--mode-domain-numerical requires --mode-domain.")

    bg_lut = str(args.bg_lut)
    DEBUG = bool(args.debug)
    DBG_LIM = int(args.debug_lim)
    DBG_EVERY = max(1, int(args.debug_every))

    base_dir = Path(args.base_dir)
    df = pd.read_csv(base_dir / args.flattened)

    # NEW: hard guard for duplicated columns
    dup = df.columns[df.columns.duplicated()].tolist()
    if dup:
        raise ValueError(f"Duplicated columns in CSV: {dup[:30]}{' ...' if len(dup) > 30 else ''}")


    with open(base_dir / args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    label_col = list(config.get("LABEL_COL", []))
    if not label_col:
        raise ValueError("LABEL_COL is missing in config YAML.")
    drop_cols = list(config.get("DROP_COLS") or [])

    # [NEW][mode-provenance] basic sanity checks for DROP_COLS
    if args.mode_provenance is not None:
        if not drop_cols:
            raise ValueError("[mode-provenance] config.DROP_COLS is empty; need id columns listed there.")
        missing = [c for c in drop_cols if c not in df.columns]
        if missing:
            raise ValueError(f"[mode-provenance] DROP_COLS missing in CSV columns: {missing}")


    missing = [c for c in label_col if c not in df.columns]
    if missing:
        raise ValueError(f"LABEL_COL missing in data columns: {missing}")

    
    prov_th = args.provenance_threshold
    if prov_th is None:
        prov_th = int(config.get("PROVENANCE_THRESHOLD", 1) or 1)
    prov_th = int(prov_th)
    if prov_th < 1:
        raise ValueError("PROVENANCE_THRESHOLD must be >= 1")


    X = df.drop(columns=label_col+drop_cols, errors="raise")
    y = df[label_col].to_numpy().ravel()
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    classes = le.classes_
    is_multiclass = (len(classes) > 2)


    feature_cols = list(X.columns)

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = base_dir / model_path

    # treat given path as prefix
    candidates = [
        model_path.with_suffix(".tabpfn_fit"),
        model_path.with_suffix(".joblib"),
    ]

    real_path = None
    for p in candidates:
        if p.exists():
            real_path = p
            break

    if real_path is None:
        raise FileNotFoundError(
            f"Model not found. Tried: {[str(p) for p in candidates]}"
        )

    if real_path.suffix == ".tabpfn_fit":
        from tabpfn.model_loading import load_fitted_tabpfn_model
        model = load_fitted_tabpfn_model(str(real_path), device="auto")
    else:
        model = joblib.load(real_path)

    print("Loading model:", str(real_path))


    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not have predict_proba(). Expected a sklearn Pipeline classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    # print(len(y_test))
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    # AUC: binary vs multiclass
    if not is_multiclass:
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
    else:
        proba = model.predict_proba(X_test)  # (n, K)
        print(f"AUC(OVR, macro): {roc_auc_score(y_test, proba, multi_class='ovr', average='macro'):.4f}")


    # =========================================================
    # Background selection
    # =========================================================
    ic_payload = None
    cache_path = None
    constraints_payload = None

    if args.constraints_cache:
        cache_path = Path(args.constraints_cache)
        if not cache_path.is_absolute():
            cache_path = base_dir / cache_path
        if not cache_path.exists():
            raise FileNotFoundError(f"Constraints cache not found: {cache_path}")

        # LOAD ONCE
        with open(cache_path, "rb") as f:
            constraints_payload = pickle.load(f)
        if not isinstance(constraints_payload, dict):
            raise ValueError("constraints cache pkl must be a dict payload")

    if args.mode_domain or args.mode_denial:
        if cache_path is None:
            raise ValueError("--mode-domain/--mode-denial require --constraints-cache")

        ic_payload = constraints_payload
        repair_fn = None
        if args.mode_domain or args.mode_denial:
            if ic_payload is None:
                raise ValueError("--mode-domain/--mode-denial require --constraints-cache")
            repair_fn = _build_ic_repair_fn(ic_payload, mode_domain=args.mode_domain, 
                                            mode_denial=args.mode_denial, debug_enabled=bool(DEBUG),
                                            domain_numerical_mode=domain_num_mode,
                                            seed=SEED)


        enabled = []
        if args.mode_domain:
            enabled.append("--mode-domain")
        if args.mode_denial:
            enabled.append("--mode-denial")

    if bg_lut == 'train':
        bg = X_train.sample(min(args.background_n, len(X_train)), random_state=SEED)
    elif bg_lut == 'full':
        bg = X.sample(min(args.background_n, len(X)), random_state=SEED)
    ex = X_test.sample(min(args.explain_n, len(X_test)), random_state=SEED)

    # =========================================================
    # Closure cache (mode-bg only, unchanged behavior)
    # =========================================================
    closure_cache = None
    if args.mode_bg:
        if not args.constraints_cache:
            raise ValueError("--mode-bg requires --constraints-cache")

        closure_cache = ClosureCache(
            str(cache_path),
            enable_copy_write=bool(args.cache_copy),
            copy_out_path=args.cache_copy_path,
            payload=constraints_payload,
        )

    else:
        if args.mode_coalition_canon or args.mode_coalition_budget:
            # should be impossible due to earlier checks, but keep explicit
            raise ValueError("Coalition modes require --mode-bg.")

    def _build_forward_fds_from_cache(closure_cache: ClosureCache, drop_cols: List[str], feature_cols: List[str]):
        drop_set = set(str(c).strip() for c in drop_cols)
        feat_set = set(str(c).strip() for c in feature_cols)

        forward_by_id = defaultdict(set)

        # base_cache: key=lhs tuple, val=closure tuple
        for lhs, clo in closure_cache.base_cache.items():
            lhs_set = set(lhs)
            ids_in_lhs = list(lhs_set & drop_set)
            if not ids_in_lhs:
                continue

            # candidate rhs = closure(lhs) minus lhs, restricted to ML feature space
            for a in clo:
                if a in lhs_set:
                    continue
                if a not in feat_set:
                    continue
                for idc in ids_in_lhs:
                    forward_by_id[idc].add(a)

        return {k: sorted(v) for k, v in forward_by_id.items()}

    def build_idcol_to_rules_from_cache(closure_cache, drop_cols, feature_cols):
        """
        Build ID -> list of (req_feats, req_ids, rhs) rules
        from closure_cache.base_cache.

        req_feats : LHS features excluding IDs
        req_ids   : IDs appearing in LHS
        rhs       : single feature in closure(lhs) \ lhs
        """
        drop_set = set(str(c).strip() for c in drop_cols)
        feat_set = set(str(c).strip() for c in feature_cols)

        idcol_to_rules = defaultdict(list)

        for lhs, clo in closure_cache.base_cache.items():
            lhs_set = set(lhs)

            ids = [c for c in lhs_set if c in drop_set]
            if not ids:
                continue

            req_feats = [c for c in lhs_set if c in feat_set]

            for rhs in clo:
                if rhs in lhs_set:
                    continue
                if rhs not in feat_set:
                    continue

                for idc in ids:
                    idcol_to_rules[idc].append(
                        (frozenset(req_feats), ids, rhs)
                    )

        return dict(idcol_to_rules)

    idcol_to_rhs = {}
    if args.mode_provenance is not None:
        if not args.constraints_cache:
            raise ValueError("--mode-provenance requires --constraints-cache (to read FD rules).")
        if closure_cache is None:
            closure_cache = ClosureCache(
                str(cache_path),
                enable_copy_write=bool(args.cache_copy),
                copy_out_path=args.cache_copy_path,
                payload=constraints_payload,
            )
        if closure_cache is not None:
            closure_cache._dbg = _dbg_maker(bool(DEBUG), int(DBG_LIM), int(DBG_EVERY), prefix="[dbg]") if DEBUG else None

        idcol_to_rhs = _build_forward_fds_from_cache(closure_cache, drop_cols, feature_cols)
        if len(idcol_to_rhs) == 0:
            print("[mode-provenance] WARNING: no usable FDs found from drop_cols -> feature_cols; provenance will be no-op.")


    prov_index = None
    if args.mode_provenance and len(idcol_to_rhs) > 0:
        df_test_ref = df.loc[X_test.index].copy()

        idcol_to_rules = build_idcol_to_rules_from_cache(
            closure_cache,
            drop_cols,
            feature_cols,
        )

        prov_index = ProvenanceIndex(
            idcol_to_rhs=idcol_to_rhs,
            idcol_to_rules=idcol_to_rules,
            feature_cols=feature_cols,
            df_ref=df_test_ref,
            drop_cols=drop_cols,
        )

        if prov_index is not None:
            prov_index._dbg = _dbg_maker(bool(DEBUG), int(DBG_LIM), int(DBG_EVERY), prefix="[dbg]")

        if DEBUG and prov_index is not None:
            print("[prov] idcol_to_rhs keys:", list(idcol_to_rhs.keys()))
            # for k, v in idcol_to_rhs.items(): # does not align with |drop_cols| >=2, redundant!
            #     print(f"[prov] {k} -> {v[:12]}{'...' if len(v)>12 else ''}")
            # print("[prov] det_cols:", {k: v[:12] for k, v in prov_index.det_cols.items()})

        if len(prov_index.det_cols) == 0:
            print("[mode-provenance] WARNING: no determiners (rhs features) remain after filtering; provenance no-op.")
            prov_index = None

    ic_eval_ctx = {"in_synth": False, "total_n": None}


    def f_proba_pos(X_in: Any) -> np.ndarray:
        if not isinstance(X_in, pd.DataFrame):
            X_in = pd.DataFrame(X_in, columns=feature_cols)
        X_in = X_in.reindex(columns=feature_cols)
        return model.predict_proba(X_in)[:, 1]

    np.random.seed(SEED)
    random.seed(SEED)

    if DEBUG:
        print("[modes] "
              f"mode_bg={args.mode_bg} "
              f"coal_canon={args.mode_coalition_canon} "
              f"coal_budget={args.mode_coalition_budget} "
              f"provenance={args.mode_provenance} thr={prov_th} "
              f"IC_domain={args.mode_domain} IC_denial={args.mode_denial} "
              f"nsamples={args.nsamples} bg_n={bg.shape[0]} explain_n={ex.shape[0]} M={len(feature_cols)}")


    explainer = RelShapKernelExplainer(
        model=f_proba_pos,
        data=bg[feature_cols],
        feature_names=feature_cols,
        mode_bg=args.mode_bg,
        mode_coal_canon=args.mode_coalition_canon,
        mode_coal_budget=args.mode_coalition_budget,
        budget_max_skips=args.budget_max_skips,
        closure_cache=closure_cache,
        link="identity",
        provenance_mode=args.mode_provenance,  # None or string
        provenance_threshold=prov_th,
        provenance_index=prov_index,
        prov_strength=args.prov_strength,
        debug=int(DEBUG),
        debug_lim=int(DBG_LIM),
        debug_coal=bool(args.debug_coal),
        debug_coal_limit=int(args.debug_coal_limit),
        seed=SEED
    )

    
    explainer._ic_eval_ctx = ic_eval_ctx
    explainer._ic_repair_fn = repair_fn if (args.mode_domain or args.mode_denial) else None
    explainer._ic_feature_cols = feature_cols
    explainer._dbg_every = int(DBG_EVERY)
    explainer._dbg = _dbg_maker(bool(DEBUG), int(DBG_LIM), int(DBG_EVERY), prefix="[dbg]")
    explainer._ic_repair_fn = repair_fn if (args.mode_domain or args.mode_denial) else None
    explainer._ic_mode_domain = bool(args.mode_domain)
    explainer._ic_mode_denial = bool(args.mode_denial)


    
    SHAP_FN_MAP = {
        "kernel": explainer.shap_values,
        "mc": explainer.shap_values_mc,
        "leverage": explainer.shap_values_leverage,
    }

    try:
        shap_fn = SHAP_FN_MAP[args.base_mode]
    except KeyError:
        raise ValueError(f"Unknown base_mode: {args.base_mode}")
    
    t1 = time.time()
    sv = shap_fn(ex[feature_cols], nsamples=args.nsamples)
    t2 = time.time()
    if args.debug_coal:
        explainer.dump_coalitions()


    sv = np.asarray(sv)
    svdf = pd.DataFrame(sv, columns=feature_cols, index=ex.index)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(svdf.head(5))

    print("\n[runtime] shap_values_sec=", t2 - t1)

    (base_dir / "relshap").mkdir(parents=True, exist_ok=True)
    (base_dir / "metadata").mkdir(parents=True, exist_ok=True)

    out_tag = args.out

    csv_path = base_dir / "relshap" / f"relshap_{out_tag}.csv"
    svdf.to_csv(csv_path, index=False)
    print(f"[Saved] Shapley values -> {csv_path}")

    meta_path = base_dir / "metadata" / f"relshap_metadata_{out_tag}.json"
    with open(meta_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[Saved] Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
