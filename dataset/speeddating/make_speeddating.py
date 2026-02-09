import os
import re
import openml
import duckdb
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="make_speeddating"
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
RAW_CSV_PATH = os.path.join(args.base_dir, "original_data.csv")

dataset_id = 40536
dataset = openml.datasets.get_dataset(dataset_id)

df, _, _, _ = dataset.get_data(
    dataset_format="dataframe"
)

df.to_csv(RAW_CSV_PATH, index=False)

# -----------------------
# Load CSV
# -----------------------

# y = df["class"].to_numpy()          # label은 DB 밖
df_nolabel = df

# 오타 존중. 양뱡향 데이터 생성 (사람 2명 관점에서 모두, 그래서 데이터 개수 짝수)
# wave안에서 모든 사람이 교류하는건 아니고 interaction에 아이디가 있는 사람들만 교류

# 한 개인에 대해서 고정된 값
self_attributes = [
    # demographics / background
    "gender", "age", "race", "field",

    # values / importance
    "importance_same_race", "importance_same_religion",
    # bucketized
    "d_importance_same_race", "d_importance_same_religion",

    # self ratings
    "attractive", "sincere", "intelligence", "funny", "ambition",
    # bucketized
    "d_attractive", "d_sincere", "d_intelligence", "d_funny", "d_ambition",

    # interests
    "sports", "tvsports", "exercise", "dining", "museums", "art", "hiking",
    "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts",
    "music", "shopping", "yoga",
    # bucketized
    "d_sports", "d_tvsports", "d_exercise", "d_dining", "d_museums", "d_art",
    "d_hiking", "d_gaming", "d_clubbing", "d_reading", "d_tv", "d_theater",
    "d_movies", "d_concerts", "d_music", "d_shopping", "d_yoga",

    # expectations / beliefs / decisions
    "expected_happy_with_sd_people", "expected_num_interested_in_me",
    "expected_num_matches", 
    # bucketized
    "d_expected_happy_with_sd_people", "d_expected_num_interested_in_me",
    "d_expected_num_matches",

    # what I look for in a partner
    "attractive_important", "sincere_important", "intellicence_important",
    "funny_important", "ambtition_important", "shared_interests_important",
    # bucketized
    "d_attractive_important", "d_sincere_important", "d_intellicence_important",
    "d_funny_important", "d_ambtition_important", "d_shared_interests_important",

    'has_null', 
    # venue 파트너는 같은 wave내에서만 고를 수 있음
    'wave'
]


# 아래의 속성들은 개인 - 파트너 1-1 crossover:
# 개인 -> 파트너
#     what I look for in a partner
#     "attractive_important", "sincere_important", "intellicence_important",
#     "funny_important", "ambtition_important", "shared_interests_important",
#     # bucketized
#     "d_attractive_important", "d_sincere_important", "d_intellicence_important",
#     "d_funny_important", "d_ambtition_important", "d_shared_interests_important",

#     # demographics
#     "age", "race"

#     # my ratings of partner
#     "attractive_partner", "sincere_partner", "intelligence_partner",
#     "funny_partner", "ambition_partner", "shared_interests_partner",

#     # bucketized
#     "d_attractive_partner", "d_sincere_partner", "d_intelligence_partner",
#     "d_funny_partner", "d_ambition_partner", "d_shared_interests_partner",
# 
# 파트너 -> 개인
#     partner's preference criteria
#     "pref_o_attractive", "pref_o_sincere", "pref_o_intelligence",
#     "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests",
#     # bucketized
#     "d_pref_o_attractive", "d_pref_o_sincere", "d_pref_o_intelligence",
#     "d_pref_o_funny", "d_pref_o_ambitious", "d_pref_o_shared_interests",

#     # demographics (partner's)
#     "age_o", "race_o",

#     # partner's ratings of me
#     "attractive_o", "sinsere_o", "intelligence_o", "funny_o",
#     "ambitous_o", "shared_interests_o",
#     # bucketized
#     "d_attractive_o", "d_sinsere_o", "d_intelligence_o", "d_funny_o",
#     "d_ambitous_o", "d_shared_interests_o",

# 파트너에 대해서 고정된 값
partner_attributes = [
    # partner's preference criteria
    "pref_o_attractive", "pref_o_sincere", "pref_o_intelligence",
    "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests",
    # bucketized
    "d_pref_o_attractive", "d_pref_o_sincere", "d_pref_o_intelligence",
    "d_pref_o_funny", "d_pref_o_ambitious", "d_pref_o_shared_interests",

    # demographics (partner's)
    "age_o", "race_o",
]

# 한 개인이 파트너마다 다르게 점수를 주는 속성들
individual2partner_attributes = [
    # my ratings of partner
    "attractive_partner", "sincere_partner", "intelligence_partner",
    "funny_partner", "ambition_partner", "shared_interests_partner",
    "like", "guess_prob_liked",

    # bucketized
    "d_attractive_partner", "d_sincere_partner", "d_intelligence_partner",
    "d_funny_partner", "d_ambition_partner", "d_shared_interests_partner",
     "d_like", "d_guess_prob_liked",
]
# 한 파트너가 개인마다 다르게 점수를 주는 속성들
partner2individual_attributes = [
    # partner's ratings of me
    "attractive_o", "sinsere_o", "intelligence_o", "funny_o",
    "ambitous_o", "shared_interests_o",
    # bucketized
    "d_attractive_o", "d_sinsere_o", "d_intelligence_o", "d_funny_o",
    "d_ambitous_o", "d_shared_interests_o",
]

# 개인과 파트너가 공유하는 속성들
shared_attributes = [
    # difference between ages
    "d_age", 
    # bucketized
    "d_d_age", 
    # same race or not
    "samerace",
    # 위 3개는 계산 가능하지만
    # 아래 3개는 계산 불가능하고 값으로서만 존재
    # interests correlate or not
    "interests_correlate",
    # bucketized
    "d_interests_correlate",
    # interaction history
    "met",   
]

# 1) 모든 기대 컬럼 모으기
expected_cols = (
    self_attributes
    + partner_attributes
    + partner2individual_attributes
    + individual2partner_attributes
    + shared_attributes
)

expected_set = set(expected_cols)
df_cols_set = set(df_nolabel.columns)

# 2) df에 없는 컬럼 찾기
missing_cols = sorted(df_cols_set - expected_set)

# 3) 결과 출력
if not missing_cols:
    print("All expected columns are present in df.")
else:
    print(f"Missing {len(missing_cols)} columns:")
    for c in missing_cols:
        print("  -", c)

# ============================================================
# Self identity attributes (STRICT, NON-BUCKETIZED)
# ============================================================
self_attributes_unique = [
    # demographics / background
    "gender", "age", "race", "field",

    # self ratings
    "attractive", "sincere", "intelligence", "funny", "ambition",

    # interests
    "sports",

    # expectations / beliefs / decisions
    "expected_happy_with_sd_people",
    "expected_num_interested_in_me",
    "expected_num_matches",
]

import numpy as np
import pandas as pd


def _norm_val_strict(x):
    """No rounding, no tolerance. NaN -> None. Keep values as-is (normalized types only)."""
    if pd.isna(x):
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)  # <-- NO rounding
    return str(x)


def _row_signature_strict(row: pd.Series, cols):
    return tuple(_norm_val_strict(row.get(c, None)) for c in cols)


def _values_compatible_strict(values):
    """
    True if all non-null values are EXACTLY identical (nulls ignored).
    """
    nn = [v for v in values if v is not None]
    if len(nn) <= 1:
        return True
    base = nn[0]
    return all(v == base for v in nn[1:])


def validate_self_identity_strict_no_tolerance(
    df_nolabel: pd.DataFrame,
    self_cols: list,
):
    """
    Strict self-identity validation (NO tolerance).

    Checks per wave:
      1) (wave, self_signature) rows must have identical self attributes (exact)
      2) within a person (self_id), self attributes never change across rows (exact)

    Raises ValueError if any violation exists.
    """

    if "wave" not in df_nolabel.columns:
        raise ValueError("df_nolabel must contain column 'wave'")

    df = df_nolabel.copy()

    # only keep columns that actually exist
    self_cols = [c for c in self_cols if c in df.columns]
    if not self_cols:
        raise ValueError("No self identity columns exist in dataframe")

    # build signature (exact)
    df["_self_sig"] = df.apply(lambda r: _row_signature_strict(r, self_cols), axis=1)

    # factorize per wave
    df["self_id"] = pd.factorize(
        pd.MultiIndex.from_arrays([df["wave"].to_numpy(), df["_self_sig"].to_numpy()])
    )[0]

    bad_signature = []
    bad_self_id = []

    # --------------------------------------------------
    # A) signature collision check (exact)
    # --------------------------------------------------
    for (w, sig), g in df.groupby(["wave", "_self_sig"], sort=False):
        if len(g) <= 1:
            continue

        for c in self_cols:
            if not _values_compatible_strict(g[c].map(_norm_val_strict).tolist()):
                bad_signature.append({
                    "wave": int(w),
                    "n_rows": len(g),
                    "column": c,
                })
                break

    # --------------------------------------------------
    # B) per-person stability check (exact)
    # --------------------------------------------------
    for (w, sid), g in df.groupby(["wave", "self_id"], sort=False):
        if len(g) <= 1:
            continue

        for c in self_cols:
            if not _values_compatible_strict(g[c].map(_norm_val_strict).tolist()):
                bad_self_id.append({
                    "wave": int(w),
                    "self_id": int(sid),
                    "n_rows": len(g),
                    "column": c,
                })
                break

    if bad_signature or bad_self_id:
        raise ValueError(
            "Self identity NOT unique (NO tolerance)\n"
            f"signature collisions: {len(bad_signature)}\n"
            f"self-id instability: {len(bad_self_id)}\n"
            f"examples(sig): {bad_signature[:3]}\n"
            f"examples(id): {bad_self_id[:3]}"
        )

    print("Self identity is STRICTLY unique (NO tolerance)")
    return df[["wave", "self_id"] + self_cols]

df_id = validate_self_identity_strict_no_tolerance(
    df_nolabel,
    self_attributes_unique,
)
df_id = pd.concat([df_id, df_nolabel[df_nolabel.columns.difference(df_id.columns)]], axis=1)




def _close_or_both_nan_exact(a, b):
    """
    Exact match:
      - True if both are NaN
      - False if only one is NaN
      - Otherwise, exact equality (==) for all types (including numbers)
    """
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return a == b


def _col_equal_exact(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """
    Vectorized-ish exact equality with NaN==NaN allowed.
    Returns a boolean mask aligned to s1/s2.
    """
    a = s1.to_numpy()
    b = s2.to_numpy()
    out = np.zeros(len(s1), dtype=bool)
    for i in range(len(out)):
        out[i] = _close_or_both_nan_exact(a[i], b[i])
    return pd.Series(out, index=s1.index)


def find_partner_rows_by_reciprocal_checks_no_tolerance(
    df_id: pd.DataFrame,
    wave_col="wave",
    self_id_col="self_id",
    require_o_backmatch=True,  # If True: enforce full reciprocal swap including age_o/race_o
):
    """
    For each directed row i, find candidate partner rows j in the same wave and
    diagnose whether the event is identifiable (0 / 1 / many candidates).

    Matching rules (exact, no tolerance):
      - Same wave
      - Join anchor: left(age_o, race_o) == right(age, race)
      - If require_o_backmatch=True: also enforce right(age_o, race_o) == left(age, race)
      - interests_correlate must match (NaN==NaN allowed)
      - Reciprocal ratings must match:
          row[<trait>_partner] == partner[<trait>_o]
          row[<trait>_o]       == partner[<trait>_partner]
        Note: keep the original typo column names (sinsere_o, ambitous_o) as-is.
    """

    df = df_id.copy()

    # Create a row id if missing (useful for debugging/reporting)
    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df), dtype=int)

    # Required base columns
    base_self = ["age", "race"]
    base_partner = ["age_o", "race_o"]

    # Columns expected to match on both sides (exact)
    same_on_both = ["interests_correlate", "met", "d_age", "d_d_age"]

    # Reciprocal swap rules: left[a_col] must equal right[b_col]
    swap_pairs = [
        ("attractive_partner", "attractive_o"),
        ("attractive_o",       "attractive_partner"),

        ("sincere_partner",    "sinsere_o"),     # keep typo as-is
        ("sinsere_o",          "sincere_partner"),

        ("intelligence_partner", "intelligence_o"),
        ("intelligence_o",       "intelligence_partner"),

        ("funny_partner",      "funny_o"),
        ("funny_o",            "funny_partner"),

        ("ambition_partner",   "ambitous_o"),    # keep typo as-is
        ("ambitous_o",         "ambition_partner"),

        ("shared_interests_partner", "shared_interests_o"),
        ("shared_interests_o",       "shared_interests_partner"),
    ]

    # Keep only swap pairs whose columns exist in df
    def _keep_existing(pairs):
        out = []
        for a, b in pairs:
            if a in df.columns and b in df.columns:
                out.append((a, b))
        return out

    swap_pairs = _keep_existing(swap_pairs)

    # Validate required columns exist
    for c in base_self + base_partner + same_on_both:
        if c not in df.columns:
            raise KeyError(f"Needed column missing: {c}")
    if len(swap_pairs) == 0:
        raise ValueError("No swap_pairs columns found in df_id. Check your column names.")

    # Build left/right frames for the merge
    left_cols = [wave_col, self_id_col, "row_id"] + base_self + base_partner + same_on_both + [a for a, _ in swap_pairs]
    right_cols = [wave_col, self_id_col, "row_id"] + base_self + base_partner + same_on_both + [b for _, b in swap_pairs]

    left = df[left_cols].copy()
    right = df[right_cols].copy()

    # Anchor join:
    # left row i's (age_o, race_o) must match right row j's (age, race) in the same wave
    merged = left.merge(
        right,
        how="left",
        left_on=[wave_col, "age_o", "race_o"],
        right_on=[wave_col, "age", "race"],
        suffixes=("", "_p"),
        indicator=False,
    )

    # Remove self-matching (same person id); adjust if you want to allow it
    merged = merged[merged[self_id_col] != merged[f"{self_id_col}_p"]]

    # If requested, enforce full reciprocal backmatch:
    # right(age_o, race_o) must match left(age, race)
    if require_o_backmatch:
        ok_back = (
            _col_equal_exact(merged["age"], merged["age_o_p"])
            & _col_equal_exact(merged["race"], merged["race_o_p"])
        )
        merged = merged[ok_back]

    # interests_correlate must match (exact, NaN==NaN allowed)
    ok_ic = _col_equal_exact(merged["interests_correlate"], merged["interests_correlate_p"])
    merged = merged[ok_ic]

    # Apply all reciprocal swap constraints (exact)
    for a_col, b_col in swap_pairs:
        b_col_p = f"{b_col}_p"
        if b_col_p not in merged.columns:
            continue
        ok = _col_equal_exact(merged[a_col], merged[b_col_p])
        merged = merged[ok]

    # Aggregate candidates per left row (row_id)
    out = (
        merged.groupby(["row_id"], sort=False)
        .agg(
            candidate_partner_row_ids=("row_id_p", lambda s: list(pd.unique(s.dropna().astype(int)))),
            candidate_partner_self_ids=(f"{self_id_col}_p", lambda s: list(pd.unique(s.dropna().astype(int)))),
            n_candidates=("row_id_p", lambda s: int(s.dropna().shape[0])),
        )
        .reset_index()
    )

    # Include rows with zero matches
    all_rows = df[["row_id", wave_col, self_id_col]].drop_duplicates()
    out = all_rows.merge(out, on="row_id", how="left")

    out["candidate_partner_row_ids"] = out["candidate_partner_row_ids"].apply(lambda x: x if isinstance(x, list) else [])
    out["candidate_partner_self_ids"] = out["candidate_partner_self_ids"].apply(lambda x: x if isinstance(x, list) else [])
    out["n_candidates"] = out["n_candidates"].fillna(0).astype(int)

    # Summary and convenience flags
    summary = out["n_candidates"].value_counts().sort_index()
    out["is_unique_partner_row"] = (out["n_candidates"] == 1)
    out["is_ambiguous_partner_row"] = (out["n_candidates"] > 1)
    out["is_missing_partner_row"] = (out["n_candidates"] == 0)

    return out, summary


# -------------------------
# Example usage
# -------------------------

df_id = df_id.copy()
if "row_id" not in df_id.columns:
    df_id["row_id"] = np.arange(len(df_id), dtype=int)

result, summary = find_partner_rows_by_reciprocal_checks_no_tolerance(df_id)

missing_ids = result.loc[result["n_candidates"] == 0, "row_id"].tolist()
df_id_filtered = df_id.loc[~df_id["row_id"].isin(missing_ids)].copy()

def _single_or_na(x):
    if not isinstance(x, list) or len(x) == 0:
        return pd.NA
    return int(x[0])

result_single = result.copy()
result_single["partner_row_id"]  = result_single["candidate_partner_row_ids"].apply(_single_or_na).astype("Int64")
result_single["partner_id"] = result_single["candidate_partner_self_ids"].apply(_single_or_na).astype("Int64")
result_single = result_single[["row_id", "partner_row_id", "partner_id"]]

df_id_final = df_id_filtered.merge(
    result_single,
    on="row_id",
    how="left",
    validate="one_to_one"  # safety check: one row -> one partner
)

# should never be NaN after filtering
assert df_id_final["partner_row_id"].notna().all()
assert df_id_final["partner_id"].notna().all()

# no remaining missing partner cases
assert (df_id_final["row_id"].isin(missing_ids)).sum() == 0

front = ["row_id", "self_id", "partner_row_id", "partner_id", "wave"]
rest  = [c for c in df_id.columns if c not in front]

df_id_final = df_id_final[front + rest]

print("missing_ids: ", missing_ids)
df_id_final = df_id_final.drop(['partner_row_id'],axis=1)
print(df_id_final)

cols_keep = [
    "wave",
    "self_id",

    "gender",
    "age",
    "race",
    "field",

    "importance_same_race",
    "importance_same_religion",

    "attractive",
    "sincere",
    "intelligence",
    "funny",
    "ambition",

    "sports",
    "tvsports",
    "exercise",
    "dining",
    "museums",
    "art",
    "hiking",
    "gaming",
    "clubbing",
    "reading",
    "tv",
    "theater",
    "movies",
    "concerts",
    "music",
    "shopping",
    "yoga",

    "expected_happy_with_sd_people",
    "expected_num_interested_in_me",
    "expected_num_matches",

    "attractive_important",
    "sincere_important",
    "intellicence_important",   # (원본 오타 유지)
    "funny_important",
    "ambtition_important",      # (원본 오타 유지)
    "shared_interests_important",

    "has_null",

    "pref_o_attractive","pref_o_sincere","pref_o_intelligence",
    "pref_o_funny","pref_o_ambitious","pref_o_shared_interests",
    "d_pref_o_attractive","d_pref_o_sincere","d_pref_o_intelligence",
    "d_pref_o_funny","d_pref_o_ambitious","d_pref_o_shared_interests",

]

df_person = (
    df_id_final
    .loc[:, cols_keep]                 # 컬럼 선택
    .drop_duplicates(
        subset=["wave", "self_id"],    # 개인 단위
        keep="first"                   # 동일 인물 중 첫 row 사용
    )
    .reset_index(drop=True)
)

# (wave, self_id) 유일성 확인
assert not df_person.duplicated(["wave", "self_id"]).any()



if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

con = duckdb.connect(DB_PATH)

con.register("raw_df_person", df_person)
con.execute("DROP TABLE IF EXISTS raw_speeddating_person")
con.execute("CREATE TABLE raw_speeddating_person AS SELECT * FROM raw_df_person")

con.register("raw_df", df_id_final)
con.execute("DROP TABLE IF EXISTS raw_speeddating")
con.execute("CREATE TABLE raw_speeddating AS SELECT * FROM raw_df")

import re
import duckdb
import pandas as pd

# ============================================================
# Helpers
# ============================================================

def cols_of(con, rel: str):
    # works for tables/views
    return [r[1] for r in con.execute(f"PRAGMA table_info('{rel}')").fetchall()]

def assert_has_cols(con, rel: str, required, where=""):
    existing = set(cols_of(con, rel))
    missing = [c for c in required if c not in existing]
    if missing:
        hint = f" ({where})" if where else ""
        raise ValueError(f"❌ {rel}{hint} is missing columns: {missing}")

RESERVED = {"like"}

def qident(x: str) -> str:
    if x.lower() in RESERVED:
        return f'"{x}"'
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", x):
        return x
    return f'"{x}"'



# ============================================================
# 0) Define “semantic groups” (your taxonomy)
# ============================================================

# Person-fixed (self) attributes: (wave, self_id) -> these
person_fixed = [
    "gender", "age", "race", "field",
    "importance_same_race", "importance_same_religion",
    "attractive", "sincere", "intelligence", "funny", "ambition",
    "sports", "tvsports", "exercise", "dining", "museums", "art", "hiking",
    "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts",
    "music", "shopping", "yoga",
    "expected_happy_with_sd_people", "expected_num_interested_in_me",
    "expected_num_matches",
    "attractive_important", "sincere_important", "intellicence_important",
    "funny_important", "ambtition_important", "shared_interests_important",
]

# partner-fixed attributes are ALSO person-fixed (just fetched via partner_id join)
partner_pref = [
    "pref_o_attractive", "pref_o_sincere", "pref_o_intelligence",
    "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests",
]

# Interaction attributes (directed edge self->partner)
interaction_cols = [
    "interests_correlate", "met",

    "attractive_partner", "sincere_partner", "intelligence_partner",
    "funny_partner", "ambition_partner", "shared_interests_partner",

    "attractive_o", "sinsere_o", "intelligence_o",
    "funny_o", "ambitous_o", "shared_interests_o",

    "like", "guess_prob_liked",

    "has_null"
]

# SELECT 출력 순서와 동일하게!
PERSON_INSERT_COLS = ["wave", "self_id"] + person_fixed + partner_pref

# 중복 제거(혹시 모르게) - 순서 유지
seen = set()
PERSON_INSERT_COLS = [c for c in PERSON_INSERT_COLS if not (c in seen or seen.add(c))]

cols_sql = ", ".join(qident(c) for c in PERSON_INSERT_COLS)

# ============================================================
# 1) Rebuild PERSON + INTERACTION (clean, robust)
# ============================================================

# --- sanity: raw tables must have needed id columns ---
assert_has_cols(con, "raw_speeddating", ["wave", "self_id", "partner_id"])
assert_has_cols(con, "raw_speeddating_person", ["wave", "self_id"])

raw_person_cols = set(cols_of(con, "raw_speeddating_person"))
raw_inter_cols  = set(cols_of(con, "raw_speeddating"))

# We will populate Person from raw_speeddating_person.
# If pref_o_* are NOT in raw_speeddating_person, we will derive them from raw_speeddating
# by grouping on (wave, self_id) and taking ANY_VALUE (and validate uniqueness afterwards).
need_from_inter = [c for c in partner_pref if c in raw_inter_cols]

# --- Create Person table: must include partner_pref so partner join works ---
con.execute("DROP TABLE IF EXISTS Person;")
con.execute(f"""
CREATE TABLE Person (
  wave INTEGER,
  self_id INTEGER,

  -- fixed demographics / background
  gender VARCHAR,
  age DOUBLE,
  race VARCHAR,
  field VARCHAR,

  importance_same_race DOUBLE,
  importance_same_religion DOUBLE,

  attractive DOUBLE,
  sincere DOUBLE,
  intelligence DOUBLE,
  funny DOUBLE,
  ambition DOUBLE,

  sports DOUBLE,
  tvsports DOUBLE,
  exercise DOUBLE,
  dining DOUBLE,
  museums DOUBLE,
  art DOUBLE,
  hiking DOUBLE,
  gaming DOUBLE,
  clubbing DOUBLE,
  reading DOUBLE,
  tv DOUBLE,
  theater DOUBLE,
  movies DOUBLE,
  concerts DOUBLE,
  music DOUBLE,
  shopping DOUBLE,
  yoga DOUBLE,

  expected_happy_with_sd_people DOUBLE,
  expected_num_interested_in_me DOUBLE,
  expected_num_matches DOUBLE,

  attractive_important DOUBLE,
  sincere_important DOUBLE,
  intellicence_important DOUBLE,
  funny_important DOUBLE,
  ambtition_important DOUBLE,
  shared_interests_important DOUBLE,

  pref_o_attractive DOUBLE,
  pref_o_sincere DOUBLE,
  pref_o_intelligence DOUBLE,
  pref_o_funny DOUBLE,
  pref_o_ambitious DOUBLE,
  pref_o_shared_interests DOUBLE,

  PRIMARY KEY (wave, self_id)
);
""")

# Build SELECT for Person insert.
# For pref_o_*: if present in raw_person -> take from there
# else if present in raw_interaction -> coalesce from derived aggregate
person_select_list = [
    "rp.wave",
    "rp.self_id",
]

def sel_person(col):
    if col == "like":
        return f"rp.{qident(col)} AS {qident(col)}" if col in raw_person_cols else f"NULL AS {qident(col)}"
    return f"rp.{col} AS {col}" if col in raw_person_cols else f"NULL AS {col}"

# all fixed cols except pref_o_*
for c in person_fixed:
    person_select_list.append(sel_person(c))

# pref_o_* columns: rp.col if exists else agg.col if derived else NULL
# agg alias: ai (aggregated interaction by (wave, self_id))
for c in partner_pref:
    if c in need_from_inter:
        person_select_list.append(f"ai.{c} AS {c}")
    else:
        person_select_list.append(f"NULL AS {c}")


# CTE ai only if needed
if need_from_inter:
    ai_select = ",\n      ".join([f"ANY_VALUE({c}) AS {c}" for c in need_from_inter])
    person_insert_sql = f"""
    INSERT INTO Person ({cols_sql})
    WITH ai AS (
      SELECT
        wave,
        partner_id,
        {ai_select}
      FROM raw_speeddating
      GROUP BY wave, partner_id
    )
    SELECT DISTINCT
      {",      ".join(person_select_list)}
    FROM raw_speeddating_person rp
    LEFT JOIN ai
      ON ai.wave = rp.wave AND ai.partner_id = rp.self_id
    ;
    """
else:
    person_insert_sql = f"""
    INSERT INTO Person ({cols_sql})
    SELECT DISTINCT
      {",      ".join(person_select_list)}
    FROM raw_speeddating_person rp
    ;
    """

con.execute(person_insert_sql)

# Optional: validate that pref_o_* derived from raw_speeddating is consistent per person
# (if any person has >1 distinct non-null value, that's a real data issue)
if need_from_inter:
    for c in need_from_inter:
        bad = con.execute(f"""
        SELECT COUNT(*) FROM (
          SELECT wave, partner_id
          FROM raw_speeddating
          WHERE {c} IS NOT NULL
          GROUP BY wave, partner_id
          HAVING COUNT(DISTINCT {c}) > 1
        ) t
        """).fetchone()[0]
        if bad:
            print(f"⚠️ WARNING: {bad} (wave,partner_id) have multiple distinct values for {c} in raw_speeddating")


# --- Create Interaction table (edge attributes only, plus ids) ---
con.execute("DROP TABLE IF EXISTS Interaction;")
con.execute("""
CREATE TABLE Interaction (
  row_id INTEGER,          
  wave INTEGER,
  self_id INTEGER,
  partner_id INTEGER,

  interests_correlate DOUBLE,
  met DOUBLE,

  attractive_partner DOUBLE,
  sincere_partner DOUBLE,
  intelligence_partner DOUBLE,
  funny_partner DOUBLE,
  ambition_partner DOUBLE,
  shared_interests_partner DOUBLE,

  attractive_o DOUBLE,
  sinsere_o DOUBLE,
  intelligence_o DOUBLE,
  funny_o DOUBLE,
  ambitous_o DOUBLE,
  shared_interests_o DOUBLE,
            
  "like" DOUBLE,
  guess_prob_liked DOUBLE,

  has_null INTEGER,

  PRIMARY KEY (wave, self_id, partner_id),

  FOREIGN KEY (wave, self_id) REFERENCES Person(wave, self_id),
  FOREIGN KEY (wave, partner_id) REFERENCES Person(wave, self_id)
);
""")

# Make sure raw_speeddating has all columns we need for Interaction (otherwise fail fast)
assert_has_cols(con, "raw_speeddating", ["wave", "self_id", "partner_id"] + interaction_cols)

def qident(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'

interaction_cols_sql = ", ".join(qident(c) for c in interaction_cols)

con.execute(f"""
INSERT INTO Interaction
SELECT
  row_id,
  wave,
  self_id,
  partner_id,
  {interaction_cols_sql}
FROM raw_speeddating;
""")


# ============================================================
# 2) BucketRule recreate + populate (from df_id_final)
# ============================================================

_INTERVAL_RE = re.compile(r'^\s*([\[\(])\s*([-+]?\d*\.?\d+)\s*-\s*([-+]?\d*\.?\d+)\s*([\]\)])\s*$')

def _parse_interval_label(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x)
    m = _INTERVAL_RE.match(s)
    if not m:
        return None
    left, lo, hi, right = m.groups()
    return (float(lo), float(hi), left == "[", right == "]")

_NULL_LIKE = {"", "nan", "NaN", "None", "NULL", "null"}

def _is_null_like_label(x) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s in _NULL_LIKE

def build_bucket_rules_from_df(df: pd.DataFrame, d_prefix="d_") -> pd.DataFrame:
    d_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(d_prefix)]
    rules = []

    def _base_for_dcol(dcol: str) -> str:
        """
        기본: d_{base} -> base
        예외: d_d_age -> d_age  (d_age는 나이차이 파생피처)
        또한 d_age는 base=age로 취급하면 안 되므로 rules 생성 대상에서 제외.
        베이스값이 null이라고 해도 뭔가 매핑되기도 함.
        """
        if dcol == "d_d_age":
            return "d_age"
        if dcol == "d_age":
            return None  # 스킵 (age로 매핑 금지)
        return dcol[len(d_prefix):]

    for dcol in d_cols:
        base = _base_for_dcol(dcol)
        if base is None:
            continue  # d_age는 여기서 제외

        if base not in df.columns:
            continue

        # ===== NEW: base NULL -> bucket_label 규칙 (여기서도 d_age는 이미 스킵됨) =====
        base_s = df[base]
        d_s = df[dcol].astype("object")

        base_is_null = base_s.isna()
        d_is_not_null_like = ~d_s.map(_is_null_like_label)

        null_rows = df.loc[base_is_null & d_is_not_null_like, dcol]
        if len(null_rows) > 0:
            vc = null_rows.astype(str).value_counts(dropna=True)
            if vc.shape[0] == 1:
                lab = vc.index[0]
            else:
                lab = vc.index[0]
                print(f"⚠️ WARNING: base={base} is NULL but {dcol} has multiple labels. "
                      f"Using mode={lab}. top5={vc.head(5).to_dict()}")

            rules.append({
                "bucket_col": dcol,
                "base_col": base,
                "bucket_label": lab,
                "lo": None,
                "hi": None,
                "lo_inclusive": None,
                "hi_inclusive": None,
            })
        # ===== /NEW =====

        raw_labels = pd.Series(pd.unique(df[dcol]), dtype="object")

        # 1) NULL/NaN/빈문자 라벨 규칙 추가
        null_like_labels = [x for x in raw_labels if _is_null_like_label(x)]
        for lab in null_like_labels:
            lab_str = "" if lab is None else str(lab)
            rules.append({
                "bucket_col": dcol,
                "base_col": base,
                "bucket_label": lab_str,
                "lo": None,
                "hi": None,
                "lo_inclusive": None,
                "hi_inclusive": None,
            })

        # 2) 구간 라벨 규칙 추가
        labels_str = raw_labels.map(lambda x: None if _is_null_like_label(x) else str(x))
        parsed = labels_str.map(_parse_interval_label)

        ok = parsed.notna()
        if int(ok.sum()) == 0:
            continue

        for lab, p in zip(labels_str[ok], parsed[ok]):
            lo, hi, li, hi_i = p
            rules.append({
                "bucket_col": dcol,
                "base_col": base,
                "bucket_label": lab,
                "lo": float(lo),
                "hi": float(hi),
                "lo_inclusive": bool(li),
                "hi_inclusive": bool(hi_i),
            })

    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.drop_duplicates(
            subset=["bucket_col", "base_col", "bucket_label", "lo", "hi", "lo_inclusive", "hi_inclusive"]
        ).reset_index(drop=True)
    return rules_df



rules_df = build_bucket_rules_from_df(df_id_final)

con.execute("DROP TABLE IF EXISTS BucketRule;")
con.execute("""
CREATE TABLE BucketRule (
  rule_id BIGINT,
  bucket_col VARCHAR,
  base_col VARCHAR,
  bucket_label VARCHAR,
  lo DOUBLE,
  hi DOUBLE,
  lo_inclusive BOOLEAN,
  hi_inclusive BOOLEAN,
  PRIMARY KEY (rule_id)
);
""")
con.register("rules_df", rules_df)
con.execute("""
INSERT INTO BucketRule
SELECT
  row_number() OVER () AS rule_id,
  bucket_col, base_col, bucket_label, lo, hi, lo_inclusive, hi_inclusive
FROM rules_df;
""")

# print("BucketRule rows:", con.execute("SELECT COUNT(*) FROM BucketRule").fetchone()[0])


# ============================================================
# 3) Views: Person_with_buckets / Interaction_with_features
# -- null-rule과 interval-rule 둘 다 후보로 올려놓고,
# -- 아래 ORDER BY에서 "포함/가까움"으로 최종 1개를 고른다.
# -- 0) x가 NULL/NaN이면 null-rule을 최우선으로
#  -- 1) x가 숫자면: "포함되는 구간" 우선 (거리 0)
# --    포함 안 되면: 구간까지의 거리 최소(closest)
# -- 2) 동점이면 큰 하단 선호 (ASC)

# ============================================================

def _bucket_expr(x_expr: str, bucket_col: str, base_col: str) -> str:
    raw = f"try_cast({x_expr} AS DOUBLE)"

    if bucket_col == "d_interests_correlate":
        x_num = raw
    else:
        x_num = f"ceil({raw})"

    return f"""
    (
      SELECT br.bucket_label
      FROM BucketRule br
      WHERE br.bucket_col='{bucket_col}'
        AND br.base_col='{base_col}'
        AND (
          
          (br.lo IS NULL AND br.hi IS NULL)
          OR
          (br.lo IS NOT NULL AND br.hi IS NOT NULL)
        )
      ORDER BY
        
        CASE
          WHEN ({x_expr} IS NULL OR {raw} IS NULL OR isnan({raw}))
          THEN CASE WHEN (br.lo IS NULL AND br.hi IS NULL) THEN 0 ELSE 1 END
          ELSE 1
        END ASC,

        CASE
          WHEN (br.lo IS NULL AND br.hi IS NULL) THEN 1e308
          WHEN ({raw} IS NULL OR isnan({raw})) THEN 1e308
          WHEN {x_num} < br.lo THEN (br.lo - {x_num})
          WHEN {x_num} > br.hi THEN ({x_num} - br.hi)
          ELSE 0
        END ASC,

        br.lo ASC,
        br.bucket_label ASC
      LIMIT 1
    )
    """.strip()





pairs = con.execute("""
SELECT DISTINCT bucket_col, base_col
FROM BucketRule
ORDER BY bucket_col, base_col
""").fetchall()

person_cols = set(cols_of(con, "Person"))
inter_cols  = set(cols_of(con, "Interaction"))

# bucket columns that can be computed from Person fields directly
person_bucket_pairs = [(b, base) for (b, base) in pairs if base in person_cols]

# bucket columns that can be computed from Interaction fields OR derived feature d_age
derived_ok = {"d_age"}  # e.g., d_d_age has base_col='d_age'
inter_bucket_pairs = [(b, base) for (b, base) in pairs if (base in inter_cols) or (base in derived_ok)]

# Person_with_buckets
pwb_select = ["p.*"]
for bucket_col, base_col in person_bucket_pairs:
    pwb_select.append(f"{_bucket_expr(f'p.{qident(base_col)}', bucket_col, base_col)} AS {qident(bucket_col)}")

person_view_sql = (
    "CREATE OR REPLACE VIEW Person_with_buckets AS\n"
    "SELECT\n  " + ",\n  ".join(pwb_select) + "\n"
    "FROM Person p;"
)
con.execute(person_view_sql)

# Interaction_with_features
iwf_select = [
    "i.*",
    """
    CASE
        WHEN ps.age IS NULL AND pp.age IS NULL THEN NULL
        WHEN ps.age IS NULL THEN pp.age
        WHEN pp.age IS NULL THEN ps.age
        ELSE ABS(ps.age - pp.age)
    END AS d_age

    """,
    """
    CASE
        WHEN ps.race IS NULL OR pp.race IS NULL THEN 0
        WHEN ps.race = pp.race THEN 1
        ELSE 0
    END AS samerace
    """
]

for bucket_col, base_col in inter_bucket_pairs:
    if base_col in inter_cols:
        x_expr = f"i.{qident(base_col)}"
    elif base_col == "d_age":
        x_expr = """
        CASE
            WHEN ps.age IS NULL AND pp.age IS NULL THEN NULL
            WHEN ps.age IS NULL THEN pp.age
            WHEN pp.age IS NULL THEN ps.age
            ELSE ABS(ps.age - pp.age)
        END
        """
    else:
        continue

    iwf_select.append(
        f"{_bucket_expr(x_expr, bucket_col, base_col)} AS {bucket_col}"
    )

interaction_view_sql = (
    "CREATE OR REPLACE VIEW Interaction_with_features AS\n"
    "SELECT\n  " + ",\n  ".join(iwf_select) + "\n"
    "FROM Interaction i\n"
    "JOIN Person ps ON ps.wave=i.wave AND ps.self_id=i.self_id\n"
    "JOIN Person pp ON pp.wave=i.wave AND pp.self_id=i.partner_id;"
)
con.execute(interaction_view_sql)


# ============================================================
# 4) Flatten with EXACT column names/order (no prefixes, no post-cleaning)
# ============================================================

final_cols = [
  "has_null","wave","gender","age","age_o","d_age","d_d_age","race","race_o","samerace",
  "importance_same_race","importance_same_religion","d_importance_same_race","d_importance_same_religion",
  "field",
  "pref_o_attractive","pref_o_sincere","pref_o_intelligence","pref_o_funny","pref_o_ambitious","pref_o_shared_interests",
  "d_pref_o_attractive","d_pref_o_sincere","d_pref_o_intelligence","d_pref_o_funny","d_pref_o_ambitious","d_pref_o_shared_interests",
  "attractive_o","sinsere_o","intelligence_o","funny_o","ambitous_o","shared_interests_o",
  "d_attractive_o","d_sinsere_o","d_intelligence_o","d_funny_o","d_ambitous_o","d_shared_interests_o",
  "attractive_important","sincere_important","intellicence_important","funny_important","ambtition_important","shared_interests_important",
  "d_attractive_important","d_sincere_important","d_intellicence_important","d_funny_important","d_ambtition_important","d_shared_interests_important",
  "attractive","sincere","intelligence","funny","ambition",
  "d_attractive","d_sincere","d_intelligence","d_funny","d_ambition",
  "attractive_partner","sincere_partner","intelligence_partner","funny_partner","ambition_partner","shared_interests_partner",
  "d_attractive_partner","d_sincere_partner","d_intelligence_partner","d_funny_partner","d_ambition_partner","d_shared_interests_partner",
  "sports","tvsports","exercise","dining","museums","art","hiking","gaming","clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga",
  "d_sports","d_tvsports","d_exercise","d_dining","d_museums","d_art","d_hiking","d_gaming","d_clubbing","d_reading","d_tv","d_theater","d_movies","d_concerts","d_music","d_shopping","d_yoga",
  "interests_correlate","d_interests_correlate",
  "expected_happy_with_sd_people","expected_num_interested_in_me","expected_num_matches",
  "d_expected_happy_with_sd_people","d_expected_num_interested_in_me","d_expected_num_matches",
  "like","guess_prob_liked","d_like","d_guess_prob_liked",
  "met",
]

# Mapping expressions (IMPORTANT: pref_o_* come from PARTNER person = pp, not ps)
expr = {}

final_cols = ["row_id"] + final_cols
expr["row_id"] = "iwf.row_id"

# self side from ps
self_from_ps = [
  "has_null","wave","gender","age","race","field",
  "importance_same_race","importance_same_religion",
  "attractive","sincere","intelligence","funny","ambition",
  "sports","tvsports","exercise","dining","museums","art","hiking","gaming","clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga",
  "expected_happy_with_sd_people","expected_num_interested_in_me","expected_num_matches",
  "attractive_important","sincere_important","intellicence_important","funny_important","ambtition_important","shared_interests_important",

  # bucketized (if rules exist, they appear in Person_with_buckets)
  "d_importance_same_race","d_importance_same_religion",
  "d_attractive","d_sincere","d_intelligence","d_funny","d_ambition",
  "d_sports","d_tvsports","d_exercise","d_dining","d_museums","d_art","d_hiking","d_gaming","d_clubbing","d_reading","d_tv","d_theater","d_movies","d_concerts","d_music","d_shopping","d_yoga",
  "d_expected_happy_with_sd_people","d_expected_num_interested_in_me","d_expected_num_matches",
  "d_attractive_important","d_sincere_important","d_intellicence_important","d_funny_important","d_ambtition_important","d_shared_interests_important",
]
for c in self_from_ps:
    expr[c] = f"ps.{qident(c)}"

# partner side from pp
expr["age_o"] = "pp.age"
expr["race_o"] = "pp.race"

partner_from_pp = [
  "pref_o_attractive","pref_o_sincere","pref_o_intelligence","pref_o_funny","pref_o_ambitious","pref_o_shared_interests",
  "d_pref_o_attractive","d_pref_o_sincere","d_pref_o_intelligence","d_pref_o_funny","d_pref_o_ambitious","d_pref_o_shared_interests",
]
for c in partner_from_pp:
    expr[c] = f"pp.{qident(c)}"

# interaction side from iwf
iwf_from_iwf = [
  "interests_correlate","met",
  "attractive_partner","sincere_partner","intelligence_partner","funny_partner","ambition_partner","shared_interests_partner",
  "attractive_o","sinsere_o","intelligence_o","funny_o","ambitous_o","shared_interests_o",
  "d_attractive_partner","d_sincere_partner","d_intelligence_partner","d_funny_partner","d_ambition_partner","d_shared_interests_partner",
  "d_attractive_o","d_sinsere_o","d_intelligence_o","d_funny_o","d_ambitous_o","d_shared_interests_o",
  "d_interests_correlate",
  "like","guess_prob_liked", "d_like", "d_guess_prob_liked",
  "has_null"
]
for c in iwf_from_iwf:
    expr[c] = f"iwf.{qident(c)}"

# derived in iwf
expr["d_age"] = "iwf.d_age"
expr["d_d_age"] = "iwf.d_d_age"
expr["samerace"] = "iwf.samerace"
expr["has_null"] = f"iwf.{qident('has_null')}"

# --- Fail fast: ensure every final col exists in the upstream relations we reference ---
pwb_cols = set(cols_of(con, "Person_with_buckets"))
iwf_cols = set(cols_of(con, "Interaction_with_features"))

missing_map = [c for c in final_cols if c not in expr]
if missing_map:
    raise ValueError(f"expr mapping missing for final cols: {missing_map}")

# Check referenced columns exist (no BinderError surprises)
def _check_ref(alias: str, col: str):
    if alias == "ps":
        return col in pwb_cols
    if alias == "pp":
        return col in pwb_cols
    if alias == "iwf":
        return col in iwf_cols
    return True

bad_refs = []
for c in final_cols:
    e = expr[c]
    m = re.match(r"^(ps|pp|iwf)\.(.+)$", e)
    if m:
        a, col = m.group(1), m.group(2)
        # strip quotes for membership test
        col_plain = col.strip('"')
        if not _check_ref(a, col_plain):
            bad_refs.append((c, e))
if bad_refs:
    msg = "\n".join([f"- {c}: {e}" for c, e in bad_refs[:50]])
    raise ValueError("❌ Upstream column(s) missing in views:\n" + msg)

# Build final SELECT with stable order and clean formatting
select_list = ",\n  ".join([f"{expr[c]} AS {qident(c)}" for c in final_cols])

flatten_sql = f"""
DROP TABLE IF EXISTS Flattened_SpeedDating;
CREATE TABLE Flattened_SpeedDating AS
SELECT
  {select_list}
FROM Interaction_with_features iwf
JOIN Person_with_buckets ps
  ON ps.wave = iwf.wave AND ps.self_id = iwf.self_id
JOIN Person_with_buckets pp
  ON pp.wave = iwf.wave AND pp.self_id = iwf.partner_id
;
"""
con.execute(flatten_sql)

flat = con.execute("SELECT * FROM Flattened_SpeedDating").df()


# ============================================================
# 5) Write a clean SQL script file (optional)
# ============================================================

# sql_script = "\n".join([
#     "\n".join(person_view_sql.splitlines()[1:]),
#     "",
#     "\n".join(interaction_view_sql.splitlines()[1:]),
#     "",
#     "\n".join(flatten_sql.strip().splitlines()[2:]),
# ])

sql_script = "\n".join(flatten_sql.strip().splitlines()[2:])

sql_script = sql_script.replace('"', '')



if os.path.exists(SQL_PATH):
    os.remove(SQL_PATH)

with open(SQL_PATH, "w") as f:
    f.write(sql_script)

print("Wrote SQL script to:", SQL_PATH)

flat_sorted = (
    flat
    .set_index("row_id")
    .loc[df_id_final.row_id]
    .reset_index()
).drop(['row_id'],axis=1)

self_id_arr = df_id_final["self_id"].to_numpy()
partner_id_arr = df_id_final["partner_id"].to_numpy()
perm = flat_sorted.index.to_numpy()

df_id_final = df_id_final.drop(['row_id', 'self_id', 'partner_id'], axis=1)

df_id_final = df_id_final.reindex(columns=df.columns)
flat_sorted = flat_sorted.reindex(columns=df.columns)
flat_sorted["match"] = df_id_final["match"].values

for col in flat_sorted.columns:
    df_id_final[col] = df_id_final[col].astype(flat_sorted[col].dtype)

ok = df_id_final.equals(flat_sorted)
if ok:
    print("FINAL CHECK PASSED (with class)")
else:
    n_diff = (df_id_final != flat_sorted).sum().sum()
    raise AssertionError(f"FINAL CHECK FAILED (with class), n_diff={n_diff}")

flat_sorted["self_id"] = self_id_arr[perm]
flat_sorted["partner_id"] = partner_id_arr[perm]

cols = ["self_id", "partner_id"] + [c for c in flat_sorted.columns
                                    if c not in ["self_id", "partner_id"]]

flat_sorted.to_csv(CSV_PATH, columns=cols, index=False)



# print(df_id_final)
# df_id_final.to_csv('./demo/df_id_final.csv', index=False)
# print(flat_sorted)

# df_id_final, flat_sorted 는 이미 row 순서 맞춘 상태라고 가정

# bad_cols = []
# for c in df.columns:
#     if c not in df_id_final.columns or c not in flat_sorted.columns:
#         bad_cols.append((c, "missing"))
#         continue

#     a = df_id_final[c]
#     b = flat_sorted[c]

#     # exact w/ NaN==NaN
#     neq = ~_col_equal_exact(a, b)
#     n = int(neq.sum())
#     if n > 0:
#         bad_cols.append((c, n))

# print("DIFF COLS (top):", sorted([x for x in bad_cols if x[1] != "missing"], key=lambda x: -x[1])[:30])

# # 어떤 row들이 틀리는지 몇 개 보기
# col = "pref_o_shared_interests"  # 의심 컬럼
# mask = ~_col_equal_exact(df_id_final[col], flat_sorted[col])
# print("n_diff:", int(mask.sum()))
# print(pd.DataFrame({
#     "row_id": df_id_final["row_id"] if "row_id" in df_id_final.columns else np.arange(len(df_id_final)),
#     "rleft": df_id_final["pref_o_shared_interests"],
#     "rright": flat_sorted["pref_o_shared_interests"],
#     "left": df_id_final[col],
#     "right": flat_sorted[col],
# }).loc[mask].head(20))



FD_CSV = "./speeddating/fd_query_add.csv"

fds = []

def add_fd(lhs, rhs):
    fds.append({
        "lhs": ",".join(lhs),
        "rhs": rhs,
    })

for c in person_fixed:
    add_fd(["wave", "self_id"], c)

for c in interaction_cols:
    add_fd(["wave", "self_id", "partner_id"], c)

# ps.age == age, pp.age == age_o
add_fd(["age", "age_o"], "d_age")

# ps.race == race, pp.race == race_o
add_fd(["race", "race_o"], "samerace")

# pairs = SELECT DISTINCT bucket_col, base_col FROM BucketRule

for bucket_col, base_col in pairs:
    add_fd([base_col], bucket_col)

fd_df = pd.DataFrame(fds).drop_duplicates().sort_values(["rhs", "lhs"])

fd_df.to_csv(FD_CSV, index=False)
print("Wrote FD file to:", FD_CSV)
