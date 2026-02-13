#!/bin/bash
set -e

export HF_TOKEN="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" # for TabPFN
export TABPFN_MODEL_CACHE_DIR="/Users/seungeun/nyu/relshap2026/tabpfn_weights" # where to store TabPFN weights
export TABPFN_DISABLE_TELEMETRY=1

# To Joao: use synth - xgboost for development purpose
# You don't need to check other files; just check run_relshap.py

DATASET="synth" 
# ML-side datasets: acs, amazon_employee_access, churn, churn_modeling, credit_g, speeddating
# DB-side datasets: movielens20m, olist, tpch, uwcse 
# (Note. movielens20m and olist is not available in github since file is too large)
# Synthetic datasets: synth
# ML-side (w.o. meaninful relational constraints, just to check): diabetes

BASE_DIR="/Users/seungeun/nyu/relshap2026/relshap/dataset/$DATASET" # base dir
SEED=2026
DB="$DATASET.duckdb" # Do not support CTEs; Use QUALIFY in the duckdb dialect
QUERY="$DATASET.sql"
FLATTENED="flattened_$DATASET.csv"
CONFIG="config.yaml"
MODEL="xgboost"
# models available:
# ML-side: logreg, svm, randomforest, xgboost, lightgbm
# DL-side: mlp, mlp_plr, ft_transformer, tabpfn

FD_S="fd_schema.csv"
FD_Q="fd_query.csv"
FD_D="fd_data.csv"
APPROX_FD="approximate_fd.csv"
FD_APPROX_ERROR_THRESHOLD=0.00 # 0.00 for exact mode
FD_FINAL="fd_final.csv"
DOMAIN_S="conditional_domain_constraint_s.csv"
DOMAIN_Q="conditional_domain_constraint_q.csv"
DOMAIN_CONSTRAINT_D="domain_constraint.csv"
DENIAL_CONSTRAINT_D="denial_constraint.csv"

CONSTRAINTS_CACHE="constraints_cache.pkl"
CONSTRAINTS_FINAL="constraints_final.csv"

TS=$(date +%m%d%Y_%H%M%S)
LOG_DIR="$BASE_DIR/logs/${MODEL}_${TS}"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"

# prepare duckdb, query, and data file
python -u $BASE_DIR/make_$DATASET.py \
    --base-dir "$BASE_DIR" \
    --db "$DB" \
    --query "$QUERY" \
    --flattened "$FLATTENED" \
    > "$LOG_DIR/make_$DATASET.out" \
    2> "$LOG_DIR/make_$DATASET.err"

# extract constraints from the relational schema
python constraint_schema.py \
  --base-dir "$BASE_DIR" \
  --db "$DB" \
  --fd-s "$FD_S" \
  --domain-s "$DOMAIN_S" \
  > "$LOG_DIR/constraint_schema.out" \
  2> "$LOG_DIR/constraint_schema.err"
 
# extract constraints from the queries by parsing them (do not include annotations)
python constraint_query.py \
  --base-dir "$BASE_DIR" \
  --query "$QUERY" \
  --fd-q "$FD_Q" \
  --domain-q "$DOMAIN_Q" \
  --db "$DB" \
  > "$LOG_DIR/constraint_query.out" \
  2> "$LOG_DIR/constraint_query.err"

# extract constraints from the data itself
# lut (lookup table): full or train
python constraint_data.py \
  --base-dir "$BASE_DIR" \
  --flattened "$FLATTENED" \
  --config "$CONFIG" \
  --fd-d "$FD_D" \
  --approx-fd "$APPROX_FD" \
  --domain-constraint-d "$DOMAIN_CONSTRAINT_D" \
  --denial-constraint-d "$DENIAL_CONSTRAINT_D" \
  --lut train \
  --seed "$SEED" \
  > "$LOG_DIR/constraint_data.out" \
  2> "$LOG_DIR/constraint_data.err"

# --fd-exclude "$FD_S" "$FD_Q" \
# include this to avoid detecting unnecessary duplicates

# refining relational constraints
# Users can specifiy the level of constraints to include:
# s/q/d/approx/domain/denial constraints
# --mode-provenance if the user wants to use --mode-provenance later on
# w.o. --mode-provenance, final constraints table (and the cache file) will not include
# drop_cols (basically identifiers) on the LHS (mostly from the schema and query)
python fd_ic_refinement.py \
  --base-dir "$BASE_DIR" \
  --config "$CONFIG" \
  --out "$CONSTRAINTS_CACHE" \
  --out-csv "$CONSTRAINTS_FINAL" \
  --approx-error-threshold "$FD_APPROX_ERROR_THRESHOLD" \
  --fd-s "$FD_S" \
  --fd-q "$FD_Q" \
  --fd-d "$FD_D" \
  --approx-fd "$APPROX_FD" \
  --domain-s "$DOMAIN_S" \
  --domain-q "$DOMAIN_Q" \
  --domain-d "$DOMAIN_CONSTRAINT_D" \
  --denial-d "$DENIAL_CONSTRAINT_D" \
  --mode-provenance \
  > "$LOG_DIR/fd_ic_refinement.out" \
  2> "$LOG_DIR/fd_ic_refinement.err"

# run ML/DL models and save them
python -u run_model.py \
  --base-dir "$BASE_DIR" \
  --seed "$SEED" \
  --flattened "$FLATTENED" \
  --config "$CONFIG" \
  --model $MODEL \
  --model-out "$BASE_DIR/models" \
  --tune \
  --n-iter 10 \
  --cv 2 \
  --ts $TS \
  > "$LOG_DIR/run_model.out" \
  2> "$LOG_DIR/run_model.err"

# RelShap:

# --background-n: # of background samples
# --explain-n 100: # of test samples to explain
# --nsamples 100: # of coalitions to use
# --base-mode: kernel, mc, leverage 
# (KernelExplainer, Monte Carlo coalition sampling, and LeverageSHAP, resp)

# Modes below are for RelShap; if the rest is turned off, it does vanilla SHAP, MC, and LeverageSHAP
# --mode-bg: if ON, do Relshap's background sampling
# --mode-coalition-memoization: if ON, computes coalition equivalence class and reduces runtime
# (this requires --mode-bg to be ONs)
# --mode-coalition-quotient: if ON, computes coalition equivalence class, and draw extra samples to offset the reduction
# (no runtime improvement, runtime similar to the vanilla mode)
# --mode-domain: uses domain constraints
# --mode-denial: uses denial constraints

# --bg-lut: lookup table for background sampling (train or full), need to specify when enabling --mode-bg

# --mode-provenance: provenance mode enabled
# "bg-only", "bg-coalition-memoization", "bg-coalition-quotient" >> three options available
# bg-only: RelShap's local background sampling through provenance mode
# bg-coalition-memoization: RelShap's local background sampling through provenance mode + computes local coalition equivalence class and reduces runtime
# bg-coalition-memoization: RelShap's local background sampling through provenance mode + computes local coalition equivalence class and draw extra samples to offset the reduction
# Note that this of course is independent of any modes listed above
# --prov-strength: strong or weak

# --debug: if the user wants to print out logs (this slightly increases the runtime)

# To Joao: if you want to work on mode-provenance, 
# --mode-provenance bg-only OR bg-coalition-memoization OR bg-coalition-quotient 
# --prov-strength strong OR weak 
# only vary these two modes and keep the rest fixed

python -u run_relshap.py \
  --base-dir "$BASE_DIR" \
  --seed "$SEED" \
  --flattened "$FLATTENED" \
  --config "$CONFIG" \
  --constraints-cache "$CONSTRAINTS_CACHE" \
  --model-path "$BASE_DIR/models/${MODEL}_${TS}" \
  --background-n 200 \
  --explain-n 100 \
  --nsamples 100 \
  --base-mode mc \
  --mode-provenance bg-coalition-memoization \
  --prov-strength strong \
  --debug \
  --out "$TS" \
  > "$LOG_DIR/run_relshap.out" \
  2> "$LOG_DIR/run_relshap.err"

echo "Finished."