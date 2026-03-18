import os
import random
import numpy as np

import argparse
from pathlib import Path

import yaml

import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tabpfn.model_loading import save_fitted_tabpfn_model, load_fitted_tabpfn_model


import joblib
import torch
import torch.backends.cudnn as cudnn

from models import build_model, MODEL_REGISTRY

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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _save_model(out_dir: Path, *, model_name: str, ts: str, save_format: str, obj) -> Path:
    _ensure_dir(out_dir)
    if save_format == "joblib":
        path = out_dir / f"{model_name}_{ts}.joblib"
        joblib.dump(obj, path)
        return path
    
    if save_format == "tabpfn_fit":
        path = out_dir / f"{model_name}_{ts}.tabpfn_fit"
        save_fitted_tabpfn_model(obj, str(path))
        return path

    raise ValueError(f"Unknown save_format: {save_format}")


def _param_distributions(model_key: str, *, is_multiclass: bool):

    if model_key == "logreg":
        return {
            "clf__C": np.logspace(-4, 2, 15)
        }

    if model_key == "svm":
        return {
            "clf__C": np.logspace(-2, 2, 9),
            "clf__gamma": ["scale", "auto"],
        }

    if model_key == "randomforest":
        return {
            "clf__n_estimators": [300, 500, 800],
            "clf__max_depth": [None, 6, 10, 16],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__max_features": ["sqrt", "log2", None],
        }

    if model_key == "xgboost":
        return {
            "clf__n_estimators": [300, 500, 800],
            "clf__max_depth": [3, 4, 6, 8],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__subsample": [0.7, 0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 1.0],
            "clf__min_child_weight": [1, 3, 5],
            "clf__reg_lambda": [0.5, 1.0, 2.0],
        }

    if model_key == "lightgbm":
        return {
            "clf__n_estimators": [400, 800, 1200],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_depth": [-1, 6, 10],
            "clf__num_leaves": [31, 63, 127],
            "clf__subsample": [0.7, 0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 1.0],
            "clf__reg_lambda": [0.0, 1.0, 2.0],
        }

    if model_key == "mlp":
        return {
            "clf__hidden_layer_sizes": [(128,), (256,), (256, 128), (512, 256)],
            "clf__dropout": [0.0, 0.1, 0.2],
            "clf__lr": [3e-4, 1e-3, 3e-3],
            "clf__weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
            "clf__batch_size": [128, 256, 512],
            "clf__max_epochs": [30, 50, 80],
        }
    
    if model_key == "mlp_plr":
        return {
            # two important parameters: sigma and k
            "clf__frequency_init_scale": [0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
            "clf__n_frequencies": [8, 16, 32],

            "clf__d_embedding": [12, 24, 32],
            "clf__lite": [False],

            # backbone, consistent with mlp
            "clf__hidden_layer_sizes": [(128,), (256,), (256, 128), (512, 256)],
            "clf__dropout": [0.0, 0.1, 0.2],
            "clf__lr": [3e-4, 1e-3, 3e-3],
            "clf__weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
            "clf__batch_size": [128, 256, 512],
            "clf__max_epochs": [30, 50, 80],
        }

    if model_key == "ft_transformer":
        return {
            "d_block": [128, 192, 256],
            "n_blocks": [2, 3, 4],
            "attention_n_heads": [4, 8],
            "attention_dropout": [0.0, 0.1, 0.2],
            "ffn_dropout": [0.0, 0.1, 0.2],
            "lr": [3e-5, 1e-4, 3e-4],
            "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
            "batch_size": [128, 256],
            "max_epochs": [30, 50, 80],
            "linformer_kv_compression_ratio": [None, 0.2],
        }
    
    if model_key == "tabpfn":
        return None

    return None




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--data-split-seed", required=True)
    parser.add_argument("--flattened", required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--pred-out", required=True)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--ts", required=True)


    args = parser.parse_args()
    
    SEED = int(args.seed)
    seed_everything(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    model_name = args.model.lower().strip()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}. Use one of: {sorted(MODEL_REGISTRY.keys())}")


    base_dir = Path(args.base_dir)
    data = pd.read_csv(base_dir / args.flattened)

    with open(base_dir / args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    label_col = list(config.get("LABEL_COL", []))
    if not label_col:
        raise ValueError("LABEL_COL is missing in config YAML.")

    drop_cols = list(config.get("DROP_COLS") or [])
    
    X = data.drop(columns=label_col+drop_cols, errors="raise")
    y = data[label_col].to_numpy().ravel()

    le = LabelEncoder()
    y = le.fit_transform(y)   # 'bad','good' -> 0,1 (또는 multiclass면 0..K-1)

    classes = le.classes_
    is_multiclass = (len(classes) > 2)

    num_cols = list(X.select_dtypes(include=["number", "bool"]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    
    spec, estimator = build_model(args.model, seed=SEED)

    # ===== train/test split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=int(args.data_split_seed), stratify=y
    )

    test_index = X_test.index

    if args.model == "mlp_plr":
        estimator.set_params(n_num_features=len(num_cols))

    if args.model == "ft_transformer":
        # FT는 연속/범주 컬럼 정보를 estimator에게 넘겨줘야 함
        estimator.set_params(
            cont_cols=num_cols,
            cat_cols=cat_cols,
        )
    
    if args.model in ("ft_transformer", "tabpfn"):
        model = estimator  # <-- Pipeline 안 씀
    
    else:
        model = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("clf", estimator),
            ]
        )

    if estimator is None:
        raise ValueError(
            f"Model '{args.model}' is registered but not implemented yet "
            f"(DL placeholder). Implement torch version or pick another model."
        )

    # multiclass objective/loss
    if args.model == "xgboost":
        if is_multiclass:
            estimator.set_params(objective="multi:softprob", num_class=len(classes))
        else:
            estimator.set_params(objective="binary:logistic")

    # ===== (optional) tuning =====
    best_params = None
    if args.tune:
        dist = _param_distributions(args.model, is_multiclass=is_multiclass)

        if dist is None:
            if args.model == "tabpfn":
                print("[Tuning] tabpfn has no tuning space. Skip tuning and fit default model.")
                model.fit(X_train, y_train)
                dist = None
            else:
                raise ValueError(f"No tuning space defined for model={args.model}")
        else:
            cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)
            scoring = "roc_auc" if not is_multiclass else "roc_auc_ovr"

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=dist,
                n_iter=args.n_iter,
                scoring=scoring,
                cv=cv,
                random_state=SEED,
                n_jobs=1,
                verbose=2,
            )
            
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_

            print(f"\n[Tuning] best_score for the training data ({scoring})={search.best_score_:.4f}")
            print(f"[Tuning] best_params={best_params}")
    else:
        model.fit(X_train, y_train)

    def _report_split(tag, *, y_ref, y_pred, proba, y_train_for_majority):
        # tag: "Train" or "Test"

        # ===== Majority baseline =====
        majority = pd.Series(y_train_for_majority).value_counts().idxmax()
        y_pred_maj = np.full_like(y_ref, fill_value=majority)

        print(f"\n===== [{tag}] Majority-class baseline =====")
        print(f"Accuracy: {accuracy_score(y_ref, y_pred_maj):.4f}")
        if not is_multiclass:
            print(f"F1: {f1_score(y_ref, y_pred_maj, average='binary'):.4f}")
            print("AUC: N/A (0.5)")
        else:
            print(f"F1: {f1_score(y_ref, y_pred_maj, average='macro'):.4f}")
            print("AUC (OVR, macro): N/A (0.5)")

        # ===== Model metrics =====
        print(f"\n===== [{tag}] Model =====")
        print(f"Accuracy: {accuracy_score(y_ref, y_pred):.4f}")
        if not is_multiclass:
            print(f"F1: {f1_score(y_ref, y_pred, average='binary'):.4f}")
        else:
            print(f"F1: {f1_score(y_ref, y_pred, average='macro'):.4f}")

        if proba is not None:
            if not is_multiclass:
                print(f"AUC: {roc_auc_score(y_ref, proba[:, 1]):.4f}")
            else:
                print(f"AUC (OvR, macro): {roc_auc_score(y_ref, proba, multi_class='ovr', average='macro'):.4f}")
        else:
            print("AUC: (skip) model has no predict_proba")

        # ===== FPR / FNR / TPR / TNR =====
        cm = confusion_matrix(y_ref, y_pred)

        if not is_multiclass:
            tn, fp, fn, tp = cm.ravel()
            print("\nConfusion matrix:")
            print(f"TP: {tp} | FP: {fp}")
            print(f"FN: {fn} | TN: {tn}")
            tpr = tp / (tp + fn)
            fnr = fn / (tp + fn)
            tnr = tn / (tn + fp)
            fpr = fp / (tn + fp)

            print(f"TPR: {tpr:.4f}")
            print(f"FNR: {fnr:.4f}")
            print(f"TNR: {tnr:.4f}")
            print(f"FPR: {fpr:.4f}")

        else:
            print("\nConfusion matrix:")
            for i in range(cm.shape[0]):
                row = " ".join(f"{v:6d}" for v in cm[i])
                print(f"Class {i}: {row}")
            
            print("\nMulti-class (one-vs-rest):")
            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - (tp + fn + fp)

                tpr = tp / (tp + fn)
                fnr = fn / (tp + fn)
                tnr = tn / (tn + fp)
                fpr = fp / (tn + fp)

                print(
                    f"Class {i}: "
                    f"TPR: {tpr:.4f}, "
                    f"FNR: {fnr:.4f}, "
                    f"TNR: {tnr:.4f}, "
                    f"FPR: {fpr:.4f}"
                )


    # ===== Evaluate (Train/Test) =====
    # Train
    y_pred_tr = model.predict(X_train)
    proba_tr = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
    _report_split("Train", y_ref=y_train, y_pred=y_pred_tr, proba=proba_tr, y_train_for_majority=y_train)

    # Test
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    _report_split("Test", y_ref=y_test, y_pred=y_pred, proba=proba, y_train_for_majority=y_train)


    # ===== Final params print =====
    print("\n[Final Params Used]")
    if best_params is not None:
        print(f"Tuned params: {best_params}")

    # Pipeline이면 named_steps["clf"], 아니면 model 자체가 clf
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    try:
        print(f"Classifier type: {type(clf).__name__}")
        print(f"Classifier params: {clf.get_params(deep=False)}")
    except Exception as e:
        print(f"(Could not print clf params) {e}")


    # ===== Save =====
    out_path = Path(args.model_out)
    out_dir = out_path.parent if out_path.suffix else out_path
    _save_model(out_dir, model_name=spec.name, ts=args.ts, save_format=spec.save_format, obj=model)

    df_test_pred = pd.DataFrame({
        "row_id": test_index,
        "y_true": y_test,
        "y_pred": y_pred
    })

    if proba is not None:
        if proba.shape[1] == 2:
            df_test_pred["proba_0"] = proba[:, 0]
            df_test_pred["proba_1"] = proba[:, 1]
        else:
            # multi-class
            for i in range(proba.shape[1]):
                df_test_pred[f"proba_{i}"] = proba[:, i]

    save_dir = args.pred_out
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        args.pred_out,
        f"{spec.name}_{args.ts}.csv"
    )

    df_test_pred.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
