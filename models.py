from dataclasses import dataclass
from typing import Any, Dict, Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from rtdl_num_embeddings import PeriodicEmbeddings

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin

from tabpfn import TabPFNClassifier

class MLPClassifier(ClassifierMixin, BaseEstimator):
    """
    compatible with sklearn Pipeline/RandomizedSearchCV
    """
    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_layer_sizes=(256, 128),
        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=50,
        random_state=0,
        device="auto",   # "auto" | "cpu" | "cuda" | "mps"
        verbose=False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

        self.model_ = None
        self.classes_ = None
        self.n_features_in_ = None

    def _pick_device(self):
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _to_numpy_dense(X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X)
        return X.astype(np.float32, copy=False)

    def _build_net(self, in_dim: int, out_dim: int):
        layers = []
        prev = in_dim
        for h in self.hidden_layer_sizes:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if self.dropout and float(self.dropout) > 0:
                layers.append(nn.Dropout(float(self.dropout)))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X = self._to_numpy_dense(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = int(len(self.classes_))
        if n_classes < 2:
            raise ValueError("Need at least 2 classes for classification.")

        self.n_features_in_ = int(X.shape[1])
        dev = self._pick_device()

        torch.manual_seed(int(self.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.random_state))

        X_t = torch.from_numpy(X).to(dev)
        y_t = torch.from_numpy(y.astype(np.int64, copy=False)).to(dev)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=int(self.batch_size),
            shuffle=True,
        )

        net = self._build_net(self.n_features_in_, n_classes).to(dev)
        opt = torch.optim.Adam(
            net.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        crit = nn.CrossEntropyLoss()

        for epoch in range(int(self.max_epochs)):
            net.train()
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                logits = net(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu().item())

            if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                avg = total / max(1, len(loader))
                print(f"[MLP] epoch={epoch+1} train_loss={avg:.6f}")

        self.model_ = net
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Model not fitted yet.")

        X = self._to_numpy_dense(X)
        dev = self._pick_device()
        X_t = torch.from_numpy(X).to(dev)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

class PLRNet(nn.Module):
    def __init__(self, emb: nn.Module, trunk: nn.Module):
        super().__init__()
        self.emb = emb
        self.trunk = trunk

    def forward(self, x_num, x_cat):
        z = self.emb(x_num)
        z = z.flatten(1)
        if x_cat is not None:
            z = torch.cat([z, x_cat], dim=1)
        return self.trunk(z)


class MLPPLRClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
        self,
        n_num_features=None,
        d_embedding=24,
        n_frequencies=16,
        frequency_init_scale=0.05,
        lite=True,

        hidden_layer_sizes=(256, 128),
        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=50,
        random_state=0,
        device="auto",   # "auto" | "cpu" | "cuda" | "mps"
        verbose=False,
    ):
        self.n_num_features = n_num_features
        self.d_embedding = d_embedding
        self.n_frequencies = n_frequencies
        self.frequency_init_scale = frequency_init_scale
        self.lite = lite

        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

        self.model_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.n_cat_features_ = None

    def _pick_device(self):
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _to_numpy_dense(X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X)
        return X.astype(np.float32, copy=False)

    def _build_mlp(self, in_dim: int, out_dim: int):
        layers = []
        prev = in_dim
        for h in self.hidden_layer_sizes:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if self.dropout and float(self.dropout) > 0:
                layers.append(nn.Dropout(float(self.dropout)))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def _build_net(self, n_num: int, n_cat: int, out_dim: int):
        emb = PeriodicEmbeddings(
            n_features=n_num,
            d_embedding=int(self.d_embedding),
            n_frequencies=int(self.n_frequencies),
            frequency_init_scale=float(self.frequency_init_scale),
            lite=bool(self.lite),
        )
        mlp_in = n_num * int(self.d_embedding) + n_cat
        trunk = self._build_mlp(mlp_in, out_dim)

        return PLRNet(emb, trunk)


    def fit(self, X, y):
        X = self._to_numpy_dense(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = int(len(self.classes_))
        if n_classes < 2:
            raise ValueError("Need at least 2 classes for classification.")

        if self.n_num_features is None:
            raise ValueError("n_num_features is None. Set it via set_params(n_num_features=...).")

        n_num = int(self.n_num_features)
        if X.shape[1] < n_num:
            raise ValueError(f"X has {X.shape[1]} features but n_num_features={n_num}.")

        n_cat = int(X.shape[1] - n_num)
        self.n_cat_features_ = n_cat
        self.n_features_in_ = int(X.shape[1])

        dev = self._pick_device()

        torch.manual_seed(int(self.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.random_state))

        X_num = X[:, :n_num]
        X_cat = X[:, n_num:] if n_cat > 0 else None

        X_num_t = torch.from_numpy(X_num).to(dev)
        X_cat_t = torch.from_numpy(X_cat).to(dev) if X_cat is not None else None
        y_t = torch.from_numpy(y.astype(np.int64, copy=False)).to(dev)

        if X_cat_t is None:
            dataset = TensorDataset(X_num_t, y_t)
        else:
            dataset = TensorDataset(X_num_t, X_cat_t, y_t)

        loader = DataLoader(
            dataset,
            batch_size=int(self.batch_size),
            shuffle=True,
        )

        net = self._build_net(n_num=n_num, n_cat=n_cat, out_dim=n_classes).to(dev)
        opt = torch.optim.Adam(
            net.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        crit = nn.CrossEntropyLoss()

        for epoch in range(int(self.max_epochs)):
            net.train()
            total = 0.0

            for batch in loader:
                opt.zero_grad(set_to_none=True)

                if X_cat_t is None:
                    xb_num, yb = batch
                    logits = net(xb_num, None)
                else:
                    xb_num, xb_cat, yb = batch
                    logits = net(xb_num, xb_cat)

                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu().item())

            if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                avg = total / max(1, len(loader))
                print(f"[MLP-PLR] epoch={epoch+1} train_loss={avg:.6f}")

        self.model_ = net
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Model not fitted yet.")

        X = self._to_numpy_dense(X)
        dev = self._pick_device()

        n_num = int(self.n_num_features)
        n_cat = int(X.shape[1] - n_num)

        X_num = X[:, :n_num]
        X_cat = X[:, n_num:] if n_cat > 0 else None

        X_num_t = torch.from_numpy(X_num).to(dev)
        X_cat_t = torch.from_numpy(X_cat).to(dev) if X_cat is not None else None

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_num_t, X_cat_t)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

class FTTransformerClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
        self,
        cont_cols=None,
        cat_cols=None,

        n_blocks=3,
        d_block=192,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden=None,
        ffn_d_hidden_multiplier=4/3,
        ffn_dropout=0.1,
        residual_dropout=0.0,
        linformer_kv_compression_ratio=None,
        linformer_kv_compression_sharing="headwise",

        lr=1e-4,
        weight_decay=1e-5,
        batch_size=256,
        max_epochs=50,
        random_state=0,
        device="auto",
        verbose=False,
    ):
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols

        self.n_blocks = n_blocks
        self.d_block = d_block
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.linformer_kv_compression_ratio = linformer_kv_compression_ratio
        self.linformer_kv_compression_sharing = linformer_kv_compression_sharing

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

        self.model_ = None
        self.classes_ = None
        self.cat_maps_ = None          # list[dict] per cat col: category -> int
        self.cat_cardinalities_ = None # list[int]

    def _pick_device(self):
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _fit_cat_encoders(self, X_df):
        maps = []
        cards = []
        for c in (self.cat_cols or []):
            s = X_df[c].astype("string").fillna("__NA__")
            cats = pd.Index(s.unique())
            m = {k: i for i, k in enumerate(cats)}
            maps.append(m)
            cards.append(len(m))
        self.cat_maps_ = maps
        self.cat_cardinalities_ = cards

    def _transform(self, X_df):
        # cont
        if self.cont_cols:
            x_cont = X_df[self.cont_cols].astype(np.float32).to_numpy()
        else:
            x_cont = np.zeros((len(X_df), 0), dtype=np.float32)

        # cat -> int
        if self.cat_cols:
            xs = []
            for c, m in zip(self.cat_cols, self.cat_maps_):
                s = X_df[c].astype("string").fillna("__NA__")
                xs.append(s.map(lambda v: m.get(v, 0)).astype(np.int64).to_numpy())
            x_cat = np.stack(xs, axis=1).astype(np.int64, copy=False)
        else:
            x_cat = np.zeros((len(X_df), 0), dtype=np.int64)

        return x_cont, x_cat

    def _build_model(self, n_cont, cat_cardinalities, n_classes):
        from rtdl_revisiting_models import FTTransformer

        kwargs = dict(
            n_cont_features=n_cont,
            cat_cardinalities=cat_cardinalities,
            d_out=n_classes,  # classification logits
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden=self.ffn_d_hidden,
            ffn_d_hidden_multiplier=self.ffn_d_hidden_multiplier,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
        )
        if self.linformer_kv_compression_ratio is not None:
            kwargs.update(
                linformer_kv_compression_ratio=self.linformer_kv_compression_ratio,
                linformer_kv_compression_sharing=self.linformer_kv_compression_sharing,
            )
        return FTTransformer(**kwargs)

    def fit(self, X, y):
        # X: pandas DataFrame expected
        if not hasattr(X, "__dataframe__") and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = int(len(self.classes_))
        if n_classes < 2:
            raise ValueError("Need at least 2 classes.")

        if self.cont_cols is None or self.cat_cols is None:
            raise ValueError("cont_cols/cat_cols must be set via set_params(...) in run_model.py")

        self._fit_cat_encoders(X)
        x_cont_np, x_cat_np = self._transform(X)

        dev = self._pick_device()
        torch.manual_seed(int(self.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.random_state))

        x_cont = torch.from_numpy(x_cont_np).to(dev)
        x_cat = torch.from_numpy(x_cat_np).to(dev)
        y_t = torch.from_numpy(y.astype(np.int64, copy=False)).to(dev)

        loader = DataLoader(
            TensorDataset(x_cont, x_cat, y_t),
            batch_size=int(self.batch_size),
            shuffle=True,
        )

        model = self._build_model(
            n_cont=x_cont.shape[1],
            cat_cardinalities=self.cat_cardinalities_,
            n_classes=n_classes,
        ).to(dev)

        # 논문/문서 스타일: 일부 파라미터 weight_decay 보호
        opt = torch.optim.AdamW(
            model.make_parameter_groups(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        crit = nn.CrossEntropyLoss()

        for epoch in range(int(self.max_epochs)):
            model.train()
            total = 0.0
            for xb_cont, xb_cat, yb in loader:
                opt.zero_grad(set_to_none=True)
                logits = model(xb_cont, xb_cat)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu().item())

            if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                print(f"[FT-Transformer] epoch={epoch+1} train_loss={total/max(1,len(loader)):.6f}")

        self.model_ = model
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Model not fitted yet.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        x_cont_np, x_cat_np = self._transform(X)
        dev = self._pick_device()

        x_cont = torch.from_numpy(x_cont_np).to(dev)
        x_cat = torch.from_numpy(x_cat_np).to(dev)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(x_cont, x_cat)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

class TabPFNWrapper(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
        self,
        device="auto",
        random_state=0,
    ):
        self.device = device
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        # X: pandas DataFrame or numpy array
        self.model_ = TabPFNClassifier(device=self.device)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)



@dataclass(frozen=True)
class ModelSpec:
    name: str
    save_format: str  # "joblib" | "tabpfn_fit"
    make_estimator: Callable[[int], Any]  # seed -> estimator (or None for unimplemented torch models)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # ===== classic ML =====
    "logreg": ModelSpec(
        name="logreg",
        save_format="joblib",
        make_estimator=lambda seed: LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            n_jobs=1,
            random_state=seed,
        ),
    ),
    "svm": ModelSpec(
        name="svm",
        save_format="joblib",
        make_estimator=lambda seed: SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=seed,
        ),
    ),
    "randomforest": ModelSpec(
        name="randomforest",
        save_format="joblib",
        make_estimator=lambda seed: RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=1,
            random_state=seed,
        ),
    ),

    # ===== GBDT family (optional deps) =====
    "xgboost": ModelSpec(
        name="xgboost",
        save_format="joblib",
        make_estimator=lambda seed: (
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=1,
                random_state=seed,
                tree_method="hist",
            )
        ),
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        save_format="joblib",
        make_estimator=lambda seed: (
            LGBMClassifier(
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=0.0,
                random_state=seed,
                n_jobs=1,
            )
        ),
    ),

    "mlp": ModelSpec(
        name="mlp",
        save_format="joblib",
        make_estimator=lambda seed: MLPClassifier(
            hidden_layer_sizes=(256, 128),
            dropout=0.0,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=256,
            max_epochs=50,
            random_state=seed,
            device="auto",
            verbose=False,
        ),
    ),

    "mlp_plr": ModelSpec(
        name="mlp_plr",
        save_format="joblib",
        make_estimator=lambda seed: MLPPLRClassifier(
            n_num_features=None,
            d_embedding=24,
            n_frequencies=16,
            frequency_init_scale=0.05,
            lite=True,
            hidden_layer_sizes=(256, 128),
            dropout=0.0,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=256,
            max_epochs=50,
            random_state=seed,
            device="auto",
            verbose=False,
        ),
    ),

    "ft_transformer": ModelSpec(
        name="ft_transformer",
        save_format="joblib",
        make_estimator=lambda seed: FTTransformerClassifier(
            cont_cols=None,
            cat_cols=None,

            n_blocks=3,
            d_block=192,
            attention_n_heads=8,
            attention_dropout=0.2,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4/3,
            ffn_dropout=0.1,
            residual_dropout=0.0,

            lr=1e-4,
            weight_decay=1e-5,
            batch_size=256,
            max_epochs=50,
            random_state=seed,
            device="auto",
            verbose=False,
        ),
    ),

    "tabpfn": ModelSpec(
        name="tabpfn",
        save_format="tabpfn_fit",
        make_estimator=lambda seed: TabPFNClassifier(
            device="auto",
            random_state=seed,
        ),
    ),

    
}


def build_model(model_name: str, *, seed: int):
    name = (model_name or "").lower().strip()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    spec = MODEL_REGISTRY[name]
    est = spec.make_estimator(seed)
    return spec, est
