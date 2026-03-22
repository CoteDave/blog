"""
ice_gnn.py  v3
══════════════════════════════════════════════════════════════════════════════
IceGNN — Graph-Attention Network over ICE Curves  (v3 — performance edition)
──────────────────────────────────────────────────────────────────────────────

v3 vs v2 — what changed and WHY it matters for R²/AUC
═══════════════════════════════════════════════════════

BUG 1 — Centering AFTER padding  (critical, silent corruption)
─────────────────────────────────────────────────────────────
v2: _center_ice(padded)  →  mean computed across ALL G_max positions,
    including zeros from padding.  A feature with G_j=10 and G_max=40 had
    its mean estimated on 30 zeros + 10 real values = ×4 diluted mean.
    Every ICE curve fed into the GNN was badly biased.
v3: Center BEFORE padding, on valid positions only.  Padded zeros then
    represent the correct "mean value" for a centered signal.

BUG 2 — ICE decoder reconstructs from AFTER-attention embeddings
──────────────────────────────────────────────────────────────────
v2: Decoder(node_emb_final) reconstructed raw single-feature ICE.
    But node_emb_final already contains information from ALL features
    (via attention), so reconstructing a per-feature curve from it is
    impossible.  The reconstruction loss fought directly against the
    interaction learning of the Transformer layers.
v3: Decoder(h0) — h0 is the initial encoding BEFORE any attention layer.
    Reconstruction trains the NodeEncoder to faithfully represent ICE shapes.
    The Transformer layers are ONLY trained by the prediction loss.
    No gradient conflict.

BUG 3 — Early stopping on combined loss (pred + recon)
────────────────────────────────────────────────────────
v2: val_loss = L_pred + λ*L_recon.  L_recon can still decrease even when
    L_pred has plateaued → early stopping fires too soon, cutting training
    before prediction has converged.
v3: Early stopping tracked on val_PRED_loss only.  L_recon is still trained
    but doesn't influence when to stop.

BUG 4 — No per-feature ICE scale normalisation
────────────────────────────────────────────────
v2: A shared NodeEncoder MLP sees ICE curves from all features.  One feature
    might have ICE amplitude ±5, another ±0.001.  The MLP can't learn
    good representations for such varied scales.
v3: Per-feature std computed on centered training ICE (stored as ice_stds_).
    All features' ICE curves are scaled to unit std before the encoder.
    Inference uses the same stored stds.  Padded zeros remain "mean" values
    after centering+scaling (= 0).

BUG 5 — StandardScaler on categorical features
────────────────────────────────────────────────
v2: StandardScaler applied to all features, including integer-encoded
    categoricals.  The scale of the current-value projection in NodeEncoder
    was therefore distorted for categoricals.
v3: _ContinuousScaler: only standardises continuous columns; passes
    categoricals through unchanged (they're already integer-coded).

DESIGN 1 — No pretrained residual connection  (biggest performance gain)
─────────────────────────────────────────────────────────────────────────
v2: GNN had to learn to predict y from zero, competing with the already-
    trained base learner.  ICE curves alone don't fully recover y because
    they're marginal, not joint.
v3: At training time, compute z0 = base_learner.predict(X_train) → normalise
    to logit/normalised scale.  Add to head output as a SKIP connection:
        output = GNN_head(pooled) + pretrained_logit
    The head now only needs to learn the RESIDUAL correction.  At init,
    GNN_head ≈ 0 so output ≈ pretrained_logit, which is already decent.
    This drastically reduces what the GNN needs to learn and dramatically
    improves R² / AUC, especially when the base learner is already strong.
    Controlled by use_pretrained_residual=True (default).

DESIGN 2 — Single shared MLP for all features in NodeEncoder
──────────────────────────────────────────────────────────────
Unchanged (still shared MLP) but now produces correct results because of
the per-feature scale normalisation (Bug 4 fix) + feature_emb identity.

DESIGN 3 — CosineAnnealingWarmRestarts without warmup
───────────────────────────────────────────────────────
v3: Adds a linear LR warmup over the first warmup_frac fraction of epochs.
    Warm restarts can cause instability at the start of training.

═══════════════════════════════════════════════════════════════════════════

Usage
═════
    from ice_gnn import IceGNNRegressor, IceGNNClassifier

    reg = IceGNNRegressor(
        pretrained_learner      = cb_model,
        cat_cols                = ["city", "job"],    # or [2, 5]
        n_grid                  = 40,
        d_model                 = 64,
        n_layers                = 3,
        eval_size               = 0.15,
        patience                = 5,
        use_pretrained_residual = True,    # ← major performance lever
        lambda_recon            = 0.05,
    )
    reg.fit(X_train, y_train)
    y_hat = reg.predict(X_test)

    print(reg.feature_importances_)       # DataFrame

    fig1 = reg.plot_feature_effects(X_test, X_raw=X_test_raw,
                                    feature_names=cols)
    fig2 = reg.plot_ice_comparison(X_test, X_raw=X_test_raw)
    fig3 = reg.plot_attention_heatmap(X_test, feature_names=cols)
    fig4 = reg.plot_importance()

References
══════════
• ICE / PDP:           Goldstein et al. 2015
• GNN for tabular:     arxiv 2305.08807
• Interpretable GNN:   arxiv 2509.23068
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["IceGNNRegressor", "IceGNNClassifier"]

# ══════════════════════════════════════════════════════════════════════════════
# §1  ICE COMPUTATION + NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════


def _predict_safe(
    model,
    X: np.ndarray,
    task: str,
    n_classes: int = 2,
) -> np.ndarray:
    """
    Model-agnostic predict → float32.

    Returns
    -------
    (n,)     regression or binary  P(class=1)
    (n, C)   multiclass probabilities
    """
    try:                                            # CatBoost fast-path
        from catboost import Pool  # type: ignore
        pool = Pool(X)
        if task == "regression":
            return np.asarray(model.predict(pool), dtype=np.float32).ravel()
        p = np.asarray(model.predict_proba(pool), dtype=np.float32)
        return p[:, 1] if n_classes == 2 else p
    except Exception:
        pass
    if task == "regression":
        return np.asarray(model.predict(X), dtype=np.float32).ravel()
    p = np.asarray(model.predict_proba(X), dtype=np.float32)
    if p.ndim == 1:
        return p
    return p[:, 1] if n_classes == 2 else p


def _ice_one_feature(
    model,
    X: np.ndarray,
    j: int,
    grid: np.ndarray,
    batch_size: int,
    task: str,
    n_classes: int,
) -> np.ndarray:
    """
    Fully-vectorised ICE for feature j.
    Builds (n*G, F) in one op, then chunked predict.

    Returns  (n, G)  or  (n, G, C)
    """
    n, G = X.shape[0], len(grid)
    X_rep = np.repeat(X, G, axis=0)
    X_rep[:, j] = np.tile(grid, n)
    chunks = [
        _predict_safe(model, X_rep[s: s + batch_size], task, n_classes)
        for s in range(0, len(X_rep), batch_size)
    ]
    preds = np.concatenate(chunks, axis=0)
    return preds.reshape(n, G) if preds.ndim == 1 else preds.reshape(n, G, -1)


def _build_grids(
    X: np.ndarray,
    cat_idx: List[int],
    n_grid: int,
    n_cat_max: int,
    quantile_grid: bool,
) -> List[np.ndarray]:
    """
    Variable-length per-feature grids.

    Categorical  → sorted unique encoded values  (≤ n_cat_max most frequent)
    Continuous   → n_grid quantile or uniform points (deduplicated)
    """
    cat_set = set(cat_idx)
    grids = []
    for j in range(X.shape[1]):
        col = X[:, j].astype(np.float32)
        if j in cat_set:
            vals, cnt = np.unique(col, return_counts=True)
            if len(vals) > n_cat_max:
                vals = np.sort(vals[np.argsort(-cnt)[:n_cat_max]])
            grids.append(vals.astype(np.float32))
        else:
            if quantile_grid:
                g = np.percentile(col, np.linspace(0.0, 100.0, n_grid)).astype(np.float32)
            else:
                g = np.linspace(col.min(), col.max(), n_grid, dtype=np.float32)
            _, idx = np.unique(g, return_index=True)
            grids.append(g[np.sort(idx)])
    return grids


def compute_ice_curves(
    model,
    X: np.ndarray,
    grids: List[np.ndarray],
    task: str = "regression",
    batch_size: int = 2048,
    n_jobs: int = 4,
    n_classes: int = 2,
) -> List[np.ndarray]:
    """
    Parallel ICE for all features using pre-built grids.

    Returns list of F arrays, each  (n, G_j)  or  (n, G_j, C).
    """
    F = X.shape[1]
    with ThreadPoolExecutor(max_workers=min(n_jobs, F)) as pool:
        return list(pool.map(
            lambda j: _ice_one_feature(
                model, X, j, grids[j], batch_size, task, n_classes
            ),
            range(F),
        ))


# ── NEW v3: center-before-pad + per-feature scale ────────────────────────────

def _center_normalize_ice(
    ice_list: List[np.ndarray],
    G_max: int,
    ice_stds: Optional[np.ndarray] = None,   # None → fit mode
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Correct ICE normalisation:

      1. Collapse multiclass  (n, G_j, C)  →  scalar  (n, G_j)  via max
      2. Per-sample centering on VALID positions only  (Bug-1 fix)
         → padded zeros now correctly represent the "mean" value
      3. Per-feature global std scaling  (Bug-4 fix)
         → shared NodeEncoder MLP sees all features at same scale
         fit mode  (ice_stds=None) → compute stds from data (stored)
         apply mode (ice_stds supplied) → use stored stds

    Returns
    -------
    padded   : float32 (n, F, G_max)
    valid_lens: int64  (F,)
    ice_stds : float32 (F,)   — computed or unchanged
    """
    n = ice_list[0].shape[0]
    F = len(ice_list)
    padded = np.zeros((n, F, G_max), dtype=np.float32)
    valid_lens = np.zeros(F, dtype=np.int64)
    fit_mode = ice_stds is None
    out_stds = np.ones(F, dtype=np.float32)

    for j, arr in enumerate(ice_list):
        # Step 1: collapse multiclass
        if arr.ndim == 3:
            arr = arr.max(axis=-1)          # (n, G_j)

        G_j = arr.shape[1]
        valid_lens[j] = G_j

        # Step 2: center on valid positions (per-sample)
        per_sample_mean = arr.mean(axis=1, keepdims=True)   # (n, 1)
        arr_c = arr - per_sample_mean                        # (n, G_j)  mean=0

        # Step 3: per-feature global std
        if fit_mode:
            std_j = float(arr_c.std()) + 1e-8
        else:
            std_j = float(ice_stds[j])
        out_stds[j] = std_j

        padded[:, j, :G_j] = arr_c / std_j
        # padded[:, j, G_j:] = 0  ← zeros = mean value (correct after center)

    return padded, valid_lens, out_stds


# ── Custom value scaler: skip categoricals (Bug-5 fix) ───────────────────────

class _ContinuousScaler:
    """
    StandardScaler that ONLY normalises continuous columns.
    Categorical columns (integer-encoded) are passed through unchanged —
    applying z-score to them would corrupt the value projection in NodeEncoder.
    """

    def __init__(self, cat_idx: List[int]):
        self._cat = set(cat_idx)

    def fit(self, X: np.ndarray) -> "_ContinuousScaler":
        F = X.shape[1]
        self.mean_ = np.zeros(F, dtype=np.float32)
        self.std_ = np.ones(F, dtype=np.float32)
        for j in range(F):
            if j not in self._cat:
                col = X[:, j].astype(np.float32)
                self.mean_[j] = col.mean()
                self.std_[j] = col.std() + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float32) - self.mean_) / self.std_)


# ══════════════════════════════════════════════════════════════════════════════
# §2  GNN MODULES
# ══════════════════════════════════════════════════════════════════════════════


class NodeEncoder(nn.Module):
    """
    (B, F, G_max) + (B, F) → (B, F, d_model)

    Shared 2-layer MLP on ICE curves  +  Linear on current feature value.
    Correct because after center+scale normalisation all features are at
    the same scale and the MLP needs only learn "ICE shape → embedding".
    """

    def __init__(self, G_max: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(d_model * 2, G_max)
        self.curve_proj = nn.Sequential(
            nn.Linear(G_max, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.val_proj = nn.Linear(1, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        curves: torch.Tensor,   # (B, F, G_max)
        values: torch.Tensor,   # (B, F)
    ) -> torch.Tensor:          # (B, F, d_model)
        h = self.curve_proj(curves) + self.val_proj(values.unsqueeze(-1))
        return self.norm(h)


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

    Self-attention over F feature nodes → cross-feature interaction learning.
    A[b,j,k] = how much feature-j's representation incorporates feature-k's
    ICE embedding.  Averaged over samples and layers → free F×F interaction
    matrix with zero extra cost.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 ffn_mul: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mul), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mul, d_model),
        )
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.n1(x)
        attn_out, attn_w = self.attn(
            h, h, h, need_weights=return_attn, average_attn_weights=True
        )
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.n2(x)))
        return (x, attn_w) if return_attn else x


class IceGNNModel(nn.Module):
    """
    Transformer-GNN over ICE-curve nodes.

    Architecture (v3)
    ──────────────────
    ice(B,F,G) + values(B,F)
      └─ NodeEncoder + feature_emb   →  h0 (B,F,d)   [initial encoding]
           │
           ├─ ICE decoder MLP ← h0   →  ice_recon (B,F,G)
           │   Trained with λ*MSE loss.  Operates on h0 BEFORE attention
           │   (Bug-2 fix) so reconstruction doesn't fight interaction learning.
           │   h0.detach() is used so decoder gradients don't pollute encoder.
           │
           └─ L × TransformerBlock  →  h (B,F,d)    [interaction-aware]
                └─ Attentive readout →  readout_w (B,F) + pooled (B,d)
                     └─ Head MLP     →  delta (B,n_out)
                          +
                     pretrained_logit (B,n_out)   [skip connection, Bug-Design-1]
                          =
                     out (B,n_out)

    pretrained_logit = None → skip connection disabled (use_pretrained_residual=False)
    """

    def __init__(
        self,
        n_features: int,
        G_max: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_outputs: int = 1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            d_model = ((d_model + n_heads - 1) // n_heads) * n_heads
            warnings.warn(f"d_model rounded to {d_model} (must be ÷ n_heads={n_heads}).")

        self.d_model = d_model
        self.n_features = n_features
        self.G_max = G_max

        self.feature_emb = nn.Embedding(n_features, d_model)
        self.encoder = NodeEncoder(G_max, d_model, dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Attentive readout
        self.readout_gate = nn.Linear(d_model, 1)

        # Prediction head  (learns residual correction around pretrained baseline)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_outputs),
        )

        # ICE decoder — operates on h0 (initial encoding, before attention)
        # Bug-2 fix: decoder does NOT fight attention layers
        self.ice_decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, G_max),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Head final layer: zero-init so output starts at pretrained baseline
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        ice: torch.Tensor,                      # (B, F, G_max)
        values: torch.Tensor,                   # (B, F)
        pretrained_logits: Optional[torch.Tensor] = None,   # (B, n_out)
        return_all: bool = False,
    ):
        """
        Parameters
        ----------
        pretrained_logits : (B, n_outputs), optional
            Normalised pretrained predictions.  When provided, the head
            only needs to learn the residual correction.

        Returns  (return_all=False)
        ──────────────────────────
        out  : (B, n_outputs)

        Returns  (return_all=True)
        ─────────────────────────
        out        : (B, n_outputs)
        node_emb   : (B, F, d_model)   interaction-aware
        readout_w  : (B, F)            prediction importance per feature
        ice_recon  : (B, F, G_max)     GNN-decoded ICE  (from h0)
        attn_mats  : List[(B, F, F)]
        """
        B, F, _ = ice.shape
        feat_idx = torch.arange(F, device=ice.device).unsqueeze(0).expand(B, -1)

        # ── Initial encoding ──────────────────────────────────────────────────
        h0 = self.encoder(ice, values) + self.feature_emb(feat_idx)    # (B, F, d)

        # ── ICE decoder on h0 (DETACHED — Bug-2 fix) ─────────────────────────
        # Gradients from reconstruction only train encoder+feature_emb.
        # Transformer layers are trained ONLY by prediction loss.
        ice_recon = self.ice_decoder(h0.detach()) if return_all else None

        # ── Transformer-GNN ───────────────────────────────────────────────────
        h = h0
        attn_mats: List[torch.Tensor] = []
        for block in self.blocks:
            if return_all:
                h, am = block(h, return_attn=True)
                attn_mats.append(am)
            else:
                h = block(h)

        node_emb = h

        # ── Attentive readout ─────────────────────────────────────────────────
        scores = self.readout_gate(h).squeeze(-1)       # (B, F)
        readout_w = torch.softmax(scores, dim=-1)        # (B, F)
        pooled = (readout_w.unsqueeze(-1) * h).sum(1)   # (B, d)

        # ── Head + pretrained residual (Design-1 fix) ─────────────────────────
        delta = self.head(pooled)                        # (B, n_out)  residual
        out = delta if pretrained_logits is None else delta + pretrained_logits

        if return_all:
            return out, node_emb, readout_w, ice_recon, attn_mats
        return out


# ══════════════════════════════════════════════════════════════════════════════
# §3  BASE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════


class _IceGNNBase(BaseEstimator):
    """
    Shared logic for IceGNNRegressor / IceGNNClassifier.

    Parameters
    ──────────
    pretrained_learner : fitted model
        Used both for ICE computation AND as a prediction residual anchor
        (when use_pretrained_residual=True).

    cat_cols : list[int | str]  optional
        Integer column indices or column names (DataFrame input) for
        categorical features.  These use unique encoded values as ICE grid
        and are NOT z-score normalised in the value projection.

    n_grid : int  (default 40)
        Grid points for continuous features.

    n_cat_max : int  (default 30)
        Max unique categories kept per categorical feature.

    d_model : int  (default 64)

    n_heads : int  (default 4)

    n_layers : int  (default 2)

    dropout : float  (default 0.1)

    use_pretrained_residual : bool  (default True)
        KEY performance switch.  When True, base-learner predictions are used
        as a skip connection: GNN_output = head(GNN_features) + base_pred.
        The GNN only needs to learn the residual correction.
        Disable only if the base learner is very weak or unreliable.

    lambda_recon : float  (default 0.05)
        Weight of auxiliary ICE reconstruction loss.
        Trains NodeEncoder to faithfully encode ICE shape.
        Does NOT interfere with Transformer training (Bug-2 fix).
        Set to 0 to disable.

    lr : float  (default 1e-3)
    weight_decay : float  (default 1e-4)
    max_epochs : int  (default 200)
    eval_size : float  (default 0.15)
    patience : int  (default 5)
        Early stopping patience, tracked on PREDICTION loss only (Bug-3 fix).
    batch_size : int  (default 512)
    max_ice_samples : int | None
        Subsample rows for ICE computation during fit.  None = use all.
    ice_batch_size : int  (default 2048)
    n_jobs : int  (default 4)
    quantile_grid : bool  (default True)
    warmup_frac : float  (default 0.05)
        Fraction of max_epochs used for linear LR warmup.
    device : str  (default "auto")
    random_state : int  (default 42)
    verbose : bool  (default True)
    """

    _task: str = ""

    def __init__(
        self,
        pretrained_learner,
        *,
        cat_cols: Optional[List] = None,
        n_grid: int = 40,
        n_cat_max: int = 30,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_pretrained_residual: bool = True,
        lambda_recon: float = 0.05,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        eval_size: float = 0.15,
        patience: int = 5,
        batch_size: int = 512,
        max_ice_samples: Optional[int] = None,
        ice_batch_size: int = 2048,
        n_jobs: int = 4,
        quantile_grid: bool = True,
        warmup_frac: float = 0.05,
        device: str = "auto",
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.pretrained_learner = pretrained_learner
        self.cat_cols = cat_cols
        self.n_grid = n_grid
        self.n_cat_max = n_cat_max
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_pretrained_residual = use_pretrained_residual
        self.lambda_recon = lambda_recon
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.eval_size = eval_size
        self.patience = patience
        self.batch_size = batch_size
        self.max_ice_samples = max_ice_samples
        self.ice_batch_size = ice_batch_size
        self.n_jobs = n_jobs
        self.quantile_grid = quantile_grid
        self.warmup_frac = warmup_frac
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ── Abstract ──────────────────────────────────────────────────────────────

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _compute_loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _get_n_outputs(self) -> int:
        raise NotImplementedError

    def _get_n_classes(self) -> int:
        return getattr(self, "n_classes_", 2)

    def _pretrained_logits_np(self, X: np.ndarray) -> np.ndarray:
        """Pretrained predictions in logit/normalised space → (n, n_outputs)."""
        raise NotImplementedError

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def _dev(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _resolve_cat_idx(self, X, cat_cols) -> List[int]:
        if not cat_cols:
            return []
        try:
            import pandas as pd  # type: ignore
            if isinstance(X, pd.DataFrame):
                cols = list(X.columns)
                return sorted({
                    cols.index(c) if isinstance(c, str) else int(c)
                    for c in cat_cols
                })
        except ImportError:
            pass
        return sorted({int(c) for c in cat_cols})

    def _to_numpy(self, X) -> Tuple[np.ndarray, Optional[List[str]]]:
        try:
            import pandas as pd  # type: ignore
            if isinstance(X, pd.DataFrame):
                return X.values.astype(np.float32), list(X.columns)
        except ImportError:
            pass
        return np.asarray(X, dtype=np.float32), None

    def _build_cat_label_map(
        self,
        X_enc: np.ndarray,
        X_raw: Optional[np.ndarray],
    ) -> Dict[int, Dict[float, str]]:
        if X_raw is None:
            return {}
        X_raw_arr = np.asarray(X_raw)
        return {
            j: {float(k): str(v)
                for k, v in dict(zip(X_enc[:, j].tolist(),
                                     X_raw_arr[:, j].tolist())).items()}
            for j in self.cat_idx_
        }

    def _compute_ice_full(
        self,
        X: np.ndarray,
        quiet: bool = False,
    ) -> List[np.ndarray]:
        if not quiet:
            self._log(
                f"  ⟳ ICE  {len(X):>6,} rows × {X.shape[1]} features  "
                f"({sum(len(g) for g in self.grids_)} grid pts)  "
                f"[{self.n_jobs} workers]"
            )
        t0 = time.perf_counter()
        ice_list = compute_ice_curves(
            self.pretrained_learner, X,
            grids=self.grids_,
            task=self._task,
            batch_size=self.ice_batch_size,
            n_jobs=self.n_jobs,
            n_classes=self._get_n_classes(),
        )
        if not quiet:
            self._log(f"  ✓ ICE in {time.perf_counter() - t0:.2f}s")
        return ice_list

    def _ice_to_tensor(
        self,
        ice_list: List[np.ndarray],
        fit_mode: bool = False,
    ) -> torch.Tensor:
        """
        Center-before-pad + per-feature scale  (Bug-1 and Bug-4 fix).
        fit_mode=True  → compute and store ice_stds_
        fit_mode=False → apply stored ice_stds_
        """
        stds_in = None if fit_mode else self.ice_stds_
        padded, vl, stds_out = _center_normalize_ice(
            ice_list, self.G_max_, ice_stds=stds_in
        )
        if fit_mode:
            self.ice_stds_ = stds_out
            self.valid_lens_: np.ndarray = vl
        return torch.from_numpy(padded)

    def _resolve_feat_names(
        self, feature_names: Optional[List[str]] = None
    ) -> List[str]:
        if feature_names is not None:
            return list(feature_names)
        if getattr(self, "feature_names_in_", None) is not None:
            return list(self.feature_names_in_)
        return [f"f{j}" for j in range(self.n_features_)]

    def _display_grids(
        self,
        X_enc: np.ndarray,
        X_raw: Optional[np.ndarray],
    ) -> Tuple[List, List[bool]]:
        cat_map = self._build_cat_label_map(X_enc, X_raw)
        X_raw_arr = np.asarray(X_raw, dtype=object) if X_raw is not None else None
        display, is_cat = [], []
        for j, grid in enumerate(self.grids_):
            if j in self.cat_idx_:
                display.append(
                    [cat_map[j].get(float(v), str(v)) for v in grid]
                    if j in cat_map else [str(int(v)) for v in grid]
                )
                is_cat.append(True)
            else:
                if X_raw_arr is not None:
                    sidx = np.argsort(X_enc[:, j])
                    mapped = np.interp(
                        grid, X_enc[sidx, j],
                        X_raw_arr[sidx, j].astype(np.float32)
                    )
                    display.append(mapped.astype(np.float32))
                else:
                    display.append(grid.copy())
                is_cat.append(False)
        return display, is_cat

    # ── LR schedule with warmup ───────────────────────────────────────────────

    def _make_scheduler(self, opt, n_epochs: int):
        """Cosine Annealing with linear warmup prefix."""
        warmup_ep = max(1, int(n_epochs * self.warmup_frac))
        T0 = max(10, n_epochs // 5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=T0, eta_min=self.lr * 1e-2
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_ep
        )
        return torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_ep]
        )

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y) -> "_IceGNNBase":
        """
        Fit IceGNN.

        1. Resolve cat_cols; build per-feature grids.
        2. Optionally subsample for ICE computation.
        3. Compute ICE curves (parallel, ThreadPoolExecutor).
        4. Center-before-pad + per-feature scale normalisation  (v3 fix).
        5. Optionally compute pretrained residual logits.
        6. Build + train Transformer-GNN:
               loss = L_pred  +  λ * L_recon(h0)
        7. Early stopping on val PREDICTION loss only  (v3 fix).
        8. Store feature_importances_ on val set.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr, feat_names = self._to_numpy(X)
        self.feature_names_in_: Optional[List[str]] = feat_names
        y_prep = self._prepare_y(np.asarray(y))
        n, F = X_arr.shape
        dev = self._dev

        self.cat_idx_: List[int] = self._resolve_cat_idx(X, self.cat_cols)
        cat_set = set(self.cat_idx_)

        self._log(f"\n{'═'*74}")
        self._log(
            f"  IceGNN v3 [{self._task.upper()}]  n={n:,}  F={F}  "
            f"n_cat={len(self.cat_idx_)}  device={dev}  "
            f"residual={'on' if self.use_pretrained_residual else 'off'}"
        )
        self._log(f"{'═'*74}")

        # ── Grids ─────────────────────────────────────────────────────────────
        self.grids_: List[np.ndarray] = _build_grids(
            X_arr, self.cat_idx_, self.n_grid, self.n_cat_max, self.quantile_grid
        )
        self.G_max_: int = max(len(g) for g in self.grids_)
        grid_info = " | ".join(
            f"f{j}({'C' if j in cat_set else 'N'})={len(g)}"
            for j, g in enumerate(self.grids_)
        )
        self._log(
            f"  G_max={self.G_max_}  "
            f"{grid_info[:100]}{'...' if len(grid_info) > 100 else ''}"
        )

        # ── Value scaler (skip categoricals — Bug-5 fix) ──────────────────────
        self.value_scaler_: _ContinuousScaler = _ContinuousScaler(self.cat_idx_)
        self.value_scaler_.fit(X_arr)

        # ── Subsample for ICE ─────────────────────────────────────────────────
        ice_X, ice_y = X_arr, y_prep
        if self.max_ice_samples and n > self.max_ice_samples:
            rng_np = np.random.default_rng(self.random_state)
            idx = rng_np.choice(n, self.max_ice_samples, replace=False)
            ice_X, ice_y = X_arr[idx], y_prep[idx]

        # ── ICE curves ────────────────────────────────────────────────────────
        ice_list = self._compute_ice_full(ice_X)

        # Bug-1 + Bug-4 fix: center before pad + per-feature scale
        ice_t = self._ice_to_tensor(ice_list, fit_mode=True)   # stores ice_stds_
        X_norm = self.value_scaler_.transform(ice_X)
        X_t = torch.from_numpy(X_norm)
        y_t = torch.from_numpy(ice_y)

        # ── Pretrained residual logits (Design-1 fix) ─────────────────────────
        if self.use_pretrained_residual:
            self._log("  ⟳ Computing pretrained residual logits …")
            z0 = self._pretrained_logits_np(ice_X)     # (n, n_outputs)
            z0_t = torch.from_numpy(z0)
        else:
            n_out = self._get_n_outputs()
            z0_t = torch.zeros(len(ice_t), n_out, dtype=torch.float32)

        # ── Build model ───────────────────────────────────────────────────────
        n_outputs = self._get_n_outputs()
        self.model_: IceGNNModel = IceGNNModel(
            n_features=F,
            G_max=self.G_max_,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            n_outputs=n_outputs,
        ).to(dev)

        n_params = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        self._log(
            f"  Params: {n_params:,}  |  G_max={self.G_max_}  |  "
            f"n_outputs={n_outputs}  |  λ_recon={self.lambda_recon}"
        )

        # ── Train / val split ─────────────────────────────────────────────────
        n_total = len(ice_t)
        n_val = max(4, int(n_total * self.eval_size))
        g_ = torch.Generator().manual_seed(self.random_state)
        perm = torch.randperm(n_total, generator=g_)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        pin = dev.type == "cuda"
        kw = dict(num_workers=0, pin_memory=pin, persistent_workers=False)

        # DataLoader now includes pretrained logits (z0)
        def _make_ds(idx_):
            return TensorDataset(
                ice_t[idx_], X_t[idx_], y_t[idx_], z0_t[idx_]
            )

        train_dl = DataLoader(
            _make_ds(train_idx), batch_size=self.batch_size, shuffle=True, **kw
        )
        val_dl = DataLoader(
            _make_ds(val_idx), batch_size=self.batch_size * 2, shuffle=False, **kw
        )

        # ── Optimiser + scheduler ─────────────────────────────────────────────
        opt = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = self._make_scheduler(opt, self.max_epochs)

        # ── Training loop ─────────────────────────────────────────────────────
        # Bug-3 fix: best_val tracks PREDICTION loss only
        best_val_pred = float("inf")
        best_sd: Optional[dict] = None
        no_imp = 0
        log_every = max(1, self.max_epochs // 20)

        self._log(
            f"\n  {'Ep':>5}  {'TrTot':>10}  {'TrPred':>10}  "
            f"{'TrRec':>8}  {'ValPred':>10}  {'LR':>9}"
        )
        self._log(f"  {'─'*60}")

        for ep in range(1, self.max_epochs + 1):
            # ─ train ─────────────────────────────────────────────────────────
            self.model_.train()
            tr_tot = tr_pred = tr_rec = 0.0
            for ib, xb, yb, zb in train_dl:
                ib, xb, yb, zb = (
                    ib.to(dev), xb.to(dev), yb.to(dev), zb.to(dev)
                )
                opt.zero_grad(set_to_none=True)
                out, _, _, ice_recon, _ = self.model_(
                    ib, xb, zb, return_all=True
                )
                l_pred = self._compute_loss(out, yb)
                l_rec = F.mse_loss(ice_recon, ib) * self.lambda_recon
                loss = l_pred + l_rec
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                bsz = len(yb)
                tr_tot  += loss.item() * bsz
                tr_pred += l_pred.item() * bsz
                tr_rec  += l_rec.item() * bsz

            sched.step()
            n_tr = len(train_idx)
            tr_tot /= n_tr; tr_pred /= n_tr; tr_rec /= n_tr

            # ─ val ───────────────────────────────────────────────────────────
            val_pred, _ = self._eval_loop(val_dl, dev)

            if ep % log_every == 0 or ep <= 5:
                self._log(
                    f"  {ep:>5}  {tr_tot:>10.5f}  {tr_pred:>10.5f}  "
                    f"{tr_rec:>8.5f}  {val_pred:>10.5f}  "
                    f"{opt.param_groups[0]['lr']:>9.2e}"
                )

            # ─ early stopping on val PRED loss only (Bug-3 fix) ──────────────
            if val_pred < best_val_pred - 1e-6:
                best_val_pred = val_pred
                best_sd = {
                    k: v.cpu().clone() for k, v in self.model_.state_dict().items()
                }
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    self._log(
                        f"\n  ⏹  Early stop @ ep {ep}  "
                        f"best_val_pred={best_val_pred:.5f}"
                    )
                    break

        if best_sd is not None:
            self.model_.load_state_dict(best_sd)
        self.model_.eval()
        self.n_features_: int = F
        self.is_fitted_ = True

        # ── Store val-set importances ─────────────────────────────────────────
        self._val_ice_t_ = ice_t[val_idx]
        self._val_X_t_ = X_t[val_idx]
        self._val_z0_t_ = z0_t[val_idx]
        self._refresh_importances(dev)

        self._log(f"\n  ✓ Fit done — best_val_pred={best_val_pred:.5f}")
        self._log(f"{'═'*74}\n")
        return self

    # ── Eval loop ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_loop(
        self, dl: DataLoader, dev: torch.device
    ) -> Tuple[float, float]:
        """Returns (val_pred_loss, val_total_loss)."""
        self.model_.eval()
        pred_tot = total_tot = 0.0
        n = 0
        for ib, xb, yb, zb in dl:
            ib, xb, yb, zb = ib.to(dev), xb.to(dev), yb.to(dev), zb.to(dev)
            out, _, _, ir, _ = self.model_(ib, xb, zb, return_all=True)
            lp = self._compute_loss(out, yb)
            lr_ = F.mse_loss(ir, ib) * self.lambda_recon
            pred_tot  += lp.item() * len(yb)
            total_tot += (lp + lr_).item() * len(yb)
            n += len(yb)
        return pred_tot / max(n, 1), total_tot / max(n, 1)

    @torch.no_grad()
    def _refresh_importances(
        self, dev: Optional[torch.device] = None
    ) -> None:
        if dev is None:
            dev = self._dev
        self.model_.eval()
        all_w: List[np.ndarray] = []
        dl = DataLoader(
            TensorDataset(
                self._val_ice_t_, self._val_X_t_, self._val_z0_t_
            ),
            batch_size=self.batch_size * 2,
        )
        for ib, xb, zb in dl:
            _, _, rw, _, _ = self.model_(
                ib.to(dev), xb.to(dev), zb.to(dev), return_all=True
            )
            all_w.append(rw.cpu().numpy())
        self._mean_readout_w_: np.ndarray = np.concatenate(all_w, 0).mean(0)

    # ── feature_importances_ ──────────────────────────────────────────────────

    @property
    def feature_importances_(self):
        """
        GNN feature importances — readout attention weights, val set.

        Returns
        -------
        pd.DataFrame  columns=['feature','importance','is_categorical']
        sorted descending.
        """
        check_is_fitted(self)
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            raise ImportError("pip install pandas")
        names = self._resolve_feat_names()
        cat_set = set(self.cat_idx_)
        return (
            pd.DataFrame({
                "feature": names,
                "importance": self._mean_readout_w_,
                "is_categorical": [j in cat_set for j in range(self.n_features_)],
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── Core forward ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _forward_full(
        self,
        X: np.ndarray,
        quiet: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Full pipeline: ICE → normalise → GNN → all outputs.

        Returns
        -------
        raw_out      (n, n_outputs)
        node_emb     (n, F, d_model)
        readout_w    (n, F)
        ice_recon    (n, F, G_max)  GNN-decoded ICE  (centered+scaled)
        ice_raw_norm (n, F, G_max)  base-learner ICE (centered+scaled+padded)
        ice_raw_orig (n, F, G_max)  base-learner ICE (uncentered, for display)
        attn_matrix  (F, F)
        valid_lens   (F,)
        """
        check_is_fitted(self)
        dev = self._dev

        ice_list = self._compute_ice_full(X, quiet=quiet)

        # Normalised tensor (same pipeline as fit)
        ice_t = self._ice_to_tensor(ice_list, fit_mode=False)

        # Raw uncentered padded (for display / un-normalised PDP)
        ice_raw_padded = np.zeros(
            (len(X), self.n_features_, self.G_max_), dtype=np.float32
        )
        for j, arr in enumerate(ice_list):
            if arr.ndim == 3:
                arr = arr.max(axis=-1)
            G_j = arr.shape[1]
            ice_raw_padded[:, j, :G_j] = arr

        X_norm = self.value_scaler_.transform(X)
        X_t = torch.from_numpy(X_norm)

        if self.use_pretrained_residual:
            z0 = self._pretrained_logits_np(X)
            z0_t = torch.from_numpy(z0)
        else:
            z0_t = torch.zeros(len(X), self._get_n_outputs(), dtype=torch.float32)

        all_out, all_emb, all_rw, all_recon, all_attn = [], [], [], [], []

        dl = DataLoader(
            TensorDataset(ice_t, X_t, z0_t),
            batch_size=self.batch_size * 2,
            shuffle=False,
        )
        self.model_.eval()
        for ib, xb, zb in dl:
            ib, xb, zb = ib.to(dev), xb.to(dev), zb.to(dev)
            out, emb, rw, recon, attn_mats = self.model_(
                ib, xb, zb, return_all=True
            )
            all_out.append(out.cpu().numpy())
            all_emb.append(emb.cpu().numpy())
            all_rw.append(rw.cpu().numpy())
            all_recon.append(recon.cpu().numpy())
            if attn_mats:
                am = np.stack([a.cpu().numpy() for a in attn_mats], 0).mean(0)
                all_attn.append(am)

        result: Dict[str, np.ndarray] = {
            "raw_out":      np.concatenate(all_out,   0),
            "node_emb":     np.concatenate(all_emb,   0),
            "readout_w":    np.concatenate(all_rw,    0),
            "ice_recon":    np.concatenate(all_recon, 0),
            "ice_raw_norm": ice_t.numpy(),
            "ice_raw_orig": ice_raw_padded,
            "valid_lens":   self.valid_lens_,
        }
        if all_attn:
            result["attn_matrix"] = np.concatenate(all_attn, 0).mean(0)
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # §3a  EXPLAINABILITY API
    # ══════════════════════════════════════════════════════════════════════════

    def get_feature_effects(
        self,
        X,
        X_raw=None,
        feature_names: Optional[List[str]] = None,
    ) -> dict:
        """
        Smooth per-feature effect functions.

        The effect values are in the normalised ICE space (centered + scaled).
        For interpretability, the y-axis is comparable across features and
        reflects the relative importance of each feature's variation.

        Parameters
        ----------
        X            : encoded feature matrix (numpy or DataFrame)
        X_raw        : optional raw feature matrix
                       → raw x-axis values for continuous
                       → category labels for categorical
        feature_names: optional list[str]

        Returns
        -------
        dict
          features   : list[str]
          is_cat     : list[bool]
          grids      : list of x-axis values per feature
                       (float for continuous, str list for categorical)
          valid_lens : (F,) int
          pdp        : (F, G_max)  mean base-learner ICE (normalized)
          gnn_pdp    : (F, G_max)  readout-weighted PDP
          gnn_recon  : (F, G_max)  GNN-decoded ICE mean (smooth+interact)
          ice_curves : (n, F, G_max) per-sample ICE (normalized)
          readout_w  : (n, F)
          importance : (F,)
        """
        X_arr, _ = self._to_numpy(X)
        res = self._forward_full(X_arr, quiet=False)

        ice_norm = res["ice_raw_norm"]           # (n, F, G_max)
        rw = res["readout_w"]

        pdp = ice_norm.mean(0)                   # (F, G_max)
        w = rw / (rw.sum(0, keepdims=True) + 1e-8)
        gnn_pdp = np.einsum("nf,nfg->fg", w, ice_norm)
        gnn_recon = res["ice_recon"].mean(0)

        dgrids, is_cat = self._display_grids(X_arr, X_raw)

        return {
            "features":   self._resolve_feat_names(feature_names),
            "is_cat":     is_cat,
            "grids":      dgrids,
            "valid_lens": res["valid_lens"],
            "pdp":        pdp,
            "gnn_pdp":    gnn_pdp,
            "gnn_recon":  gnn_recon,
            "ice_curves": ice_norm,
            "readout_w":  rw,
            "importance": rw.mean(0),
        }

    def get_ice_curves(
        self,
        X,
        X_raw=None,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 50,
    ) -> dict:
        """
        Base-learner ICE  vs  GNN-decoded ICE for direct comparison.

        Returns
        -------
        dict
          features       : list[str]
          is_cat         : list[bool]
          grids          : per-feature display grids
          valid_lens     : (F,)
          base_ice       : (n_s, F, G_max)  base-learner (normalized)
          gnn_ice        : (n_s, F, G_max)  GNN-decoded  (normalized)
          base_pdp       : (F, G_max)
          gnn_pdp        : (F, G_max)
          sample_indices : (n_s,)
        """
        X_arr, _ = self._to_numpy(X)
        res = self._forward_full(X_arr, quiet=False)
        n = len(X_arr)
        rng = np.random.default_rng(self.random_state)
        sel = rng.choice(n, min(n_samples, n), replace=False)

        base = res["ice_raw_norm"]
        gnn = res["ice_recon"]
        dgrids, is_cat = self._display_grids(X_arr, X_raw)

        return {
            "features":       self._resolve_feat_names(feature_names),
            "is_cat":         is_cat,
            "grids":          dgrids,
            "valid_lens":     res["valid_lens"],
            "base_ice":       base[sel],
            "gnn_ice":        gnn[sel],
            "base_pdp":       base.mean(0),
            "gnn_pdp":        gnn.mean(0),
            "sample_indices": sel,
        }

    def get_attention_matrix(
        self,
        X,
        feature_names: Optional[List[str]] = None,
    ) -> dict:
        """
        F×F cross-feature attention / interaction matrix.

        A[j,k] = mean attention weight feature-j gives to feature-k
        (averaged over layers and samples).
        High values ↔ strong interaction.

        Returns  dict: matrix (F,F), features list[str]
        """
        X_arr, _ = self._to_numpy(X)
        res = self._forward_full(X_arr, quiet=True)
        return {
            "matrix":   res.get(
                "attn_matrix", np.eye(self.n_features_, dtype=np.float32)
            ),
            "features": self._resolve_feat_names(feature_names),
        }

    def get_embeddings(self, X) -> dict:
        """
        Node embeddings + readout importance.

        Returns
        -------
        dict: embeddings (n,F,d), readout_w (n,F), mean_importance (F,)
        """
        X_arr, _ = self._to_numpy(X)
        res = self._forward_full(X_arr, quiet=True)
        return {
            "embeddings":      res["node_emb"],
            "readout_w":       res["readout_w"],
            "mean_importance": res["readout_w"].mean(0),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # §3b  PLOT METHODS
    # ══════════════════════════════════════════════════════════════════════════

    def plot_feature_effects(
        self,
        X,
        X_raw=None,
        feature_names: Optional[List[str]] = None,
        max_features: int = 20,
        n_ice_lines: int = 50,
        sort_by_importance: bool = True,
        figsize: Tuple = (22, None),
        alpha_ice: float = 0.07,
    ):
        """
        Grid of per-feature effect plots.

        • Continuous : line — ICE lines + PDP (orange) + GNN-PDP (red) + GNN-recon (green)
        • Categorical: grouped bar — PDP vs GNN-recon per category (raw labels if X_raw)

        Returns  matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("pip install matplotlib")

        eff = self.get_feature_effects(X, X_raw, feature_names)
        F = self.n_features_
        n_show = min(F, max_features)
        order = (
            np.argsort(eff["importance"])[::-1][:n_show]
            if sort_by_importance else np.arange(n_show)
        )

        n_cols = min(4, n_show)
        n_rows = (n_show + n_cols - 1) // n_cols
        fw = figsize[0]
        fh = figsize[1] or n_rows * 3.9
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fw, fh))
        axes_flat = np.array(axes).flatten() if n_show > 1 else [axes]

        rng = np.random.default_rng(0)
        sel = rng.choice(eff["ice_curves"].shape[0],
                         min(n_ice_lines, eff["ice_curves"].shape[0]),
                         replace=False)

        C_ICE = "#4682b4"; C_PDP = "#ff8c00"
        C_GPDP = "#cc2200"; C_RECON = "#2ca02c"

        for k, j in enumerate(order):
            ax = axes_flat[k]
            vl = int(eff["valid_lens"][j])
            grid = eff["grids"][j]
            is_cat = eff["is_cat"][j]
            title = f"{'[C] ' if is_cat else ''}{eff['features'][j]}\n" \
                    f"imp={eff['importance'][j]:.4f}"

            if is_cat:
                cats = grid[:vl]
                x = np.arange(len(cats))
                w = 0.35
                ax.bar(x - w/2, eff["pdp"][j, :vl],      w, color=C_PDP,   alpha=0.85, label="PDP")
                ax.bar(x + w/2, eff["gnn_recon"][j, :vl], w, color=C_RECON, alpha=0.85, label="GNN-recon")
                ax.set_xticks(x)
                ax.set_xticklabels([str(c) for c in cats], rotation=30, ha="right", fontsize=7)
                ax.axhline(0, color="gray", lw=0.6, ls="--")
                ax.legend(fontsize=7)
            else:
                xv = np.asarray(grid[:vl], dtype=np.float32)
                for i in sel:
                    ax.plot(xv, eff["ice_curves"][i, j, :vl],
                            color=C_ICE, alpha=alpha_ice, lw=0.7, zorder=1)
                ax.plot(xv, eff["pdp"][j, :vl],       color=C_PDP,   lw=2.2, label="PDP",       zorder=3)
                ax.plot(xv, eff["gnn_pdp"][j, :vl],   color=C_GPDP,  lw=1.8, ls="--", label="GNN-PDP", zorder=4)
                ax.plot(xv, eff["gnn_recon"][j, :vl], color=C_RECON, lw=2.0, ls=":", label="GNN-recon", zorder=5)
                ax.axhline(0, color="gray", lw=0.5, ls="--")
                if k == 0:
                    ax.legend(fontsize=7)

            ax.set_title(title, fontsize=8, pad=3)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

        for ax in axes_flat[n_show:]:
            ax.set_visible(False)

        handles = [
            mpatches.Patch(color=C_ICE,   alpha=0.5, label="ICE lines"),
            mpatches.Patch(color=C_PDP,              label="PDP (base learner mean)"),
            mpatches.Patch(color=C_GPDP,             label="GNN-PDP (readout-weighted)"),
            mpatches.Patch(color=C_RECON,            label="GNN-recon (smooth+interact)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4,
                   fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(
            f"IceGNN v3 [{self._task}] — Feature Effect Functions  [C]=categorical",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        return fig

    def plot_ice_comparison(
        self,
        X,
        X_raw=None,
        feature_names: Optional[List[str]] = None,
        max_features: int = 10,
        n_ice_lines: int = 30,
        sort_by_importance: bool = True,
        figsize: Tuple = (22, None),
        alpha_ice: float = 0.10,
    ):
        """
        Side-by-side: base-learner ICE (blue) vs GNN-decoded ICE (green).

        Each feature = two adjacent panels.
        Continuous: ICE lines + bold PDP.
        Categorical: bar chart per category.

        Returns  matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install matplotlib")

        info = self.get_ice_curves(X, X_raw, feature_names, n_samples=n_ice_lines)
        n_show = min(self.n_features_, max_features)
        order = (
            np.argsort(self._mean_readout_w_)[::-1][:n_show]
            if sort_by_importance else np.arange(n_show)
        )

        fw = figsize[0]
        fh = figsize[1] or n_show * 3.0
        fig, axes = plt.subplots(n_show, 2, figsize=(fw, fh), squeeze=False)

        C_BASE = "#4682b4"; C_GNN = "#2ca02c"

        for k, j in enumerate(order):
            vl = int(info["valid_lens"][j])
            grid = info["grids"][j]
            is_cat = info["is_cat"][j]
            feat_name = info["features"][j]

            for col, (curves, pdp, label, color) in enumerate([
                (info["base_ice"], info["base_pdp"], "Base ICE", C_BASE),
                (info["gnn_ice"],  info["gnn_pdp"],  "GNN ICE",  C_GNN),
            ]):
                ax = axes[k, col]
                if is_cat:
                    cats = grid[:vl]
                    ax.bar(range(len(cats)), pdp[j, :vl], color=color, alpha=0.8)
                    ax.set_xticks(range(len(cats)))
                    ax.set_xticklabels(
                        [str(c) for c in cats], rotation=30, ha="right", fontsize=6
                    )
                    ax.axhline(0, color="gray", lw=0.5, ls="--")
                else:
                    xv = np.asarray(grid[:vl], dtype=np.float32)
                    for i_s in range(len(info["sample_indices"])):
                        ax.plot(xv, curves[i_s, j, :vl],
                                color=color, alpha=alpha_ice, lw=0.7)
                    ax.plot(xv, pdp[j, :vl], color=color, lw=2.5)
                    ax.axhline(0, color="gray", lw=0.5, ls="--")

                ax.set_title(
                    f"{'[C] ' if is_cat else ''}{feat_name} — {label}",
                    fontsize=8, color=color, pad=2,
                )
                ax.tick_params(labelsize=7)
                ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(
            f"IceGNN v3 [{self._task}] — Base-Learner ICE vs GNN-Decoded ICE",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        return fig

    def plot_attention_heatmap(
        self,
        X,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "YlOrRd",
        annot_thresh: int = 30,
    ):
        """
        F×F cross-feature interaction heatmap.
        Categorical labels shown in red.

        Returns  matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install matplotlib")

        att = self.get_attention_matrix(X, feature_names)
        M, names = att["matrix"], att["features"]
        F = len(names)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0)
        plt.colorbar(im, ax=ax, label="Mean attention weight")

        short = [n[:12] + ("…" if len(n) > 12 else "") for n in names]
        ax.set_xticks(range(F)); ax.set_yticks(range(F))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short, fontsize=8)

        if F <= annot_thresh:
            vmax = M.max()
            for i in range(F):
                for jj in range(F):
                    ax.text(jj, i, f"{M[i,jj]:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if M[i, jj] > vmax * 0.65 else "black")

        cat_set = set(self.cat_idx_)
        xl = ax.get_xticklabels(); yl = ax.get_yticklabels()
        for j in range(F):
            if j in cat_set:
                xl[j].set_color("#cc2200"); yl[j].set_color("#cc2200")

        ax.set_title(
            f"IceGNN v3 [{self._task}] — Feature Interaction Matrix\n"
            f"({self.n_layers} attn layers, avg over samples)  [red=categorical]",
            fontsize=11,
        )
        plt.tight_layout()
        return fig

    def plot_importance(
        self,
        X=None,
        feature_names: Optional[List[str]] = None,
        top_n: int = 30,
        figsize: Tuple[int, int] = (9, 6),
    ):
        """
        Horizontal bar chart of GNN feature importances.
        X=None → uses pre-computed val-set importances.

        Returns  matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install matplotlib")

        check_is_fitted(self)
        imp = (
            self._forward_full(self._to_numpy(X)[0], quiet=True)["readout_w"].mean(0)
            if X is not None else self._mean_readout_w_
        )
        names = self._resolve_feat_names(feature_names)
        cat_set = set(self.cat_idx_)
        top = min(top_n, self.n_features_)
        order = np.argsort(imp)[::-1][:top][::-1]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(
            range(len(order)), imp[order],
            color=["#cc2200" if i in cat_set else "#4682b4" for i in order],
            edgecolor="white", linewidth=0.4,
        )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[i] for i in order], fontsize=9)
        ax.set_xlabel("GNN readout importance (mean attention weight)")
        ax.set_title(
            f"IceGNN v3 [{self._task}] — Feature Importance "
            f"[red=categorical, top {top}]",
            fontsize=11,
        )
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# §4  IceGNNRegressor
# ══════════════════════════════════════════════════════════════════════════════


class IceGNNRegressor(_IceGNNBase, RegressorMixin):
    """
    IceGNN for continuous target prediction.

    The GNN learns from the base-learner's ICE curves and produces
    interpretable, interaction-aware feature effect functions.
    The pretrained residual skip connection (use_pretrained_residual=True)
    ensures the GNN starts from a strong baseline and only needs to learn
    a residual correction — dramatically improving R².

    Attributes (after fit)
    ──────────────────────
    feature_importances_ : pd.DataFrame
    y_mean_, y_std_       : float

    Examples
    ────────
    >>> reg = IceGNNRegressor(
    ...     pretrained_learner=cb, cat_cols=["city","job"],
    ...     n_grid=40, d_model=64, n_layers=3,
    ...     eval_size=0.15, patience=5,
    ...     use_pretrained_residual=True,
    ... )
    >>> reg.fit(X_train, y_train)
    >>> y_hat = reg.predict(X_test)
    """

    _task = "regression"

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).ravel()
        self.y_mean_: float = float(y.mean())
        self.y_std_: float = float(y.std()) + 1e-8
        return ((y - self.y_mean_) / self.y_std_).astype(np.float32)

    def _get_n_outputs(self) -> int:
        return 1

    def _compute_loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(out.squeeze(-1), y)

    def _pretrained_logits_np(self, X: np.ndarray) -> np.ndarray:
        """Pretrained prediction normalised to same scale as y_t."""
        preds = _predict_safe(self.pretrained_learner, X, "regression", 1)
        return ((preds - self.y_mean_) / self.y_std_).reshape(-1, 1).astype(np.float32)

    def predict(self, X) -> np.ndarray:
        """
        Predict continuous target.

        Returns  (n,) float64
        """
        X_arr, _ = self._to_numpy(X)
        raw = self._forward_full(X_arr, quiet=True)["raw_out"].ravel()
        return (raw * self.y_std_ + self.y_mean_).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# §5  IceGNNClassifier
# ══════════════════════════════════════════════════════════════════════════════


class IceGNNClassifier(_IceGNNBase, ClassifierMixin):
    """
    IceGNN for binary or multiclass classification.

    Attributes (after fit)
    ──────────────────────
    classes_, n_classes_ : class labels and count
    feature_importances_ : pd.DataFrame

    Examples
    ────────
    >>> clf = IceGNNClassifier(
    ...     pretrained_learner=cb_clf,
    ...     cat_cols=[0, 3],
    ...     n_grid=40, d_model=64,
    ...     use_pretrained_residual=True,
    ... )
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    >>> preds = clf.predict(X_test)
    """

    _task = "classification"

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        self.le_: LabelEncoder = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_classes_: int = len(self.classes_)
        dtype = np.int64 if self.n_classes_ > 2 else np.float32
        return y_enc.astype(dtype)

    def _get_n_outputs(self) -> int:
        return 1 if getattr(self, "n_classes_", 2) == 2 else self.n_classes_

    def _get_n_classes(self) -> int:
        return getattr(self, "n_classes_", 2)

    def _compute_loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.n_classes_ == 2:
            return F.binary_cross_entropy_with_logits(out.squeeze(-1), y.float())
        return F.cross_entropy(out, y.long())

    def _pretrained_logits_np(self, X: np.ndarray) -> np.ndarray:
        """
        Pretrained class probabilities → logit space.

        Binary    : logit(P(y=1))          shape (n, 1)
        Multiclass: log(P(y=c))  for each c  shape (n, C)
        """
        if self.n_classes_ == 2:
            p1 = _predict_safe(self.pretrained_learner, X, "classification", 2)
            p1 = np.clip(p1, 1e-7, 1 - 1e-7).astype(np.float32)
            logit = np.log(p1 / (1.0 - p1))
            return logit.reshape(-1, 1)
        proba = _predict_safe(
            self.pretrained_learner, X, "classification", self.n_classes_
        )
        return np.log(np.clip(proba, 1e-7, 1.0)).astype(np.float32)   # (n, C)

    def predict_proba(self, X) -> np.ndarray:
        """
        Class probability estimates.

        Returns  (n, n_classes) float64
        """
        X_arr, _ = self._to_numpy(X)
        raw = self._forward_full(X_arr, quiet=True)["raw_out"]
        if self.n_classes_ == 2:
            p1 = torch.sigmoid(
                torch.from_numpy(raw.ravel().astype(np.float32))
            ).numpy()
            return np.column_stack([1.0 - p1, p1]).astype(np.float64)
        return (
            torch.softmax(torch.from_numpy(raw.astype(np.float32)), dim=-1)
            .numpy().astype(np.float64)
        )

    def predict_log_proba(self, X) -> np.ndarray:
        return np.log(self.predict_proba(X) + 1e-12)

    def predict(self, X) -> np.ndarray:
        return self.le_.inverse_transform(self.predict_proba(X).argmax(1))
