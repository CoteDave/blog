"""
Tab-JEPA 2.1
============
Adaptation des principes V-JEPA 2.1 aux données tabulaires.

Innovations V-JEPA 2.1 :
  1. Dense Predictive Loss  — loss sur TOUS les tokens (visibles + masqués)
  2. Deep Self-Supervision  — loss à plusieurs profondeurs de l'encodeur
  3. EMA teacher-student    — target encoder = EMA(context encoder)
  4. Masquage multi-blocs   — blocs contigus de features

API publique — sklearn-like, zéro boilerplate
─────────────────────────────────────────────

  # Non-supervisé
  trainer = TabJEPA21Trainer(cat_cols=["ville", "secteur"])
  trainer.fit(X)                          # X : DataFrame OU ndarray
  Z = trainer.extract_embeddings(X)       # (N, d_model)

  # Supervisé
  sup = TabJEPA21Supervised(cat_cols=["ville"], task="multiclass", n_outputs=5)
  sup.fit(X_train, y_train)
  sup.predict(X_test)                     # → labels / valeurs
  sup.predict_proba(X_test)              # → probabilités (classification)

cat_cols
────────
  Liste de noms de colonnes (str) si X est un DataFrame,
  OU d'indices (int) si X est un ndarray.
  Toutes les autres colonnes sont traitées comme numériques.
  Les valeurs catégorielles peuvent être brutes (str, int, object) —
  un LabelEncoder est appliqué automatiquement.

Scaling automatique
───────────────────
  X num  → StandardScaler  (fitté sur le split train)
  y      → StandardScaler  (régression uniquement, fitté sur le split train)

Dépendances : PyTorch, NumPy uniquement (pas de sklearn).
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

# pandas est optionnel : importé uniquement si disponible
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

# Type alias pour les entrées acceptées par l'API publique
ArrayLike = Union[np.ndarray, "pd.DataFrame", "pd.Series"]


# ══════════════════════════════════════════════════════════════════════════════
#  0.  Utilitaires de base
# ══════════════════════════════════════════════════════════════════════════════

class TabScaler:
    """
    StandardScaler pur NumPy.  z = (x − μ) / (σ + ε)

    clip > 0 : clipping dans [−clip, +clip] après normalisation (défaut 10).
               Mettre clip=0 pour désactiver (recommandé pour y).
    """

    def __init__(self, clip: float = 10.0, eps: float = 1e-8) -> None:
        self.clip  = clip
        self.eps   = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "TabScaler":
        kw = dict(axis=0, keepdims=(X.ndim > 1))
        self.mean_ = np.mean(X, **kw)
        self.std_  = np.std(X,  **kw)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("TabScaler non fitté — appelez .fit() d'abord")
        z = (X - self.mean_) / (self.std_ + self.eps)
        if self.clip > 0:
            z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("TabScaler non fitté")
        return (z * (self.std_ + self.eps) + self.mean_).astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None


class LabelEncoder:
    """
    Encodeur ordinal pur NumPy pour une colonne catégorielle brute.
    Valeurs inconnues à l'inférence → classe 0 (réservée à <unk>).
    Les classes connues démarrent à 1.
    """

    def __init__(self) -> None:
        self.classes_: Optional[np.ndarray] = None
        self._map: Optional[Dict] = None

    def fit(self, col: np.ndarray) -> "LabelEncoder":
        unique = sorted(set(col.tolist()) - {None, np.nan})
        self.classes_ = np.array(unique, dtype=object)
        # 0 réservé pour <unk>, classes connues de 1 à card
        self._map = {v: i + 1 for i, v in enumerate(unique)}
        return self

    def transform(self, col: np.ndarray) -> np.ndarray:
        if self._map is None:
            raise RuntimeError("LabelEncoder non fitté")
        return np.array(
            [self._map.get(v, 0) for v in col.tolist()], dtype=np.int64
        )

    def fit_transform(self, col: np.ndarray) -> np.ndarray:
        return self.fit(col).transform(col)

    @property
    def n_classes(self) -> int:
        """Nombre de classes connues (hors <unk>)."""
        return len(self.classes_) if self.classes_ is not None else 0


# ══════════════════════════════════════════════════════════════════════════════
#  1.  TabPreprocessor — pipeline X → (X_num, X_cat) automatique
# ══════════════════════════════════════════════════════════════════════════════

class TabPreprocessor:
    """
    Convertit un tableau X brut (DataFrame ou ndarray) en paire
    (X_num : float32, X_cat : int64) prêts pour le modèle.

    Pipeline automatique
    ────────────────────
    1. Accepte DataFrame (colonnes nommées) ou ndarray (colonnes indexées).
    2. Résout cat_cols (noms ou indices) → indices numériques stables.
    3. Colonnes numériques → StandardScaler (fitté sur le split train).
    4. Colonnes catégorielles → LabelEncoder par colonne (gère str/int/object).
    5. Déduit automatiquement : n_num, cat_cardinalities.

    Paramètres
    ──────────
    cat_cols : list[str] | list[int] | None
        Noms (DataFrame) ou indices (ndarray) des colonnes catégorielles.
        None ou [] → toutes les colonnes traitées comme numériques.
    scale    : bool, défaut True — applique StandardScaler sur X_num
    clip     : float, défaut 10.0 — clipping post-normalisation (0 = désactivé)
    """

    def __init__(
        self,
        cat_cols: Optional[List[Union[str, int]]] = None,
        scale:    bool  = True,
        clip:     float = 10.0,
    ) -> None:
        self.cat_cols_input = cat_cols or []
        self.scale = scale
        self.clip  = clip

        # Résolus dans fit()
        self.cat_idx_:    List[int]         = []   # indices colonnes cat dans X brut
        self.num_idx_:    List[int]         = []   # indices colonnes num dans X brut
        self.col_names_:  Optional[List[str]] = None
        self.label_encoders_: List[LabelEncoder] = []
        self.scaler_num_: Optional[TabScaler] = None
        self._fitted = False

    # ── Fit ───────────────────────────────────────────────────────────────
    def fit(self, X: ArrayLike) -> "TabPreprocessor":
        arr, col_names = self._to_numpy_and_names(X)
        self.col_names_ = col_names
        n_cols = arr.shape[1]

        # Résoudre les indices des colonnes catégorielles
        self.cat_idx_ = self._resolve_cat_idx(col_names, n_cols)
        self.num_idx_ = [i for i in range(n_cols) if i not in set(self.cat_idx_)]

        # Fitter les LabelEncoders sur chaque colonne cat
        self.label_encoders_ = []
        for ci in self.cat_idx_:
            le = LabelEncoder()
            le.fit(arr[:, ci])
            self.label_encoders_.append(le)

        # Fitter le scaler sur les colonnes numériques
        self.scaler_num_ = None
        if self.num_idx_ and self.scale:
            self.scaler_num_ = TabScaler(clip=self.clip)
            self.scaler_num_.fit(arr[:, self.num_idx_].astype(np.float32))

        self._fitted = True
        return self

    # ── Transform ─────────────────────────────────────────────────────────
    def transform(
        self, X: ArrayLike
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retourne (X_num, X_cat) :
          X_num : (N, n_num)  float32  ou None si aucune colonne numérique
          X_cat : (N, n_cat)  int64    ou None si aucune colonne catégorielle
        """
        if not self._fitted:
            raise RuntimeError("TabPreprocessor non fitté — appelez .fit() d'abord")
        arr, _ = self._to_numpy_and_names(X)

        X_num = None
        if self.num_idx_:
            X_num = arr[:, self.num_idx_].astype(np.float32)
            if self.scaler_num_ is not None:
                X_num = self.scaler_num_.transform(X_num)

        X_cat = None
        if self.cat_idx_:
            X_cat = np.column_stack([
                le.transform(arr[:, ci])
                for le, ci in zip(self.label_encoders_, self.cat_idx_)
            ]).astype(np.int64)

        return X_num, X_cat

    def fit_transform(
        self, X: ArrayLike
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.fit(X).transform(X)

    # ── Propriétés dérivées ────────────────────────────────────────────────
    @property
    def n_num(self) -> int:
        return len(self.num_idx_)

    @property
    def n_cat(self) -> int:
        return len(self.cat_idx_)

    @property
    def cat_cardinalities(self) -> List[int]:
        """Cardinalité de chaque feature catégorielle (hors <unk>)."""
        return [le.n_classes for le in self.label_encoders_]

    @property
    def feature_names_num(self) -> List[str]:
        if self.col_names_:
            return [self.col_names_[i] for i in self.num_idx_]
        return [f"num_{i}" for i in range(self.n_num)]

    @property
    def feature_names_cat(self) -> List[str]:
        if self.col_names_:
            return [self.col_names_[i] for i in self.cat_idx_]
        return [f"cat_{i}" for i in range(self.n_cat)]

    @property
    def feature_names(self) -> List[str]:
        return self.feature_names_num + self.feature_names_cat

    # ── Helpers internes ─────────────────────────────────────────────────
    @staticmethod
    def _to_numpy_and_names(
        X: ArrayLike,
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Convertit X en ndarray 2D + extrait les noms de colonnes si disponibles."""
        if _PANDAS and isinstance(X, (pd.DataFrame, pd.Series)):
            df = X if isinstance(X, pd.DataFrame) else X.to_frame()
            return df.to_numpy(dtype=object), list(df.columns.astype(str))
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr, None

    def _resolve_cat_idx(
        self, col_names: Optional[List[str]], n_cols: int
    ) -> List[int]:
        """Convertit cat_cols (noms ou indices) en liste d'indices entiers."""
        if not self.cat_cols_input:
            return []
        resolved: List[int] = []
        for c in self.cat_cols_input:
            if isinstance(c, str):
                if col_names is None:
                    raise ValueError(
                        f"cat_cols contient un nom de colonne '{c}' mais X est un ndarray "
                        "sans noms de colonnes. Utilisez des indices entiers."
                    )
                if c not in col_names:
                    raise ValueError(f"Colonne '{c}' introuvable dans X.")
                resolved.append(col_names.index(c))
            elif isinstance(c, int):
                if not (0 <= c < n_cols):
                    raise ValueError(f"Indice de colonne {c} hors bornes (n_cols={n_cols}).")
                resolved.append(c)
            else:
                raise TypeError(f"cat_cols doit contenir des str ou int, reçu {type(c)}")
        return sorted(set(resolved))

    def __repr__(self) -> str:
        if not self._fitted:
            return f"TabPreprocessor(cat_cols={self.cat_cols_input!r}, fitted=False)"
        return (
            f"TabPreprocessor("
            f"n_num={self.n_num}, n_cat={self.n_cat}, "
            f"cat_cardinalities={self.cat_cardinalities}, "
            f"scale={self.scale})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Feature Tokenizer (interne)
# ══════════════════════════════════════════════════════════════════════════════

class _FeatureTokenizer(nn.Module):
    """
    Projette chaque feature en un token de dimension d_model.
    Numérique : LayerNorm(xᵢ) → Linear(1, d).   Catégorielle : Embedding(card+1, d).
    Feature-identity embedding additionné sur tous les tokens.
    """

    def __init__(
        self,
        n_num:            int,
        cat_cardinalities: List[int],
        d_model:          int,
    ) -> None:
        super().__init__()
        self.n_num  = n_num
        self.n_cat  = len(cat_cardinalities)
        self.n_feat = n_num + self.n_cat

        if n_num > 0:
            self.num_norm = nn.ModuleList([nn.LayerNorm(1)       for _ in range(n_num)])
            self.num_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_num)])
        if self.n_cat > 0:
            self.cat_emb = nn.ModuleList(
                [nn.Embedding(c + 1, d_model) for c in cat_cardinalities]
            )
        self.feat_id = nn.Embedding(self.n_feat, d_model)

    def forward(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        toks: List[torch.Tensor] = []
        if self.n_num > 0 and x_num is not None:
            for i in range(self.n_num):
                toks.append(self.num_proj[i](self.num_norm[i](x_num[:, i:i+1])))
        if self.n_cat > 0 and x_cat is not None:
            for i in range(self.n_cat):
                toks.append(self.cat_emb[i](x_cat[:, i]))
        x = torch.stack(toks, dim=1)
        x = x + self.feat_id(torch.arange(self.n_feat, device=x.device)).unsqueeze(0)
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Transformer Encoder / Predictor (internes)
# ══════════════════════════════════════════════════════════════════════════════

class _TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        d_ff = int(d * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class _TabularEncoder(nn.Module):
    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        n_layers: int,
        dropout:  float,
        ds_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.ds_set = set(ds_layers or [n_layers])
        self.blocks = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        ints: List[torch.Tensor] = []
        for i, b in enumerate(self.blocks, 1):
            x = b(x)
            if i in self.ds_set:
                ints.append(self.norm(x))
        if not ints:
            ints.append(self.norm(x))
        return self.norm(x), ints


class _TabularPredictor(nn.Module):
    def __init__(
        self, d_model: int, d_pred: int, n_heads: int, n_layers: int, n_feat: int
    ) -> None:
        super().__init__()
        self.proj_in    = nn.Linear(d_model, d_pred)
        self.mask_token = nn.Parameter(torch.randn(d_pred) * 0.02)
        self.pos_emb    = nn.Embedding(n_feat, d_pred)
        self.blocks     = nn.ModuleList([_TransformerBlock(d_pred, n_heads) for _ in range(n_layers)])
        self.norm       = nn.LayerNorm(d_pred)
        self.proj_out   = nn.Linear(d_pred, d_model)

    def forward(self, ctx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(ctx)
        x[mask] = self.mask_token
        x = x + self.pos_emb(torch.arange(x.shape[1], device=x.device)).unsqueeze(0)
        for b in self.blocks:
            x = b(x)
        return self.proj_out(self.norm(x))


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Masquage par blocs
# ══════════════════════════════════════════════════════════════════════════════

def _block_mask(
    n_feat: int, B: int, ratio: float, n_blocks: int, device: torch.device
) -> torch.Tensor:
    n_target   = max(1, int(n_feat * ratio))
    block_size = max(1, n_target // n_blocks)
    mask = torch.zeros(B, n_feat, dtype=torch.bool, device=device)
    for b in range(B):
        covered: set = set()
        for _ in range(n_blocks):
            s = int(torch.randint(0, n_feat, ()).item())
            for j in range(block_size):
                covered.add((s + j) % n_feat)
        for idx in covered:
            mask[b, idx] = True
    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Dataset interne
# ══════════════════════════════════════════════════════════════════════════════

class _TabDataset(Dataset):
    def __init__(
        self,
        X_num: Optional[np.ndarray],
        X_cat: Optional[np.ndarray],
        y:     Optional[np.ndarray] = None,
    ) -> None:
        ref = X_num if X_num is not None else X_cat
        assert ref is not None
        self.N     = ref.shape[0]
        self.X_num = torch.from_numpy(X_num.astype(np.float32)) if X_num is not None else None
        self.X_cat = torch.from_numpy(X_cat.astype(np.int64))   if X_cat is not None else None
        self.y     = torch.from_numpy(y)                         if y     is not None else None

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if self.X_num is not None: out["x_num"] = self.X_num[idx]
        if self.X_cat is not None: out["x_cat"] = self.X_cat[idx]
        if self.y     is not None: out["y"]     = self.y[idx]
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  6.  _TabJEPA21Core — backbone PyTorch (interne)
# ══════════════════════════════════════════════════════════════════════════════

class _TabJEPA21Core(nn.Module):
    """Backbone SSL Tab-JEPA 2.1 — usage interne uniquement."""

    def __init__(
        self,
        n_num:             int,
        cat_cardinalities: List[int],
        d_model:           int,
        n_encoder_layers:  int,
        n_heads:           int,
        n_pred_layers:     int,
        d_pred:            int,
        mask_ratio:        float,
        n_mask_blocks:     int,
        ema_momentum:      float,
        ds_layers:         List[int],
        dropout:           float,
        vicreg_coef:       float,
    ) -> None:
        super().__init__()
        self.n_num          = n_num
        self.n_cat          = len(cat_cardinalities)
        self.n_feat         = n_num + len(cat_cardinalities)
        self.d_model        = d_model
        self.mask_ratio     = mask_ratio
        self.n_mask_blocks  = n_mask_blocks
        self.ema_momentum   = ema_momentum
        self.vicreg_coef    = vicreg_coef

        self.tokenizer  = _FeatureTokenizer(n_num, cat_cardinalities, d_model)
        self.encoder    = _TabularEncoder(d_model, n_heads, n_encoder_layers, dropout, ds_layers)
        self.target_enc = copy.deepcopy(self.encoder)
        for p in self.target_enc.parameters():
            p.requires_grad_(False)
        self.predictor  = _TabularPredictor(
            d_model, d_pred, max(1, d_pred // 32), n_pred_layers, self.n_feat
        )
        self.depth_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(len(ds_layers))]
        )

    @torch.no_grad()
    def update_ema(self) -> None:
        m = self.ema_momentum
        for p, q in zip(self.encoder.parameters(), self.target_enc.parameters()):
            q.data.mul_(m).add_(p.data, alpha=1.0 - m)

    def ssl_loss(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ref = x_num if x_num is not None else x_cat
        B, dev = ref.shape[0], ref.device
        mask   = _block_mask(self.n_feat, B, self.mask_ratio, self.n_mask_blocks, dev)

        tokens    = self.tokenizer(x_num, x_cat)
        ctx       = tokens.clone(); ctx[mask] = 0.0
        ctx_out, _ = self.encoder(ctx)

        with torch.no_grad():
            _, tgt_ints = self.target_enc(tokens)

        pred = self.predictor(ctx_out, mask)
        loss = torch.tensor(0.0, device=dev)
        metrics: Dict[str, float] = {}

        for i, (proj, tgt) in enumerate(zip(self.depth_proj, tgt_ints)):
            tgt    = tgt.detach()
            p      = proj(pred)
            ld     = F.smooth_l1_loss(F.normalize(p, dim=-1), F.normalize(tgt, dim=-1))
            vl     = F.relu(1.0 - tgt.std(dim=0).mean())
            loss   = loss + ld + self.vicreg_coef * vl
            metrics[f"loss_d{i}"] = ld.item()

        loss = loss / max(len(self.depth_proj), 1)
        metrics["ssl_loss"] = loss.item()
        return loss, metrics

    def encode(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
        pool:  str = "mean",
    ) -> torch.Tensor:
        out, _ = self.encoder(self.tokenizer(x_num, x_cat))
        if pool == "mean":  return out.mean(1)
        if pool == "max":   return out.max(1).values
        return out[:, 0]


# ══════════════════════════════════════════════════════════════════════════════
#  7.  Helpers partagés entre Trainer et Supervised
# ══════════════════════════════════════════════════════════════════════════════

def _make_loaders(
    X_num:   Optional[np.ndarray],
    X_cat:   Optional[np.ndarray],
    y:       Optional[np.ndarray],
    val_size: float,
    batch_size: int,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Crée train/val DataLoaders depuis des arrays déjà préprocessés."""
    ref    = X_num if X_num is not None else X_cat
    N      = ref.shape[0]
    n_val  = max(1, int(N * val_size))
    n_tr   = N - n_val

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(N)
    tri, vai = perm[:n_tr], perm[n_tr:]

    def _sub(idx: np.ndarray) -> DataLoader:
        xn = X_num[idx] if X_num is not None else None
        xc = X_cat[idx] if X_cat is not None else None
        yb = y[idx]     if y     is not None else None
        ds = _TabDataset(xn, xc, yb)
        is_train = len(idx) == n_tr
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=is_train, drop_last=is_train)

    return _sub(tri), _sub(vai)


def _train_one_epoch(
    model:     _TabJEPA21Core,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device:    str,
    grad_clip: float,
    warmup_lr: Optional[float],
    supervised_head: Optional[nn.Module] = None,
    task:            Optional[str]       = None,
) -> float:
    model.train()
    if supervised_head is not None:
        supervised_head.train()
    total, n = 0.0, 0

    for batch in loader:
        xn = batch["x_num"].to(device) if "x_num" in batch else None
        xc = batch["x_cat"].to(device) if "x_cat" in batch else None

        if supervised_head is None:
            loss, _ = model.ssl_loss(xn, xc)
        else:
            yb     = batch["y"].to(device)
            logits = supervised_head(model.encode(xn, xc)).squeeze(-1)
            loss   = _sup_loss(logits, yb, task)

        optimizer.zero_grad()
        loss.backward()
        # Clipper les gradients de tous les paramètres entraînables
        all_p = [p for p in model.parameters() if p.requires_grad]
        if supervised_head is not None:
            all_p += list(supervised_head.parameters())
        nn.utils.clip_grad_norm_(all_p, grad_clip)
        optimizer.step()

        if supervised_head is None:
            model.update_ema()

        total += loss.item(); n += 1

    if scheduler is not None and warmup_lr is None:
        scheduler.step()
    return total / max(n, 1)


@torch.no_grad()
def _eval_epoch(
    model:           _TabJEPA21Core,
    loader:          DataLoader,
    device:          str,
    supervised_head: Optional[nn.Module] = None,
    task:            Optional[str]       = None,
) -> float:
    model.eval()
    if supervised_head is not None:
        supervised_head.eval()
    total, n = 0.0, 0
    for batch in loader:
        xn = batch["x_num"].to(device) if "x_num" in batch else None
        xc = batch["x_cat"].to(device) if "x_cat" in batch else None
        if supervised_head is None:
            loss, _ = model.ssl_loss(xn, xc)
        else:
            yb     = batch["y"].to(device)
            logits = supervised_head(model.encode(xn, xc)).squeeze(-1)
            loss   = _sup_loss(logits, yb, task)
        total += loss.item(); n += 1
    return total / max(n, 1)


def _sup_loss(logits: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    if task == "regression": return F.mse_loss(logits, y)
    if task == "binary":     return F.binary_cross_entropy_with_logits(logits, y)
    return F.cross_entropy(logits, y)


def _run_training(
    model:           _TabJEPA21Core,
    train_dl:        DataLoader,
    val_dl:          DataLoader,
    n_epochs:        int,
    warmup_epochs:   int,
    base_lr:         float,
    weight_decay:    float,
    grad_clip:       float,
    ema_start:       float,
    ema_end:         float,
    patience:        int,
    device:          str,
    verbose:         bool,
    supervised_head: Optional[nn.Module] = None,
    task:            Optional[str]       = None,
    lr_head:         Optional[float]     = None,
    lr_encoder:      Optional[float]     = None,
) -> Dict[str, List[float]]:
    """Boucle d'entraînement partagée SSL + supervisé."""

    # ── Groupes de paramètres ────────────────────────────────────────────
    if supervised_head is not None:
        # Backbone = tokenizer + encoder (predictor/target_enc = composants SSL
        # non utilisés en mode supervisé → on les exclut de l'optimizer)
        backbone_params = (
            list(model.tokenizer.parameters()) +
            [p for p in model.encoder.parameters() if p.requires_grad]
        )
        # lr backbone : si c'est du fine-tuning (lr_encoder petit fourni explicitement)
        #               → lr_encoder ; sinon from-scratch → même lr que la tête
        _lr_backbone = lr_encoder if (lr_encoder is not None and lr_encoder < (lr_head or base_lr))  \
                       else (lr_head or base_lr)
        param_groups = [
            {"params": list(supervised_head.parameters()),
             "lr": lr_head or base_lr,
             "_name": "head"},
            {"params": backbone_params,
             "lr": _lr_backbone,
             "_name": "backbone"},
        ]
    else:
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad],
             "lr": base_lr,
             "_name": "ssl"}
        ]

    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, n_epochs - warmup_epochs),
        eta_min=(lr_head or base_lr) * 0.01,
    )

    history    = {"train_loss": [], "val_loss": []}
    best_val   = float("inf")
    # Sauvegarder l'intégralité du backbone (tokenizer + encoder) + tête
    best_state = {
        "model": copy.deepcopy(model.state_dict()),
        "head":  copy.deepcopy(supervised_head.state_dict()) if supervised_head else None,
    }
    no_improve = 0

    for epoch in range(n_epochs):
        # Warmup LR
        warmup_lr: Optional[float] = None
        if epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / max(warmup_epochs, 1)
            for pg in optimizer.param_groups:
                base = pg.get("_base_lr", pg["lr"])
                pg["_base_lr"] = base
                pg["lr"]       = base * (epoch + 1) / max(warmup_epochs, 1)

        # EMA schedule (SSL uniquement)
        if supervised_head is None:
            t = epoch / max(n_epochs, 1)
            model.ema_momentum = (
                ema_end - (ema_end - ema_start) * (math.cos(math.pi * t) + 1) / 2
            )

        tr_loss = _train_one_epoch(
            model, train_dl, optimizer, scheduler, device, grad_clip, warmup_lr,
            supervised_head, task,
        )
        val_loss = _eval_epoch(model, val_dl, device, supervised_head, task)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        es_str = ""
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "head":  copy.deepcopy(supervised_head.state_dict()) if supervised_head else None,
            }
            no_improve = 0
            es_str = " ✓"
        else:
            no_improve += 1
            es_str = f" [{no_improve}/{patience}]"

        if verbose:
            lr_shown = optimizer.param_groups[0]["lr"]
            ema_str  = f" | ema={model.ema_momentum:.4f}" if supervised_head is None else ""
            print(
                f"Epoch {epoch+1:03d}/{n_epochs}"
                f" | train={tr_loss:.4f} | val={val_loss:.4f}"
                f" | lr={lr_shown:.2e}{ema_str}{es_str}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"\n⚑ Early stopping époque {epoch+1} (best val={best_val:.4f})")
            break

    # Restauration du meilleur état (tokenizer + encoder + tête)
    model.load_state_dict(best_state["model"])
    if supervised_head is not None and best_state["head"] is not None:
        supervised_head.load_state_dict(best_state["head"])

    if verbose:
        print(f"✓ Meilleur val={best_val:.4f}")

    return history


# ══════════════════════════════════════════════════════════════════════════════
#  8.  TabJEPA21Trainer — pré-entraînement non-supervisé
# ══════════════════════════════════════════════════════════════════════════════

class TabJEPA21Trainer:
    """
    Pré-entraîne Tab-JEPA 2.1 de façon non-supervisée.

    Paramètres
    ──────────
    cat_cols        : noms (str, si X est DataFrame) ou indices (int) des colonnes
                      catégorielles. Toutes les autres → numériques.
    d_model         : dimension des tokens (défaut 128)
    n_encoder_layers: profondeur de l'encodeur (défaut 6)
    n_heads         : têtes d'attention (défaut 4)
    mask_ratio      : fraction de features masquées (défaut 0.40)
    val_size        : fraction de validation pour l'early stopping (défaut 0.15)
    patience        : époques sans amélioration avant arrêt (défaut 15)
    n_epochs        : nombre max d'époques (défaut 100)
    batch_size      : taille des mini-batchs (défaut 256)
    lr              : learning rate de base (défaut 3e-4)
    scale_X         : StandardScaler sur les features numériques (défaut True)
    device          : "cpu" | "cuda" | "mps"

    Usage
    ─────
    trainer = TabJEPA21Trainer(cat_cols=["ville", "type"])
    trainer.fit(X_train)                        # DataFrame ou ndarray
    Z = trainer.extract_embeddings(X_test)      # (N, d_model)
    Z_dict = trainer.extract_embeddings(X_test, per_token=True, return_dict=True)
    """

    def __init__(
        self,
        cat_cols:          Optional[List[Union[str, int]]] = None,
        d_model:           int   = 128,
        n_encoder_layers:  int   = 6,
        n_heads:           int   = 4,
        n_predictor_layers: int  = 3,
        d_pred:            int   = 64,
        mask_ratio:        float = 0.40,
        n_mask_blocks:     int   = 2,
        ema_start:         float = 0.996,
        ema_end:           float = 0.9999,
        ds_layers:         Optional[List[int]] = None,
        dropout:           float = 0.1,
        vicreg_coef:       float = 0.10,
        lr:                float = 3e-4,
        weight_decay:      float = 1e-2,
        n_epochs:          int   = 100,
        warmup_epochs:     int   = 10,
        grad_clip:         float = 1.0,
        val_size:          float = 0.15,
        patience:          int   = 15,
        batch_size:        int   = 256,
        scale_X:           bool  = True,
        device:            str   = "cpu",
    ) -> None:
        self._ctor_kwargs = dict(
            d_model=d_model, n_encoder_layers=n_encoder_layers, n_heads=n_heads,
            n_pred_layers=n_predictor_layers, d_pred=d_pred,
            mask_ratio=mask_ratio, n_mask_blocks=n_mask_blocks,
            ema_momentum=ema_start, ds_layers=ds_layers,
            dropout=dropout, vicreg_coef=vicreg_coef,
        )
        self._train_kwargs = dict(
            n_epochs=n_epochs, warmup_epochs=warmup_epochs, base_lr=lr,
            weight_decay=weight_decay, grad_clip=grad_clip,
            ema_start=ema_start, ema_end=ema_end, patience=patience, device=device,
        )
        self.cat_cols   = cat_cols or []
        self.val_size   = val_size
        self.batch_size = batch_size
        self.device     = device

        self.preprocessor: Optional[TabPreprocessor] = None
        self.model_:        Optional[_TabJEPA21Core]  = None
        self._fitted = False

    # ── fit() ─────────────────────────────────────────────────────────────
    def fit(self, X: ArrayLike, verbose: bool = True) -> "TabJEPA21Trainer":
        """
        Pré-entraîne le modèle sur X (DataFrame ou ndarray).
        Scaling + split train/val automatiques.
        """
        # ── Préprocessing (split puis fit sur train) ──────────────────────
        arr_raw, col_names = TabPreprocessor._to_numpy_and_names(X)
        N      = arr_raw.shape[0]
        n_val  = max(1, int(N * self.val_size))
        perm   = np.random.default_rng(42).permutation(N)
        tr_idx = perm[: N - n_val]
        va_idx = perm[N - n_val :]

        scale = self._ctor_kwargs.get("d_model", 128) > 0  # toujours True
        self.preprocessor = TabPreprocessor(
            cat_cols=self.cat_cols, scale=True,
        )
        # Fitter uniquement sur le train
        X_tr_raw = arr_raw[tr_idx]
        if col_names:
            import pandas as _pd
            X_tr_raw = _pd.DataFrame(X_tr_raw, columns=col_names)
        self.preprocessor.fit(X_tr_raw)

        # Transformer train + val
        def _transform(idx):
            sub = arr_raw[idx]
            if col_names:
                import pandas as _pd
                sub = _pd.DataFrame(sub, columns=col_names)
            return self.preprocessor.transform(sub)

        Xn_tr, Xc_tr = _transform(tr_idx)
        Xn_va, Xc_va = _transform(va_idx)

        # ── Construire le backbone ────────────────────────────────────────
        ds_layers = self._ctor_kwargs.pop("ds_layers", None)
        nl = self._ctor_kwargs["n_encoder_layers"]
        ds_layers = ds_layers or [nl // 2, nl]

        self.model_ = _TabJEPA21Core(
            n_num=self.preprocessor.n_num,
            cat_cardinalities=self.preprocessor.cat_cardinalities,
            ds_layers=ds_layers,
            **self._ctor_kwargs,
        ).to(self.device)
        self._ctor_kwargs["ds_layers"] = ds_layers  # remettre pour __repr__

        n_p = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        if verbose:
            print(
                f"TabJEPA21Trainer — backbone créé : "
                f"{self.preprocessor.n_num} num + {self.preprocessor.n_cat} cat → "
                f"n_features={self.model_.n_feat}  params={n_p:,}\n"
                f"[val_size={self.val_size}] split → train={len(tr_idx)} | val={len(va_idx)}"
            )

        # ── DataLoaders ───────────────────────────────────────────────────
        train_dl = DataLoader(_TabDataset(Xn_tr, Xc_tr), batch_size=self.batch_size,
                              shuffle=True, drop_last=True)
        val_dl   = DataLoader(_TabDataset(Xn_va, Xc_va), batch_size=self.batch_size,
                              shuffle=False, drop_last=False)

        # ── Entraînement ──────────────────────────────────────────────────
        _run_training(
            self.model_, train_dl, val_dl,
            verbose=verbose, **self._train_kwargs,
        )
        self._fitted = True
        return self

    # ── extract_embeddings() ──────────────────────────────────────────────
    @torch.no_grad()
    def extract_embeddings(
        self,
        X:          ArrayLike,
        batch_size: int  = 512,
        pool:       str  = "mean",
        per_token:  bool = False,
        return_dict: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extrait les embeddings de l'encodeur pré-entraîné.

        Paramètres
        ──────────
        pool        : "mean" | "max" | "first"  (ignoré si per_token=True)
        per_token   : retourne (N, n_features, d_model) sans pooling
        return_dict : (per_token=True requis) → dict {feature_name: (N, d_model)}

        Retourne
        ────────
        np.ndarray (N, d_model)                       pool & !per_token
        np.ndarray (N, n_features, d_model)           per_token & !return_dict
        dict {str → np.ndarray (N, d_model)}          per_token & return_dict
        """
        self._check_fitted()
        X_num, X_cat = self.preprocessor.transform(X)

        was_training = self.model_.training
        self.model_.eval()
        dev = torch.device(self.device)
        ref = X_num if X_num is not None else X_cat
        N   = ref.shape[0]
        out: List[np.ndarray] = []

        for s in range(0, N, batch_size):
            e  = min(s + batch_size, N)
            xn = torch.from_numpy(X_num[s:e]).to(dev) if X_num is not None else None
            xc = torch.from_numpy(X_cat[s:e]).to(dev) if X_cat is not None else None
            tokens = self.model_.tokenizer(xn, xc)
            emb, _ = self.model_.encoder(tokens)
            if per_token:
                out.append(emb.cpu().numpy())
            else:
                if pool == "mean":    out.append(emb.mean(1).cpu().numpy())
                elif pool == "max":   out.append(emb.max(1).values.cpu().numpy())
                else:                 out.append(emb[:, 0].cpu().numpy())

        if was_training:
            self.model_.train()

        result = np.concatenate(out, axis=0)
        if per_token and return_dict:
            names = self.preprocessor.feature_names
            return {name: result[:, i, :] for i, name in enumerate(names)}
        return result

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Appelez .fit(X) d'abord.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        cats   = self.cat_cols or []
        return f"TabJEPA21Trainer(cat_cols={cats!r}, status={status!r})"


# ══════════════════════════════════════════════════════════════════════════════
#  9.  TabJEPA21Supervised — estimateur supervisé (API sklearn-like)
# ══════════════════════════════════════════════════════════════════════════════

class TabJEPA21Supervised:
    """
    Estimateur supervisé Tab-JEPA 2.1. API sklearn-like : fit / predict / predict_proba.

    Tâches supportées
    ─────────────────
    "regression"   MSE sur y scalé → predict() inverse-transforme automatiquement
    "binary"       BCE + sigmoid    → predict() → {0,1}   predict_proba() → (N,2)
    "multiclass"   CrossEntropy     → predict() → {0…C-1} predict_proba() → (N,C)

    Paramètres principaux
    ─────────────────────
    cat_cols    : colonnes catégorielles (noms ou indices).
    task        : "regression" | "binary" | "multiclass"
    n_outputs   : nombre de classes (multiclass) ou 1 pour regression/binary.
    pretrained  : TabJEPA21Trainer déjà fitté pour initialiser le backbone
                  (transfer learning). Si None → backbone entraîné from scratch.
    freeze_encoder : True = linear probe, False = fine-tuning complet.

    Usage
    ─────
    sup = TabJEPA21Supervised(cat_cols=["ville"], task="binary")
    sup.fit(X_train, y_train)
    sup.predict(X_test)
    sup.predict_proba(X_test)
    """

    def __init__(
        self,
        cat_cols:           Optional[List[Union[str, int]]] = None,
        task:               str   = "multiclass",
        n_outputs:          int   = 2,
        pretrained:         Optional[TabJEPA21Trainer] = None,
        freeze_encoder:     bool  = False,
        # Architecture
        d_model:            int   = 128,
        n_encoder_layers:   int   = 6,
        n_heads:            int   = 4,
        n_pred_layers:      int   = 3,
        d_pred:             int   = 64,
        ds_layers:          Optional[List[int]] = None,
        dropout:            float = 0.1,
        head_hidden_ratio:  float = 0.5,
        head_dropout:       float = 0.1,
        # Entraînement
        lr_head:            float = 1e-3,
        lr_encoder:         float = 3e-5,
        weight_decay:       float = 1e-2,
        n_epochs:           int   = 100,
        warmup_epochs:      int   = 5,
        grad_clip:          float = 1.0,
        val_size:           float = 0.15,
        patience:           int   = 15,
        batch_size:         int   = 256,
        scale_X:            bool  = True,
        device:             str   = "cpu",
    ) -> None:
        assert task in ("regression", "binary", "multiclass"), \
            "task ∈ {'regression', 'binary', 'multiclass'}"

        self.cat_cols       = cat_cols or []
        self.task           = task
        self.n_outputs      = 1 if task == "binary" else n_outputs
        self.pretrained     = pretrained
        self.freeze_encoder = freeze_encoder
        self.device         = device
        self.val_size       = val_size
        self.batch_size     = batch_size
        self.scale_X        = scale_X
        self.n_epochs       = n_epochs
        self.warmup_epochs  = warmup_epochs
        self.grad_clip      = grad_clip
        self.patience       = patience
        self.lr_head        = lr_head
        self.lr_encoder     = lr_encoder
        self.weight_decay   = weight_decay

        self._arch = dict(
            d_model=d_model, n_encoder_layers=n_encoder_layers,
            n_heads=n_heads, n_pred_layers=n_pred_layers, d_pred=d_pred,
            ds_layers=ds_layers, dropout=dropout,
        )
        self._head_arch = dict(
            head_hidden_ratio=head_hidden_ratio,
            head_dropout=head_dropout,
        )

        self.preprocessor: Optional[TabPreprocessor] = None
        self.model_:        Optional[_TabJEPA21Core]  = None
        self._head:         Optional[nn.Module]       = None
        self.scaler_y:      Optional[TabScaler]       = None
        self._is_fitted = False

    # ── fit() ─────────────────────────────────────────────────────────────
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        verbose: bool = True,
    ) -> "TabJEPA21Supervised":
        """
        Entraîne l'estimateur.

        X peut être un DataFrame (colonnes nommées) ou un ndarray.
        y peut être un array 1D, une Series pandas, ou une liste.

        Scaling automatique :
          - X num → StandardScaler (fitté sur le split train)
          - y     → StandardScaler (régression uniquement, fitté sur le split train)
        """
        # ── Normaliser y → np.ndarray 1D ─────────────────────────────────
        if _PANDAS and isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = y.to_numpy().ravel()
        else:
            y_arr = np.asarray(y).ravel()

        # ── Split indices ──────────────────────────────────────────────────
        N      = y_arr.shape[0]
        n_val  = max(1, int(N * self.val_size))
        perm   = np.random.default_rng(42).permutation(N)
        tr_idx = perm[: N - n_val]
        va_idx = perm[N - n_val :]

        # ── Préprocessing X ────────────────────────────────────────────────
        arr_raw, col_names = TabPreprocessor._to_numpy_and_names(X)

        # Construire un sous-array train pour fitter le preprocessor
        def _sub_raw(idx):
            sub = arr_raw[idx]
            if col_names:
                import pandas as _pd
                return _pd.DataFrame(sub, columns=col_names)
            return sub

        self.preprocessor = TabPreprocessor(cat_cols=self.cat_cols, scale=self.scale_X)
        self.preprocessor.fit(_sub_raw(tr_idx))

        Xn_tr, Xc_tr = self.preprocessor.transform(_sub_raw(tr_idx))
        Xn_va, Xc_va = self.preprocessor.transform(_sub_raw(va_idx))

        if verbose:
            print(
                f"[val_size={self.val_size}] split → train={len(tr_idx)} | val={len(va_idx)}\n"
                f"  features : {self.preprocessor.n_num} numériques + "
                f"{self.preprocessor.n_cat} catégorielles "
                f"(cardinalities={self.preprocessor.cat_cardinalities})"
            )

        # ── Scaler y (régression) ─────────────────────────────────────────
        y_tr = y_arr[tr_idx].astype(np.float32)
        y_va = y_arr[va_idx].astype(np.float32)

        if self.task == "regression":
            self.scaler_y = TabScaler(clip=0)
            y_tr = self.scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
            y_va = self.scaler_y.transform(y_va.reshape(-1, 1)).ravel()
            if verbose:
                print(f"  scaler_y → μ={float(self.scaler_y.mean_):.3f}  σ={float(self.scaler_y.std_):.3f}")
        elif self.task == "binary":
            y_tr = y_tr.astype(np.float32)
            y_va = y_va.astype(np.float32)
        else:
            y_tr = y_tr.astype(np.int64)
            y_va = y_va.astype(np.int64)

        # ── Backbone ──────────────────────────────────────────────────────
        if self.pretrained is not None and self.pretrained._fitted:
            # Transfer learning depuis un trainer pré-entraîné
            self.model_ = copy.deepcopy(self.pretrained.model_).to(self.device)
            if verbose:
                print("  backbone initialisé depuis le trainer pré-entraîné (transfer learning)")
        else:
            nl  = self._arch["n_encoder_layers"]
            dsl = self._arch.pop("ds_layers", None) or [nl // 2, nl]
            self.model_ = _TabJEPA21Core(
                n_num=self.preprocessor.n_num,
                cat_cardinalities=self.preprocessor.cat_cardinalities,
                ema_momentum=0.996,
                vicreg_coef=0.0,
                mask_ratio=0.40,
                n_mask_blocks=2,
                ds_layers=dsl,
                **self._arch,
            ).to(self.device)
            self._arch["ds_layers"] = dsl

        if self.freeze_encoder:
            for p in self.model_.encoder.parameters():
                p.requires_grad_(False)

        # ── Tête supervisée ───────────────────────────────────────────────
        d   = self.model_.d_model
        d_h = max(16, int(d * self._head_arch["head_hidden_ratio"]))
        self._head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d_h),
            nn.GELU(),
            nn.Dropout(self._head_arch["head_dropout"]),
            nn.Linear(d_h, self.n_outputs),
        ).to(self.device)

        if verbose:
            n_p = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
            n_h = sum(p.numel() for p in self._head.parameters())
            print(f"  params backbone={n_p:,}  tête={n_h:,}")

        # ── DataLoaders ───────────────────────────────────────────────────
        train_dl = DataLoader(_TabDataset(Xn_tr, Xc_tr, y_tr), batch_size=self.batch_size,
                              shuffle=True, drop_last=True)
        val_dl   = DataLoader(_TabDataset(Xn_va, Xc_va, y_va), batch_size=self.batch_size,
                              shuffle=False, drop_last=False)

        # ── Entraînement ──────────────────────────────────────────────────
        # Si pretrained : lr_encoder petit (fine-tuning prudent)
        # Si from scratch : lr_encoder = lr_head (entraîner tout à même vitesse)
        _lr_enc = self.lr_encoder if (self.pretrained is not None and self.pretrained._fitted) \
                  else self.lr_head
        _run_training(
            self.model_, train_dl, val_dl,
            n_epochs=self.n_epochs, warmup_epochs=self.warmup_epochs,
            base_lr=self.lr_head, weight_decay=self.weight_decay,
            grad_clip=self.grad_clip, ema_start=0.996, ema_end=0.9999,
            patience=self.patience, device=self.device, verbose=verbose,
            supervised_head=self._head, task=self.task,
            lr_head=self.lr_head, lr_encoder=_lr_enc,
        )
        self._is_fitted = True
        return self

    # ── predict() ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, X: ArrayLike, batch_size: int = 512) -> np.ndarray:
        """
        regression  → (N,) float32 dans l'espace original (inverse-scalé)
        binary      → (N,) int64  ∈ {0, 1}
        multiclass  → (N,) int64  ∈ {0, …, C−1}
        """
        self._check_fitted()
        logits = self._forward(X, batch_size)
        if self.task == "regression":
            v = logits.ravel()
            return self.scaler_y.inverse_transform(v.reshape(-1, 1)).ravel() \
                   if self.scaler_y else v
        if self.task == "binary":
            return (torch.sigmoid(torch.from_numpy(logits)).numpy().ravel() >= 0.5
                    ).astype(np.int64)
        return np.argmax(logits, axis=1).astype(np.int64)

    # ── predict_proba() ───────────────────────────────────────────────────
    @torch.no_grad()
    def predict_proba(self, X: ArrayLike, batch_size: int = 512) -> np.ndarray:
        """
        binary      → (N, 2)  [[P(0), P(1)], …]
        multiclass  → (N, C)  softmax
        regression  → lève ValueError
        """
        if self.task == "regression":
            raise ValueError("predict_proba() n'est pas disponible pour la régression.")
        self._check_fitted()
        logits = self._forward(X, batch_size)
        if self.task == "binary":
            p1 = torch.sigmoid(torch.from_numpy(logits)).numpy().ravel()
            return np.column_stack([1 - p1, p1]).astype(np.float32)
        return torch.softmax(torch.from_numpy(logits), dim=1).numpy().astype(np.float32)

    # ── Helpers internes ─────────────────────────────────────────────────
    @torch.no_grad()
    def _forward(self, X: ArrayLike, batch_size: int) -> np.ndarray:
        self.model_.eval(); self._head.eval()
        X_num, X_cat = self.preprocessor.transform(X)
        ref = X_num if X_num is not None else X_cat
        N   = ref.shape[0]
        dev = torch.device(self.device)
        out: List[np.ndarray] = []
        for s in range(0, N, batch_size):
            e  = min(s + batch_size, N)
            xn = torch.from_numpy(X_num[s:e]).to(dev) if X_num is not None else None
            xc = torch.from_numpy(X_cat[s:e]).to(dev) if X_cat is not None else None
            out.append(self._head(self.model_.encode(xn, xc)).cpu().numpy())
        return np.concatenate(out, axis=0)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Appelez .fit(X, y) d'abord.")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"TabJEPA21Supervised("
            f"task={self.task!r}, n_outputs={self.n_outputs}, "
            f"cat_cols={self.cat_cols!r}, "
            f"freeze_encoder={self.freeze_encoder}, "
            f"status={status!r})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  10.  Smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Tab-JEPA 2.1 — smoke test")
    print("=" * 65)

    torch.manual_seed(42)
    np.random.seed(42)

    N = 3_000

    # ── Données synthétiques avec pandas ─────────────────────────────────
    try:
        import pandas as pd

        villes   = ["Paris", "Lyon", "Marseille", "Bordeaux", "Lille"]
        secteurs = ["Tech", "Finance", "Santé", "Industrie"]

        df = pd.DataFrame({
            "age":       np.random.randint(22, 65, N).astype(float),
            "salaire":   np.random.randn(N) * 15_000 + 45_000,
            "ancienneté": np.random.randint(0, 30, N).astype(float),
            "score":     np.random.randn(N) * 10 + 50,
            "ville":     np.random.choice(villes,   N),   # str brut
            "secteur":   np.random.choice(secteurs, N),   # str brut
            "niveau":    np.random.randint(1, 6, N),      # int catégoriel
        })
        y_reg = (df["salaire"] * 0.3 + df["age"] * 200 + np.random.randn(N) * 2000
                 ).to_numpy(np.float32)
        y_bin = (y_reg > y_reg.mean()).astype(np.float32)
        y_clf = np.random.randint(0, 4, N).astype(np.int64)

        cat_cols = ["ville", "secteur", "niveau"]   # noms de colonnes
        use_pandas = True
        print("Pandas disponible — test avec DataFrame")

    except ImportError:
        # Fallback : ndarray pur
        df = np.column_stack([
            np.random.randn(N, 4) * 50 + 100,     # 4 colonnes num
            np.random.randint(0, 5, N),             # cat int
            np.random.randint(0, 4, N),             # cat int
        ])
        y_reg = (df[:, 0] * 2 + np.random.randn(N) * 10).astype(np.float32)
        y_bin = (y_reg > y_reg.mean()).astype(np.float32)
        y_clf = np.random.randint(0, 4, N).astype(np.int64)
        cat_cols = [4, 5]                           # indices de colonnes
        use_pandas = False
        print("Pandas absent — test avec ndarray")

    # ═══ PHASE 1 : Pré-entraînement SSL ══════════════════════════════════
    print("\n── Phase 1 : Pré-entraînement SSL ──")
    trainer = TabJEPA21Trainer(
        cat_cols=cat_cols,
        d_model=64, n_encoder_layers=4, n_heads=4,
        n_predictor_layers=2, d_pred=32,
        n_epochs=4, warmup_epochs=1,
        val_size=0.15, patience=3, batch_size=256,
    )
    trainer.fit(df, verbose=True)

    # ── extract_embeddings ────────────────────────────────────────────────
    print("\n── extract_embeddings ──")
    Z = trainer.extract_embeddings(df, pool="mean")
    print(f"  pooled    : {Z.shape}  dtype={Z.dtype}")

    Z_pt = trainer.extract_embeddings(df, per_token=True)
    print(f"  per_token : {Z_pt.shape}")

    Z_d  = trainer.extract_embeddings(df, per_token=True, return_dict=True)
    first_key = list(Z_d.keys())[0]
    print(f"  dict  : {list(Z_d.keys())}  shape['{first_key}']={Z_d[first_key].shape}")

    # ═══ PHASE 2 : Supervisé — régression ════════════════════════════════
    print("\n── Supervisé — régression ──")
    sup_reg = TabJEPA21Supervised(
        cat_cols=cat_cols, task="regression", n_outputs=1,
        pretrained=trainer,            # ← transfer learning
        n_epochs=5, warmup_epochs=1, val_size=0.15, patience=3, batch_size=256,
    )
    sup_reg.fit(df, y_reg, verbose=True)
    p_reg = sup_reg.predict(df.iloc[:16] if use_pandas else df[:16])
    print(f"  predict()  → {p_reg.shape}  sample={p_reg[:3].round(0)}")
    try:
        sup_reg.predict_proba(df.iloc[:4] if use_pandas else df[:4])
    except ValueError as e:
        print(f"  predict_proba bloqué ✓")

    # ═══ PHASE 3 : Supervisé — binaire ═══════════════════════════════════
    print("\n── Supervisé — binaire ──")
    sup_bin = TabJEPA21Supervised(
        cat_cols=cat_cols, task="binary", n_outputs=1,
        n_epochs=5, warmup_epochs=1, val_size=0.15, patience=3, batch_size=256,
    )
    sup_bin.fit(df, y_bin, verbose=True)
    p_bin = sup_bin.predict(df.iloc[:16] if use_pandas else df[:16])
    q_bin = sup_bin.predict_proba(df.iloc[:16] if use_pandas else df[:16])
    print(f"  predict()       → {p_bin.shape}  unique={np.unique(p_bin)}")
    print(f"  predict_proba() → {q_bin.shape}  sum={q_bin[:2].sum(1).round(4)}")

    # ═══ PHASE 4 : Supervisé — multiclasses ══════════════════════════════
    print("\n── Supervisé — multiclasses ──")
    sup_clf = TabJEPA21Supervised(
        cat_cols=cat_cols, task="multiclass", n_outputs=4,
        n_epochs=5, warmup_epochs=1, val_size=0.15, patience=3, batch_size=256,
    )
    sup_clf.fit(df, y_clf, verbose=True)
    p_clf = sup_clf.predict(df.iloc[:16] if use_pandas else df[:16])
    q_clf = sup_clf.predict_proba(df.iloc[:16] if use_pandas else df[:16])
    print(f"  predict()       → {p_clf.shape}  classes={np.unique(p_clf)}")
    print(f"  predict_proba() → {q_clf.shape}  sum={q_clf[:2].sum(1).round(4)}")
    print(f"\n  repr : {sup_clf}")

    print("\n✓ Smoke test réussi.")
