"""
Tab-JEPA 2.1
============
Adaptation des principes V-JEPA 2.1 aux données tabulaires.

Innovations V-JEPA 2.1 portées :
  1. Dense Predictive Loss  — loss sur TOUS les tokens (visibles + masqués)
  2. Deep Self-Supervision  — loss à plusieurs profondeurs de l'encodeur
  3. EMA teacher-student    — target encoder = EMA(context encoder)
  4. Masquage multi-blocs   — blocs contigus de features

Spécificités tabulaires :
  - Tokenisation : numérique → Linear(1,d) ; catégorielle → Embedding
  - Feature-identity embedding learnable (analogue aux positional embeddings ViT)
  - VICReg variance term pour prévenir l'effondrement des représentations
  - StandardScaler intégré dans le pipeline (X_num et y pour la régression)
  - Aucune dépendance externe autre que PyTorch + NumPy

Modes
-----
Non-supervisé : TabJEPA21 + TabJEPA21Trainer
  trainer.fit_numpy(X_num, X_cat)          # scaling + split + early stopping auto
  backbone.extract_embeddings(...)          # (N,d) | (N,F,d) | dict

Supervisé     : TabJEPA21Supervised
  Tâches : "regression" | "binary" | "multiclass"
  sup.fit(X_num, X_cat, y)
  sup.predict(X_num, X_cat)               # labels / valeurs inversement scalées
  sup.predict_proba(X_num, X_cat)         # proba (classification uniquement)
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


# ══════════════════════════════════════════════════════════════════════════════
#  0.  TabScaler — StandardScaler NumPy-natif (sans sklearn)
# ══════════════════════════════════════════════════════════════════════════════

class TabScaler:
    """
    StandardScaler pur NumPy.  z = (x − μ) / (σ + ε)

    Paramètres
    ----------
    clip : clipping post-normalisation dans [−clip, +clip].  0 = désactivé.
           (utile pour les outliers extrêmes — défaut 10.0)
    eps  : plancher de std pour éviter la division par zéro
    """

    def __init__(self, clip: float = 10.0, eps: float = 1e-8) -> None:
        self.clip  = clip
        self.eps   = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    # ── Fit / transform ───────────────────────────────────────────────────
    def fit(self, X: np.ndarray) -> "TabScaler":
        """X : (N,) ou (N, p)."""
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


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Feature Tokenizer
# ══════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    """
    Projette chaque feature en un token de dimension d_model.

    Numeric  : LayerNorm(xᵢ) → Linear(1, d_model)   (projection indépendante par feature)
    Categoric: Embedding(card_i + 1, d_model)         (+1 pour token <unk>/masque)

    Un feature-identity embedding learnable est additionné à chaque token —
    analogue aux positional embeddings de ViT.
    """

    def __init__(
        self,
        n_num:            int,
        cat_cardinalities: List[int],
        d_model:          int = 128,
    ) -> None:
        super().__init__()
        self.n_num      = n_num
        self.n_cat      = len(cat_cardinalities)
        self.n_features = n_num + self.n_cat
        self.d_model    = d_model

        if n_num > 0:
            self.num_norm = nn.ModuleList([nn.LayerNorm(1)        for _ in range(n_num)])
            self.num_proj = nn.ModuleList([nn.Linear(1, d_model)  for _ in range(n_num)])

        if self.n_cat > 0:
            self.cat_emb = nn.ModuleList(
                [nn.Embedding(c + 1, d_model) for c in cat_cardinalities]
            )

        self.feature_id_emb = nn.Embedding(self.n_features, d_model)

    def forward(
        self,
        x_num: Optional[torch.Tensor],   # (B, n_num)  float32
        x_cat: Optional[torch.Tensor],   # (B, n_cat)  int64
    ) -> torch.Tensor:                   # (B, n_features, d_model)
        tokens: List[torch.Tensor] = []

        if self.n_num > 0 and x_num is not None:
            for i in range(self.n_num):
                xi = self.num_norm[i](x_num[:, i : i + 1])
                tokens.append(self.num_proj[i](xi))

        if self.n_cat > 0 and x_cat is not None:
            for i in range(self.n_cat):
                tokens.append(self.cat_emb[i](x_cat[:, i]))

        x   = torch.stack(tokens, dim=1)                       # (B, n_features, d_model)
        ids = torch.arange(self.n_features, device=x.device)
        x   = x + self.feature_id_emb(ids).unsqueeze(0)
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Transformer Encoder
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


class TabularEncoder(nn.Module):
    """
    Encodeur Transformer (style ViT) sur tokens de features.

    deep_supervision_layers : indices de couches (1-based) dont les représentations
    intermédiaires sont collectées pour la Deep Self-Supervision.
    """

    def __init__(
        self,
        d_model:                 int            = 128,
        n_heads:                 int            = 4,
        n_layers:                int            = 6,
        mlp_ratio:               float          = 4.0,
        dropout:                 float          = 0.1,
        deep_supervision_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.n_layers  = n_layers
        self.ds_layers = set(deep_supervision_layers or [n_layers])
        self.blocks    = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        intermediates: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            if i in self.ds_layers:
                intermediates.append(self.norm(x))
        if not intermediates:
            intermediates.append(self.norm(x))
        return self.norm(x), intermediates


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Predictor dense
# ══════════════════════════════════════════════════════════════════════════════

class TabularPredictor(nn.Module):
    """
    Petit Transformer qui prédit les représentations pour TOUS les tokens
    (Dense Predictive Loss). Les positions masquées reçoivent un mask_token appris.
    """

    def __init__(
        self,
        d_model:    int = 128,
        d_pred:     int = 64,
        n_heads:    int = 4,
        n_layers:   int = 3,
        n_features: int = 32,
    ) -> None:
        super().__init__()
        self.proj_in    = nn.Linear(d_model, d_pred)
        self.mask_token = nn.Parameter(torch.randn(d_pred) * 0.02)
        self.pos_emb    = nn.Embedding(n_features, d_pred)
        self.blocks     = nn.ModuleList(
            [_TransformerBlock(d_pred, n_heads) for _ in range(n_layers)]
        )
        self.norm     = nn.LayerNorm(d_pred)
        self.proj_out = nn.Linear(d_pred, d_model)

    def forward(
        self,
        ctx_out: torch.Tensor,   # (B, n_features, d_model)
        mask:    torch.Tensor,   # (B, n_features)  bool
    ) -> torch.Tensor:
        x = self.proj_in(ctx_out)
        x[mask] = self.mask_token
        pos = torch.arange(x.shape[1], device=x.device)
        x   = x + self.pos_emb(pos).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.proj_out(self.norm(x))


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Masquage par blocs de features
# ══════════════════════════════════════════════════════════════════════════════

def feature_block_mask(
    n_features: int,
    batch_size: int,
    mask_ratio: float           = 0.40,
    n_blocks:   int             = 2,
    device:     torch.device    = torch.device("cpu"),
) -> torch.Tensor:
    """Masque booléen (B, n_features) par blocs contigus aléatoires."""
    n_target   = max(1, int(n_features * mask_ratio))
    block_size = max(1, n_target // n_blocks)
    mask = torch.zeros(batch_size, n_features, dtype=torch.bool, device=device)
    for b in range(batch_size):
        masked: set = set()
        for _ in range(n_blocks):
            start = int(torch.randint(0, n_features, ()).item())
            for j in range(block_size):
                masked.add((start + j) % n_features)
        for idx in masked:
            mask[b, idx] = True
    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Dataset tabulaire générique
# ══════════════════════════════════════════════════════════════════════════════

class TabularDataset(Dataset):
    """
    Dataset PyTorch pour données tabulaires mixtes.

    Parameters
    ----------
    X_num : (N, n_num) float32  ou None
    X_cat : (N, n_cat) int64    ou None
    y     : (N,)                ou None   (None = mode SSL)
    """

    def __init__(
        self,
        X_num: Optional[np.ndarray] = None,
        X_cat: Optional[np.ndarray] = None,
        y:     Optional[np.ndarray] = None,
    ) -> None:
        ref = X_num if X_num is not None else X_cat
        assert ref is not None, "Au moins X_num ou X_cat requis"
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
#  6.  Tab-JEPA 2.1 — backbone SSL
# ══════════════════════════════════════════════════════════════════════════════

class TabJEPA21(nn.Module):
    """
    Backbone SSL Tab-JEPA 2.1.

    Utiliser TabJEPA21Trainer      pour le pré-entraînement non-supervisé.
    Utiliser TabJEPA21Supervised   pour le fine-tuning supervisé.
    """

    def __init__(
        self,
        n_num:                   int,
        cat_cardinalities:       List[int],
        d_model:                 int            = 128,
        n_encoder_layers:        int            = 6,
        n_heads:                 int            = 4,
        n_predictor_layers:      int            = 3,
        d_pred:                  int            = 64,
        mask_ratio:              float          = 0.40,
        n_mask_blocks:           int            = 2,
        ema_momentum:            float          = 0.996,
        deep_supervision_layers: Optional[List[int]] = None,
        dropout:                 float          = 0.1,
        vicreg_coef:             float          = 0.10,
    ) -> None:
        super().__init__()
        self.n_num      = n_num
        self.n_cat      = len(cat_cardinalities)
        self.n_features = n_num + len(cat_cardinalities)
        self.d_model    = d_model
        self.mask_ratio = mask_ratio
        self.n_mask_blocks  = n_mask_blocks
        self.ema_momentum   = ema_momentum
        self.vicreg_coef    = vicreg_coef

        _ds = deep_supervision_layers or [n_encoder_layers // 2, n_encoder_layers]

        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d_model)

        self.encoder = TabularEncoder(
            d_model=d_model, n_heads=n_heads,
            n_layers=n_encoder_layers, dropout=dropout,
            deep_supervision_layers=_ds,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = TabularPredictor(
            d_model=d_model, d_pred=d_pred,
            n_heads=max(1, d_pred // 32),
            n_layers=n_predictor_layers,
            n_features=self.n_features,
        )
        self.depth_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(len(_ds))]
        )

    # ── EMA ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def update_ema(self) -> None:
        m = self.ema_momentum
        for p_ctx, p_tgt in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1.0 - m)

    # ── SSL loss ──────────────────────────────────────────────────────────
    def ssl_loss(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
        mask:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        ref = x_num if x_num is not None else x_cat
        B, device = ref.shape[0], ref.device

        if mask is None:
            mask = feature_block_mask(
                self.n_features, B, self.mask_ratio, self.n_mask_blocks, device
            )

        tokens     = self.tokenizer(x_num, x_cat)
        ctx_tokens = tokens.clone()
        ctx_tokens[mask] = 0.0
        ctx_out, _ = self.encoder(ctx_tokens)

        with torch.no_grad():
            _, tgt_intermediates = self.target_encoder(tokens)

        pred_all = self.predictor(ctx_out, mask)

        total_loss = torch.tensor(0.0, device=device)
        metrics: Dict[str, float] = {}

        for i, (proj, tgt_rep) in enumerate(zip(self.depth_proj, tgt_intermediates)):
            target   = tgt_rep.detach()
            pred     = proj(pred_all)
            loss_d   = F.smooth_l1_loss(
                F.normalize(pred,   dim=-1),
                F.normalize(target, dim=-1),
            )
            std_tgt  = target.std(dim=0).mean()
            var_loss = F.relu(1.0 - std_tgt)
            total_loss = total_loss + loss_d + self.vicreg_coef * var_loss
            metrics[f"pred_loss_d{i}"] = loss_d.item()
            metrics[f"std_tgt_d{i}"]   = std_tgt.item()

        total_loss = total_loss / max(len(self.depth_proj), 1)
        metrics["total_ssl_loss"] = total_loss.item()
        return total_loss, metrics

    # ── Encode (bas niveau) ───────────────────────────────────────────────
    def encode(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
        pool:  str = "mean",
    ) -> torch.Tensor:                                        # (B, d_model)
        out, _ = self.encoder(self.tokenizer(x_num, x_cat))
        if pool == "mean":  return out.mean(dim=1)
        if pool == "max":   return out.max(dim=1).values
        return out[:, 0]

    # ── Extract embeddings (NumPy, en masse) ──────────────────────────────
    @torch.no_grad()
    def extract_embeddings(
        self,
        X_num:         Optional[np.ndarray] = None,
        X_cat:         Optional[np.ndarray] = None,
        batch_size:    int                  = 512,
        pool:          str                  = "mean",
        per_token:     bool                 = False,
        device:        Optional[str]        = None,
        feature_names: Optional[List[str]]  = None,
        scaler_X:      Optional[TabScaler]  = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extrait les embeddings en une passe sur des arrays NumPy.

        Paramètres
        ----------
        X_num        : (N, n_num) ou None
        X_cat        : (N, n_cat) ou None
        batch_size   : taille des mini-batchs inférés
        pool         : "mean" | "max" | "first"  — ignoré si per_token=True
        per_token    : True → retourne (N, n_features, d_model) sans pooling
        device       : None = device du modèle
        feature_names: noms des features (per_token=True) → dict {name: (N, d_model)}
        scaler_X     : TabScaler fitté sur X_num (appliqué si fourni)

        Retourne
        --------
        np.ndarray (N, d_model)               — pool & !per_token
        np.ndarray (N, n_features, d_model)   — per_token & !feature_names
        dict {str: np.ndarray (N, d_model)}   — per_token & feature_names
        """
        was_training = self.training
        self.eval()
        dev = torch.device(device) if device else next(self.parameters()).device

        ref = X_num if X_num is not None else X_cat
        assert ref is not None
        N = ref.shape[0]

        if scaler_X is not None and X_num is not None:
            X_num = scaler_X.transform(X_num)

        all_out: List[np.ndarray] = []

        for s in range(0, N, batch_size):
            e  = min(s + batch_size, N)
            xn = torch.from_numpy(X_num[s:e]).to(dev)                        if X_num is not None else None
            xc = torch.from_numpy(X_cat[s:e].astype(np.int64)).to(dev)       if X_cat is not None else None

            tokens = self.tokenizer(xn, xc)
            out, _ = self.encoder(tokens)

            if per_token:
                all_out.append(out.cpu().numpy())
            else:
                if pool == "mean":    z = out.mean(dim=1)
                elif pool == "max":   z = out.max(dim=1).values
                else:                 z = out[:, 0]
                all_out.append(z.cpu().numpy())

        if was_training:
            self.train()

        result = np.concatenate(all_out, axis=0)

        if per_token and feature_names is not None:
            assert len(feature_names) == self.n_features, (
                f"feature_names doit contenir {self.n_features} noms, "
                f"reçu {len(feature_names)}"
            )
            return {name: result[:, i, :] for i, name in enumerate(feature_names)}

        return result


# ══════════════════════════════════════════════════════════════════════════════
#  7.  TabJEPA21Trainer — pré-entraînement SSL
# ══════════════════════════════════════════════════════════════════════════════

class TabJEPA21Trainer:
    """
    Pré-entraîne le backbone Tab-JEPA 2.1 de façon non-supervisée.

    Fonctionnalités
    ---------------
    - StandardScaler automatique sur X_num (scale_features=True par défaut)
      → fité uniquement sur le split train pour éviter toute fuite de données
    - Split val automatique (val_size=0.15)
    - Early stopping avec restauration du meilleur état (patience)
    - EMA momentum schedule cosine (ema_start → ema_end)
    - Warmup linéaire du learning rate

    Entrée recommandée : fit_numpy(X_num, X_cat)
    Entrée alternative : fit(train_loader)   (DataLoader pré-construit)
    """

    def __init__(
        self,
        model:          TabJEPA21,
        lr:             float = 3e-4,
        weight_decay:   float = 1e-2,
        n_epochs:       int   = 100,
        warmup_epochs:  int   = 10,
        ema_start:      float = 0.996,
        ema_end:        float = 0.9999,
        grad_clip:      float = 1.0,
        val_size:       float = 0.15,
        patience:       int   = 15,
        batch_size:     int   = 256,
        scale_features: bool  = True,
        device:         str   = "cpu",
    ) -> None:
        self.model         = model.to(device)
        self.device        = device
        self.n_epochs      = n_epochs
        self.warmup_epochs = warmup_epochs
        self.grad_clip     = grad_clip
        self.ema_start     = ema_start
        self.ema_end       = ema_end
        self.base_lr       = lr
        self.val_size      = val_size
        self.patience      = patience
        self.batch_size    = batch_size
        self.scale_features = scale_features

        # Fitté dans fit_numpy() sur le split train uniquement
        self.scaler_X: Optional[TabScaler] = None

        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, n_epochs - warmup_epochs),
            eta_min=lr * 0.01,
        )

    # ── Entrée NumPy (recommandé) ─────────────────────────────────────────
    def fit_numpy(
        self,
        X_num:   Optional[np.ndarray] = None,
        X_cat:   Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Pré-entraîne directement depuis des arrays NumPy.
        Le StandardScaler est fité sur le split train (pas de data leakage).
        """
        # Split des indices d'abord, scaler ensuite
        ref    = X_num if X_num is not None else X_cat
        N      = ref.shape[0]
        n_val  = max(1, int(N * self.val_size))
        n_tr   = N - n_val
        gen    = torch.Generator().manual_seed(42)
        perm   = torch.randperm(N, generator=gen).numpy()
        tr_idx = perm[:n_tr]
        va_idx = perm[n_tr:]

        # Scaler X_num sur le train uniquement
        Xn_tr = Xn_va = None
        if X_num is not None and self.scale_features:
            self.scaler_X = TabScaler()
            Xn_tr = self.scaler_X.fit_transform(X_num[tr_idx])
            Xn_va = self.scaler_X.transform(X_num[va_idx])
        elif X_num is not None:
            Xn_tr = X_num[tr_idx].astype(np.float32)
            Xn_va = X_num[va_idx].astype(np.float32)

        Xc_tr = X_cat[tr_idx] if X_cat is not None else None
        Xc_va = X_cat[va_idx] if X_cat is not None else None

        train_ds = TabularDataset(Xn_tr, Xc_tr)
        val_ds   = TabularDataset(Xn_va, Xc_va)

        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              shuffle=True, drop_last=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size,
                              shuffle=False, drop_last=False)

        if verbose:
            print(
                f"[val_size={self.val_size}] split → "
                f"train={n_tr} | val={n_val}"
                + (f" | scaler_X fitté" if self.scaler_X else "")
            )
        return self._run(train_dl, val_dl, verbose)

    # ── Entrée DataLoader ─────────────────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   Optional[DataLoader] = None,
        verbose:      bool = True,
    ) -> Dict[str, List[float]]:
        """
        Entraîne depuis des DataLoaders pré-construits.
        Si val_loader=None et val_size>0, split automatique (sans scaling).
        """
        if val_loader is None and self.val_size > 0.0:
            train_loader, val_loader = self._split_loader(train_loader)
            if verbose:
                print(
                    f"[val_size={self.val_size}] split → "
                    f"train={len(train_loader.dataset)} | "
                    f"val={len(val_loader.dataset)}"
                )
        return self._run(train_loader, val_loader, verbose)

    # ── Boucle principale ─────────────────────────────────────────────────
    def _run(
        self,
        train_dl: DataLoader,
        val_dl:   Optional[DataLoader],
        verbose:  bool,
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val   = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        for epoch in range(self.n_epochs):
            tr_loss, _ = self._train_epoch(train_dl, epoch)
            history["train_loss"].append(tr_loss)
            val_str = ""
            es_str  = ""

            if val_dl is not None:
                val_loss = self._evaluate(val_dl)
                history["val_loss"].append(val_loss)
                val_str = f" | val={val_loss:.4f}"
                if val_loss < best_val - 1e-6:
                    best_val   = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    no_improve = 0
                    es_str = " ✓"
                else:
                    no_improve += 1
                    es_str = f" [{no_improve}/{self.patience}]"

            if verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1:03d}/{self.n_epochs}"
                    f" | train={tr_loss:.4f}{val_str}"
                    f" | lr={lr:.2e} | ema={self.model.ema_momentum:.4f}{es_str}"
                )

            if val_dl is not None and no_improve >= self.patience:
                if verbose:
                    print(f"\n⚑ Early stopping époque {epoch+1} (best val={best_val:.4f})")
                self.model.load_state_dict(best_state)
                return history

        if val_dl is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f"\n✓ Terminé. Meilleur val={best_val:.4f}")
        return history

    # ── Helpers ───────────────────────────────────────────────────────────
    def _ema_schedule(self, epoch: int) -> float:
        t = epoch / max(self.n_epochs, 1)
        return self.ema_end - (self.ema_end - self.ema_start) * (math.cos(math.pi * t) + 1) / 2

    def _warmup_lr(self, epoch: int) -> Optional[float]:
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / max(self.warmup_epochs, 1)
        return None

    def _train_epoch(
        self, loader: DataLoader, epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        self.model.ema_momentum = self._ema_schedule(epoch)
        wlr = self._warmup_lr(epoch)
        if wlr is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = wlr

        agg, n = 0.0, 0
        agg_m: Dict[str, float] = {}
        for batch in loader:
            xn = batch["x_num"].to(self.device) if "x_num" in batch else None
            xc = batch["x_cat"].to(self.device) if "x_cat" in batch else None
            self.optimizer.zero_grad()
            loss, metrics = self.model.ssl_loss(xn, xc)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.model.update_ema()
            agg += loss.item()
            for k, v in metrics.items():
                agg_m[k] = agg_m.get(k, 0.0) + v
            n += 1

        if wlr is None:
            self.scheduler.step()
        n = max(n, 1)
        return agg / n, {k: v / n for k, v in agg_m.items()}

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        tot, n = 0.0, 0
        for batch in loader:
            xn = batch["x_num"].to(self.device) if "x_num" in batch else None
            xc = batch["x_cat"].to(self.device) if "x_cat" in batch else None
            loss, _ = self.model.ssl_loss(xn, xc)
            tot += loss.item(); n += 1
        return tot / max(n, 1)

    def _split_loader(self, loader: DataLoader) -> Tuple[DataLoader, DataLoader]:
        ds    = loader.dataset
        N     = len(ds)
        n_val = max(1, int(N * self.val_size))
        gen   = torch.Generator().manual_seed(42)
        tr_s, va_s = random_split(ds, [N - n_val, n_val], generator=gen)
        bs    = loader.batch_size or self.batch_size
        nw, pm = loader.num_workers, loader.pin_memory
        return (
            DataLoader(tr_s, batch_size=bs, shuffle=True,  drop_last=True,  num_workers=nw, pin_memory=pm),
            DataLoader(va_s, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=pm),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  8.  TabJEPA21Supervised — estimateur sklearn-like
# ══════════════════════════════════════════════════════════════════════════════

class TabJEPA21Supervised:
    """
    Estimateur supervisé sur backbone Tab-JEPA 2.1. API sklearn-like.

    Tâches
    ------
    "regression"  MSE sur y scalé → predict() inverse-transforme
    "binary"      BCE+sigmoid      → predict()→{0,1}, predict_proba()→(N,2)
    "multiclass"  CrossEntropy     → predict()→{0…C-1}, predict_proba()→(N,C)

    Scaling automatique
    -------------------
    X_num est toujours standardisé (TabScaler fitté sur le split train).
    y est standardisé uniquement pour task="regression".

    Exemple
    -------
    sup = TabJEPA21Supervised(backbone, task="multiclass", n_outputs=5)
    sup.fit(X_num_tr, X_cat_tr, y_tr)
    labels = sup.predict(X_num_te, X_cat_te)
    proba  = sup.predict_proba(X_num_te, X_cat_te)
    """

    def __init__(
        self,
        backbone:          TabJEPA21,
        task:              str   = "multiclass",
        n_outputs:         int   = 1,        # ignoré pour "binary" (toujours 1 sortie)
        freeze_encoder:    bool  = False,    # True = linear probe
        # ── Architecture tête ──
        head_hidden_ratio: float = 0.5,
        head_dropout:      float = 0.1,
        # ── Optimiseur ──
        lr_encoder:        float = 3e-5,     # lr encodeur (plus faible)
        lr_head:           float = 1e-3,     # lr tête
        weight_decay:      float = 1e-2,
        # ── Boucle d'entraînement ──
        n_epochs:          int   = 100,
        warmup_epochs:     int   = 5,
        batch_size:        int   = 256,
        val_size:          float = 0.15,
        patience:          int   = 15,
        grad_clip:         float = 1.0,
        device:            str   = "cpu",
    ) -> None:
        assert task in ("regression", "binary", "multiclass"), \
            "task ∈ {'regression', 'binary', 'multiclass'}"

        self.backbone       = backbone.to(device)
        self.task           = task
        self.freeze_encoder = freeze_encoder
        self.device         = device
        self.n_epochs       = n_epochs
        self.warmup_epochs  = warmup_epochs
        self.batch_size     = batch_size
        self.val_size       = val_size
        self.patience       = patience
        self.grad_clip      = grad_clip
        self.lr_encoder     = lr_encoder
        self.lr_head        = lr_head
        self.weight_decay   = weight_decay

        # Scalers (fitté dans fit())
        self.scaler_X: Optional[TabScaler] = None   # features numériques
        self.scaler_y: Optional[TabScaler] = None   # cible (régression uniquement)

        # Tête de prédiction
        d     = backbone.d_model
        n_out = 1 if task == "binary" else n_outputs
        d_h   = max(16, int(d * head_hidden_ratio))
        self._head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d_h),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_h, n_out),
        ).to(device)

        self._is_fitted = False

    # ── fit() ─────────────────────────────────────────────────────────────
    def fit(
        self,
        X_num:   Optional[np.ndarray],
        X_cat:   Optional[np.ndarray],
        y:       np.ndarray,
        verbose: bool = True,
    ) -> "TabJEPA21Supervised":
        """
        Entraîne la tête supervisée (+ encodeur si freeze_encoder=False).

        Scaling
        -------
        - X_num → StandardScaler fitté sur le split train
        - y     → StandardScaler fitté sur le split train (régression uniquement)
        """
        # ── Split indices ─────────────────────────────────────────────────
        N      = y.shape[0]
        n_val  = max(1, int(N * self.val_size))
        n_tr   = N - n_val
        rng    = np.random.default_rng(42)
        perm   = rng.permutation(N)
        tr_idx = perm[:n_tr]
        va_idx = perm[n_tr:]

        if verbose:
            print(f"[val_size={self.val_size}] split → train={n_tr} | val={n_val}")

        # ── Scaler X_num (fitté sur train) ────────────────────────────────
        Xn_tr = Xn_va = None
        if X_num is not None:
            self.scaler_X = TabScaler()
            Xn_tr = self.scaler_X.fit_transform(X_num[tr_idx])
            Xn_va = self.scaler_X.transform(X_num[va_idx])
            if verbose:
                print(
                    f"  scaler_X → μ̄={self.scaler_X.mean_.mean():.3f} "
                    f"σ̄={self.scaler_X.std_.mean():.3f}"
                )

        Xc_tr = X_cat[tr_idx] if X_cat is not None else None
        Xc_va = X_cat[va_idx] if X_cat is not None else None

        # ── Scaler y (régression uniquement) ─────────────────────────────
        y_tr = y[tr_idx].astype(np.float32)
        y_va = y[va_idx].astype(np.float32)

        if self.task == "regression":
            self.scaler_y = TabScaler(clip=0)      # pas de clipping pour y
            y_tr = self.scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
            y_va = self.scaler_y.transform(y_va.reshape(-1, 1)).ravel()
            if verbose:
                print(
                    f"  scaler_y → μ={float(self.scaler_y.mean_):.3f} "
                    f"σ={float(self.scaler_y.std_):.3f}"
                )
        elif self.task == "multiclass":
            y_tr = y_tr.astype(np.int64)
            y_va = y_va.astype(np.int64)

        # ── Datasets & Loaders ────────────────────────────────────────────
        train_ds = TabularDataset(Xn_tr, Xc_tr, y_tr)
        val_ds   = TabularDataset(Xn_va, Xc_va, y_va)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              shuffle=True, drop_last=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size,
                              shuffle=False, drop_last=False)

        # ── Optimiseur (dual LR) ──────────────────────────────────────────
        param_groups = [{"params": list(self._head.parameters()), "lr": self.lr_head}]
        if not self.freeze_encoder:
            param_groups.append({
                "params": [p for p in self.backbone.encoder.parameters() if p.requires_grad],
                "lr": self.lr_encoder,
            })
        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.n_epochs - self.warmup_epochs),
            eta_min=self.lr_head * 0.01,
        )

        # ── Boucle ────────────────────────────────────────────────────────
        best_val   = float("inf")
        best_state = {
            "head":    copy.deepcopy(self._head.state_dict()),
            "encoder": copy.deepcopy(self.backbone.encoder.state_dict()),
        }
        no_improve = 0

        for epoch in range(self.n_epochs):
            # Warmup
            if epoch < self.warmup_epochs:
                scale = (epoch + 1) / max(self.warmup_epochs, 1)
                for i, pg in enumerate(optimizer.param_groups):
                    base = self.lr_head if i == 0 else self.lr_encoder
                    pg["lr"] = base * scale

            # Train
            self.backbone.train()
            self._head.train()
            tr_loss, n_tr_b = 0.0, 0

            for batch in train_dl:
                xn = batch["x_num"].to(self.device) if "x_num" in batch else None
                xc = batch["x_cat"].to(self.device) if "x_cat" in batch else None
                yb = batch["y"].to(self.device)

                z      = self.backbone.encode(xn, xc)
                if self.freeze_encoder:
                    z = z.detach()
                logits = self._head(z).squeeze(-1)
                loss   = self._loss(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                all_p = list(self._head.parameters())
                if not self.freeze_encoder:
                    all_p += [p for p in self.backbone.encoder.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(all_p, self.grad_clip)
                optimizer.step()
                tr_loss += loss.item(); n_tr_b += 1

            if epoch >= self.warmup_epochs:
                scheduler.step()
            tr_loss /= max(n_tr_b, 1)

            # Validation
            val_loss = self._eval_sup(val_dl)
            es_str   = ""
            if val_loss < best_val - 1e-6:
                best_val   = val_loss
                best_state = {
                    "head":    copy.deepcopy(self._head.state_dict()),
                    "encoder": copy.deepcopy(self.backbone.encoder.state_dict()),
                }
                no_improve = 0
                es_str = " ✓"
            else:
                no_improve += 1
                es_str = f" [{no_improve}/{self.patience}]"

            if verbose:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1:03d}/{self.n_epochs}"
                    f" | train={tr_loss:.4f} | val={val_loss:.4f}"
                    f" | lr={lr:.2e}{es_str}"
                )

            if no_improve >= self.patience:
                if verbose:
                    print(f"\n⚑ Early stopping époque {epoch+1} (best val={best_val:.4f})")
                break

        # Restauration du meilleur état
        self._head.load_state_dict(best_state["head"])
        self.backbone.encoder.load_state_dict(best_state["encoder"])
        if verbose:
            print(f"\n✓ fit() terminé. Meilleur val={best_val:.4f}")

        self._is_fitted = True
        return self

    # ── predict() ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        X_num:      Optional[np.ndarray],
        X_cat:      Optional[np.ndarray],
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        Prédit les labels / valeurs.

        regression  → np.ndarray (N,) float32, dans l'espace original (inverse-scalé)
        binary      → np.ndarray (N,) int64  ∈ {0, 1}
        multiclass  → np.ndarray (N,) int64  ∈ {0, …, C−1}
        """
        self._check_fitted()
        logits = self._forward(X_num, X_cat, batch_size)   # (N, n_out)

        if self.task == "regression":
            preds = logits.ravel()
            if self.scaler_y is not None:
                preds = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
            return preds

        if self.task == "binary":
            return (
                (torch.sigmoid(torch.from_numpy(logits)).numpy().ravel() >= 0.5)
                .astype(np.int64)
            )

        # multiclass
        return np.argmax(logits, axis=1).astype(np.int64)

    # ── predict_proba() ───────────────────────────────────────────────────
    @torch.no_grad()
    def predict_proba(
        self,
        X_num:      Optional[np.ndarray],
        X_cat:      Optional[np.ndarray],
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        Retourne les probabilités de classes.

        binary      → (N, 2)   [[P(y=0), P(y=1)], …]
        multiclass  → (N, C)   softmax sur C classes
        regression  → lève ValueError
        """
        if self.task == "regression":
            raise ValueError(
                "predict_proba() n'est pas disponible pour la régression. "
                "Utilisez predict()."
            )
        self._check_fitted()
        logits = self._forward(X_num, X_cat, batch_size)

        if self.task == "binary":
            p1 = torch.sigmoid(torch.from_numpy(logits)).numpy().ravel()
            return np.column_stack([1.0 - p1, p1]).astype(np.float32)

        # multiclass
        return torch.softmax(torch.from_numpy(logits), dim=1).numpy().astype(np.float32)

    # ── Helpers internes ─────────────────────────────────────────────────
    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.task == "regression":   return F.mse_loss(logits, y)
        if self.task == "binary":       return F.binary_cross_entropy_with_logits(logits, y)
        return F.cross_entropy(logits, y)

    @torch.no_grad()
    def _eval_sup(self, loader: DataLoader) -> float:
        self.backbone.eval(); self._head.eval()
        tot, n = 0.0, 0
        for batch in loader:
            xn = batch["x_num"].to(self.device) if "x_num" in batch else None
            xc = batch["x_cat"].to(self.device) if "x_cat" in batch else None
            yb = batch["y"].to(self.device)
            logits = self._head(self.backbone.encode(xn, xc)).squeeze(-1)
            tot += self._loss(logits, yb).item(); n += 1
        return tot / max(n, 1)

    @torch.no_grad()
    def _forward(
        self,
        X_num:      Optional[np.ndarray],
        X_cat:      Optional[np.ndarray],
        batch_size: int,
    ) -> np.ndarray:
        """Passe forward en batchs → logits bruts numpy (N, n_out)."""
        self.backbone.eval(); self._head.eval()

        if self.scaler_X is not None and X_num is not None:
            X_num = self.scaler_X.transform(X_num)

        ref = X_num if X_num is not None else X_cat
        N   = ref.shape[0]
        out: List[np.ndarray] = []

        for s in range(0, N, batch_size):
            e  = min(s + batch_size, N)
            xn = torch.from_numpy(X_num[s:e]).to(self.device)                     if X_num is not None else None
            xc = torch.from_numpy(X_cat[s:e].astype(np.int64)).to(self.device)    if X_cat is not None else None
            logits = self._head(self.backbone.encode(xn, xc))
            out.append(logits.cpu().numpy())

        return np.concatenate(out, axis=0)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Appelez .fit() avant .predict() / .predict_proba()")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"TabJEPA21Supervised("
            f"task={self.task!r}, "
            f"d_model={self.backbone.d_model}, "
            f"n_features={self.backbone.n_features}, "
            f"freeze_encoder={self.freeze_encoder}, "
            f"status={status!r})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  9.  Smoke test complet
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Tab-JEPA 2.1 — smoke test complet")
    print("=" * 65)

    torch.manual_seed(42)
    np.random.seed(42)

    N         = 4_000
    n_num     = 15
    cat_card  = [4, 8, 3, 10, 6]
    n_classes = 4
    feat_names = (
        [f"num_{i}" for i in range(n_num)] +
        [f"cat_{i}" for i in range(len(cat_card))]
    )

    # Features volontairement non-centrées pour valider le scaling
    X_num = (np.random.randn(N, n_num) * 50 + 100).astype(np.float32)
    X_cat = np.column_stack([np.random.randint(0, c, N) for c in cat_card])
    y_reg = (X_num[:, 0] * 2.5 + np.random.randn(N) * 10).astype(np.float32)
    y_bin = (y_reg > y_reg.mean()).astype(np.float32)
    y_clf = np.random.randint(0, n_classes, N).astype(np.int64)

    # ── Backbone partagé ─────────────────────────────────────────────────
    backbone = TabJEPA21(
        n_num=n_num, cat_cardinalities=cat_card,
        d_model=64, n_encoder_layers=4, n_heads=4,
        n_predictor_layers=2, d_pred=32,
        mask_ratio=0.40, n_mask_blocks=2,
        deep_supervision_layers=[2, 4], dropout=0.1,
    )
    n_p = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"\nParamètres backbone : {n_p:,}")
    print(f"Features            : {backbone.n_features}  ({n_num} num + {len(cat_card)} cat)")

    # ═════ PHASE 1 : Pré-entraînement SSL ════════════════════════════════
    print("\n── Phase 1 : Pré-entraînement SSL (fit_numpy) ──")
    ssl_trainer = TabJEPA21Trainer(
        backbone, n_epochs=4, warmup_epochs=1,
        val_size=0.15, patience=3, batch_size=256,
        scale_features=True, device="cpu",
    )
    ssl_trainer.fit_numpy(X_num, X_cat, verbose=True)

    # ═════ PHASE 2 : extract_embeddings ══════════════════════════════════
    print("\n── Phase 2 : extract_embeddings ──")
    emb = backbone.extract_embeddings(
        X_num, X_cat, pool="mean", scaler_X=ssl_trainer.scaler_X
    )
    print(f"  pooled    : {emb.shape}   dtype={emb.dtype}")

    emb_pt = backbone.extract_embeddings(
        X_num, X_cat, per_token=True, scaler_X=ssl_trainer.scaler_X
    )
    print(f"  per_token : {emb_pt.shape}")

    emb_d = backbone.extract_embeddings(
        X_num, X_cat, per_token=True,
        feature_names=feat_names, scaler_X=ssl_trainer.scaler_X
    )
    print(f"  dict keys[:3] : {list(emb_d.keys())[:3]}  shape={emb_d['num_0'].shape}")

    # ═════ PHASE 3 : Supervisé — régression ══════════════════════════════
    print("\n── Phase 3 : Supervisé — régression ──")
    sup_reg = TabJEPA21Supervised(
        copy.deepcopy(backbone), task="regression", n_outputs=1,
        n_epochs=6, warmup_epochs=1, val_size=0.15, patience=3, batch_size=256,
    )
    sup_reg.fit(X_num, X_cat, y_reg, verbose=True)
    preds_reg = sup_reg.predict(X_num[:16], X_cat[:16])
    print(f"  predict()  → shape={preds_reg.shape}  sample={preds_reg[:3].round(1)}")
    try:
        sup_reg.predict_proba(X_num[:4], X_cat[:4])
    except ValueError as e:
        print(f"  predict_proba bloqué ✓ : {e}")

    # ═════ PHASE 4 : Supervisé — binaire ═════════════════════════════════
    print("\n── Phase 4 : Supervisé — binaire ──")
    sup_bin = TabJEPA21Supervised(
        copy.deepcopy(backbone), task="binary", n_outputs=1,
        n_epochs=6, warmup_epochs=1, val_size=0.15, patience=3, batch_size=256,
    )
    sup_bin.fit(X_num, X_cat, y_bin, verbose=True)
    preds_bin = sup_bin.predict(X_num[:16], X_cat[:16])
    proba_bin = sup_bin.predict_proba(X_num[:16], X_cat[:16])
    print(f"  predict()       → shape={preds_bin.shape}  unique={np.unique(preds_bin)}")
    print(f"  predict_proba() → shape={proba_bin.shape}  sum={proba_bin[:2].sum(1).round(4)}")

    # ═════ PHASE 5 : Supervisé — multiclasses ════════════════════════════
    print("\n── Phase 5 : Supervisé — multiclasses ──")
    sup_clf = TabJEPA21Supervised(
        copy.deepcopy(backbone), task="multiclass", n_outputs=n_classes,
        n_epochs=6, warmup_epochs=1, val_size=0.15, patience=3, batch_size=256,
    )
    sup_clf.fit(X_num, X_cat, y_clf, verbose=True)
    preds_clf = sup_clf.predict(X_num[:16], X_cat[:16])
    proba_clf = sup_clf.predict_proba(X_num[:16], X_cat[:16])
    print(f"  predict()       → shape={preds_clf.shape}  classes={np.unique(preds_clf)}")
    print(f"  predict_proba() → shape={proba_clf.shape}  sum={proba_clf[:2].sum(1).round(4)}")
    print(f"\n  repr : {sup_clf}")

    print("\n✓ Smoke test complet réussi.")
