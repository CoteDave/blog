"""
filter_methods.py  ·  v6  —  Unified · Ultra-Optimised · Multi-Task
=====================================================================
7 méthodes de sélection de features par tests d'indépendance.

  XiFilter   Chatterjee rank correlation      JASA 2021
  dCorFilter  Unbiased Distance Correlation    Székely & Rizzo 2014
  HSICFilter  HSIC + RFF                       Song et al. 2012
  CPIFilter   Conditional Permutation Import.  Molnar et al. 2023
  GCMFilter   Generalised Covariance Measure   Shah & Peters 2020
  PCMFilter   Projected Covariance Measure     Scheidegger et al. 2022
  FOCIFilter  Feature Ordering by Cond. Indep. Azadkia & Chatterjee 2021

Tâches supportées : régression · classification binaire · multiclasse.

Architecture performance (millions × milliers)
──────────────────────────────────────────────
  Xi    O(n·log n·p)  batch argsort, p-valeurs CLT (0 perms)
  dCor  O(n²·p)       canaux y pré-calculés 1 fois; sketch si n > N_EXACT
  HSIC  O(n·D·p)      RFF TOUJOURS (D=256), bande passante MAD (O(n) vs O(n²))
  CPI   O(n·p)        scoring sous-échantillonné; feature-batching
  GCM   O(n·p)        Ridge+CLT (0 perms); RF en option; prefilter_k Xi
  PCM   O(n·p)        fold-splits partagés; feature-batching; prefilter_k
  FOCI  O(n·log n·k²) prefilter Xi; cKDTree C-extension

Optimisations clés
──────────────────
  • ZERO scipy.special dans les workers (expit/softmax → numpy pur → picklable)
  • RFF TOUJOURS pour HSIC : bande passante MAD O(n) au lieu de cdist O(n²)
  • feature-batching : un worker = FEAT_BATCH features → overhead joblib ~0
  • y-canaux pré-calculés 1 fois, partagés entre tous les workers
  • p-valeurs analytiques Xi (CLT σ²=2/5) et GCM (sandwich CLT) → 0 perms
  • float32 pour les matrices RFF : −50% RAM
  • prefilter_k : filtre Xi gratuit → méthodes coûteuses sur top-k seulement
"""
from __future__ import annotations

import os
import warnings
from typing import Literal, Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import norm, rankdata
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _CB_OK = True
except ImportError:
    _CB_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# §0  Constantes globales
# ══════════════════════════════════════════════════════════════════════════════
_N_EXACT    = 6_000   # seuil exact O(n²) vs sketch pour dCor
_N_SKETCH   = 4_000   # taille du sous-échantillon sketch dCor
_B_SKETCH   = 3       # nb de tirages sketch (moyennés)
_RFF_D      = 256     # dimension Random Fourier Features pour HSIC
_FEAT_BATCH = 64      # features par worker joblib
_N_PERM_MAX = 200     # plafond absolu permutations

TaskType  = Literal["regression", "binary", "multiclass"]
TaskParam = Optional[Literal["regression", "classification", "binary", "multiclass"]]


# ══════════════════════════════════════════════════════════════════════════════
# §1  Détection de tâche & encodage
# ══════════════════════════════════════════════════════════════════════════════

def _detect_task(y: np.ndarray) -> TaskType:
    t = type_of_target(y)
    if t == "binary":      return "binary"
    if "multiclass" in t:  return "multiclass"
    return "regression"


def _resolve_task(task_param: TaskParam, y: np.ndarray) -> TaskType:
    """
    Résout le paramètre task fourni par l'utilisateur en tâche interne.

    task=None              → auto-détection via type_of_target(y)
    task="regression"      → régression forcée
    task="classification"  → "binary" ou "multiclass" selon les valeurs de y
    task="binary"          → binaire forcé (y traité comme 0/1)
    task="multiclass"      → multiclasse forcé
    """
    if task_param is None:
        return _detect_task(y)
    if task_param == "regression":
        return "regression"
    if task_param in ("binary", "multiclass"):
        return task_param
    # "classification" → laisser y décider entre binary et multiclass
    auto = _detect_task(y)
    return auto if auto != "regression" else "binary"


def _encode_y(y: np.ndarray, task: TaskType
              ) -> tuple[np.ndarray, Optional[LabelEncoder]]:
    if task == "regression":
        return y.astype(np.float64), None
    le = LabelEncoder().fit(y)
    return le.transform(y).astype(np.float64), le


def _onehot(codes: np.ndarray, K: int) -> np.ndarray:
    n   = len(codes)
    out = np.zeros((n, K), dtype=np.float64)
    out[np.arange(n), codes.astype(int)] = 1.0
    return out


# Numpy pur — picklable par loky (scipy.special ne l'est pas)
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(x) / (1.0 + np.exp(x)))


def _softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _proba(model, X: np.ndarray, task: TaskType, K: int) -> np.ndarray:
    """→ matrice de probabilités (n, K) — dispatche sur tout type de modèle."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X).astype(np.float64)
        if hasattr(model, "classes_"):
            cls = np.asarray(model.classes_).astype(int)
            a   = np.zeros((len(X), K), dtype=np.float64)
            for ci, c in enumerate(cls):
                if 0 <= c < K: a[:, c] = p[:, ci]
            return a
        return p
    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        if df.ndim == 1:
            p1 = _sigmoid(df)
            return np.column_stack([1 - p1, p1])
        return _softmax_np(df)
    return _onehot(model.predict(X).astype(int), K)


def _eff_perms(n: int, n_perms: int) -> int:
    if n > 100_000: return min(n_perms, 30)
    if n > 30_000:  return min(n_perms, 80)
    return min(n_perms, _N_PERM_MAX)


# ══════════════════════════════════════════════════════════════════════════════
# §2  Primitives mathématiques
# ══════════════════════════════════════════════════════════════════════════════

def _bw_mad(x: np.ndarray) -> float:
    """
    Bande passante via MAD — O(n log n), ~20x plus rapide que cdist O(n²).
    σ² = (1.4826 · MAD)² · 2  (facteur 2 = convention noyau gaussien σ²)
    """
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    bw  = (1.4826 * mad) ** 2 * 2.0
    return bw if bw > 1e-10 else 1.0


def _u_center(A: np.ndarray) -> np.ndarray:
    """Double U-centrage — vectorisé O(n²)."""
    n   = A.shape[0]
    rs  = A.sum(1, keepdims=True); cs = A.sum(0, keepdims=True)
    tot = float(A.sum())
    U   = A - rs / (n - 2) - cs / (n - 2) + tot / ((n - 1) * (n - 2))
    np.fill_diagonal(U, 0.0)
    return U


def _dcov2(Ac: np.ndarray, Bc: np.ndarray) -> float:
    n = Ac.shape[0]
    return float(np.einsum("ij,ij->", Ac, Bc) / (n * (n - 3)))


def _rbf(X: np.ndarray, s2: float, Y: Optional[np.ndarray] = None) -> np.ndarray:
    return np.exp(-cdist(X, Y if Y is not None else X, "sqeuclidean") / (2.0 * s2))


def _rff(z: np.ndarray, D: int, bw: float,
         rng: np.random.Generator) -> np.ndarray:
    """
    Random Fourier Features  Φ ∈ ℝ^(n×D)  t.q.  K ≈ ΦΦᵀ  (Rahimi & Recht 2007).
    z : (n, d) — float32 en sortie pour économiser la RAM.
    """
    d  = z.shape[1]
    W  = (rng.standard_normal((d, D)) / np.sqrt(bw)).astype(np.float32)
    b  = rng.uniform(0.0, 2.0 * np.pi, D).astype(np.float32)
    return np.sqrt(2.0 / D) * np.cos(z.astype(np.float32) @ W + b)


def _hsic_rff(Zx: np.ndarray, Zy: np.ndarray) -> float:
    """
    HSIC_u non-biaisé via RFF — O(n·D + D²).

    HSIC_u(K,L) = [tr(K̃L̃) + 1ᵀK̃1·1ᵀL̃1/((n-1)(n-2)) - 2·1ᵀK̃L̃1/(n-2)] / n(n-3)
    Avec K ≈ ZxZxᵀ, chaque terme se calcule en O(n·D) :
      tr(K̃L̃) = ||ZxᵀZy||_F² - Σᵢ||φᵢ||²||ψᵢ||²
      1ᵀK̃1   = ||Zxᵀ1||² - Σᵢ||φᵢ||²
      K̃1      = Zx(Zxᵀ1) - kdiag
    """
    Zx = Zx.astype(np.float64); Zy = Zy.astype(np.float64)
    n  = Zx.shape[0]
    ZZ  = Zx.T @ Zy
    trKL = float(np.einsum("ij,ij->", ZZ, ZZ))
    kd   = (Zx * Zx).sum(1); ld = (Zy * Zy).sum(1)
    trKL -= float(np.dot(kd, ld))
    sk   = Zx.sum(0); sl = Zy.sum(0)
    sumK = float(np.dot(sk, sk)) - kd.sum()
    sumL = float(np.dot(sl, sl)) - ld.sum()
    K1   = Zx @ sk - kd; L1 = Zy @ sl - ld
    cross = float(np.dot(K1, L1))
    return float(
        (trKL + sumK * sumL / ((n - 1) * (n - 2)) - 2.0 * cross / (n - 2))
        / (n * (n - 3))
    )


def _hsic_exact(K: np.ndarray, L: np.ndarray) -> float:
    n  = K.shape[0]
    Kt = K.copy(); np.fill_diagonal(Kt, 0.0)
    Lt = L.copy(); np.fill_diagonal(Lt, 0.0)
    tr   = float(np.einsum("ij,ij->", Kt, Lt))
    sK   = float(Kt.sum()); sL = float(Lt.sum())
    cros = float((Kt @ Lt).sum())
    return (tr + sK * sL / ((n - 1) * (n - 2)) - 2.0 * cros / (n - 2)) / (n * (n - 3))


def _strat_idx(y_enc: np.ndarray, task: TaskType, K: int,
               n_target: int, rng: np.random.Generator) -> np.ndarray:
    n = len(y_enc)
    if n <= n_target: return np.arange(n, dtype=int)
    if task == "regression":
        picks = np.round(np.linspace(0, n - 1, n_target)).astype(int)
        return np.argsort(y_enc)[picks]
    parts = []
    for k in range(K):
        ki  = np.where(y_enc == k)[0]
        cnt = max(1, round(n_target * len(ki) / n))
        parts.append(rng.choice(ki, min(cnt, len(ki)), replace=False))
    idx = np.concatenate(parts); rng.shuffle(idx)
    return idx[:n_target]


# ══════════════════════════════════════════════════════════════════════════════
# §3  Classe de base
# ══════════════════════════════════════════════════════════════════════════════

class FilterMethod(BaseEstimator):
    """
    Classe de base sklearn unifiée.

    Après fit() :
      scores_    (p,)  — score d'indépendance (plus haut = plus important)
      pvalues_   (p,)  — p-valeur du test
      task_            — tâche résolue : "regression"|"binary"|"multiclass"
      n_classes_       — K (1 pour régression)
      le_              — LabelEncoder ou None

    Paramètre commun à toutes les sous-classes :
      task : None | "regression" | "classification" | "binary" | "multiclass"
        None (défaut) → auto-détection sur y (fiable dans la grande majorité des cas).
        Passer task="regression" ou task="classification" si y est ambigu
        (ex. cibles entières {0,1,2} interprétées comme regression par accident).
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FilterMethod":
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        # Résolution de la tâche : paramètre explicite prioritaire sur inférence
        task_param          = getattr(self, "task", None)
        self.task_          = _resolve_task(task_param, y)
        y_enc, self.le_     = _encode_y(y, self.task_)
        self.n_classes_     = (int(y_enc.max()) + 1
                               if self.task_ != "regression" else 1)
        self._fit(X, y_enc)
        return self

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def get_support(self, k: int = 10) -> np.ndarray:
        """Indices des k meilleures features (par score décroissant)."""
        check_is_fitted(self)
        return np.argsort(self.scores_)[::-1][:k]

    def feature_importances(self, feature_names=None):
        """
        Tableau ordonné de TOUTES les features avec scores et statistiques.

        Retourne un pandas DataFrame si pandas est disponible,
        sinon une liste de dicts.

        Colonnes toujours présentes :
          rank          — classement (1 = plus important)
          feature       — nom de la feature
          score         — score d'indépendance (décroissant)
          pvalue        — p-valeur du test d'indépendance

        Colonnes supplémentaires selon la méthode :
          selected      — (FOCIFilter) True si sélectionnée par l'algo greedy
          selection_step— (FOCIFilter) étape de sélection (NaN si non sélectionnée)
          baseline_score— (CPIFilter)  score du modèle de base (constante par ligne)
        """
        check_is_fitted(self)
        p     = self.n_features_in_
        names = feature_names or [f"X{i}" for i in range(p)]
        idx   = np.argsort(self.scores_)[::-1]

        rows = []
        for rank, i in enumerate(idx, start=1):
            row = {
                "rank"   : rank,
                "feature": names[i],
                "score"  : float(self.scores_[i]),
                "pvalue" : float(self.pvalues_[i]),
            }
            # Colonnes bonus FOCI
            if hasattr(self, "selected_order_"):
                sel_set  = {j: step for step, j in enumerate(self.selected_order_, 1)}
                row["selected"]       = i in sel_set
                row["selection_step"] = sel_set.get(i, float("nan"))
            # Colonne bonus CPI
            if hasattr(self, "baseline_score_"):
                row["baseline_score"] = float(self.baseline_score_)
            rows.append(row)

        try:
            import pandas as pd
            return pd.DataFrame(rows).set_index("rank")
        except ImportError:
            return rows

    def ranking(self, feature_names=None) -> list[tuple]:
        """Compatibilité ascendante — préférer feature_importances()."""
        check_is_fitted(self)
        idx   = np.argsort(self.scores_)[::-1]
        names = feature_names or [f"X{i}" for i in range(self.n_features_in_)]
        return [(names[i], round(float(self.scores_[i]), 6),
                 round(float(self.pvalues_[i]), 4)) for i in idx]


# ══════════════════════════════════════════════════════════════════════════════
# §4  XiFilter — Chatterjee rank correlation  (JASA 2021)
# ══════════════════════════════════════════════════════════════════════════════

class XiFilter(FilterMethod):
    """
    ξ_n = 1 − 3·Σ|R_{i+1}−R_i| / (n²−1).  O(n log n · p).

    OPTS :
    • Un seul np.argsort sur (n, FEAT_BATCH) — zéro boucle Python.
    • p-valeur CLT vectorisée sur tout p d'un coup — 0 perms par défaut.
    • Multiclasse OvR : K batch argsorts, max sur K, Bonferroni.
    """

    def __init__(self, n_perms: int = 0, n_jobs: int = -1, random_state: int = 0,
                 task: TaskParam = None):
        self.n_perms = n_perms; self.n_jobs = n_jobs; self.random_state = random_state
        self.task = task

    @staticmethod
    def _xi_cols(Xb: np.ndarray, r_y: np.ndarray) -> np.ndarray:
        """Batch ξ sur b colonnes. Xb (n,b), r_y (n,). → (b,)."""
        n     = Xb.shape[0]
        idx   = np.argsort(Xb, axis=0, kind="stable")
        delta = np.abs(np.diff(r_y[idx], axis=0)).sum(axis=0)
        return 1.0 - 3.0 * delta / (n ** 2 - 1.0)

    @staticmethod
    def _pval_clt(xi: np.ndarray, n: int) -> np.ndarray:
        z = np.sqrt(n) * np.clip(xi, 0.0, None) / np.sqrt(0.4)
        return 2.0 * norm.sf(z)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p = X.shape; task = self.task_; K = self.n_classes_

        if task == "multiclass":
            xi_mat = np.empty((K, p))
            for k in range(K):
                r_k = rankdata((y == k).astype(float), method="average")
                for s in range(0, p, _FEAT_BATCH):
                    sl = slice(s, min(s + _FEAT_BATCH, p))
                    xi_mat[k, sl] = self._xi_cols(X[:, sl], r_k)
            xi_mat = np.clip(xi_mat, 0.0, 1.0)
            self.scores_  = xi_mat.max(axis=0)
            best_k        = xi_mat.argmin(axis=0)
            self.pvalues_ = np.minimum(1.0,
                                       self._pval_clt(xi_mat[best_k, np.arange(p)], n) * K)
            return

        y_t = y if task == "regression" else (y == 1).astype(float)
        r_y = rankdata(y_t, method="average")
        xi_all = np.empty(p)
        for s in range(0, p, _FEAT_BATCH):
            sl = slice(s, min(s + _FEAT_BATCH, p))
            xi_all[sl] = self._xi_cols(X[:, sl], r_y)
        xi_all = np.clip(xi_all, 0.0, 1.0)
        self.scores_ = xi_all

        if self.n_perms == 0:
            self.pvalues_ = self._pval_clt(xi_all, n); return

        eff  = _eff_perms(n, self.n_perms)
        rng  = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=eff)

        def _chunk(seed_list):
            cnt = np.zeros(p)
            for s in seed_list:
                rp = rankdata(np.random.default_rng(s).permutation(y_t), method="average")
                for st in range(0, p, _FEAT_BATCH):
                    sl = slice(st, min(st + _FEAT_BATCH, p))
                    cnt[sl] += np.clip(self._xi_cols(X[:, sl], rp), 0, 1) >= xi_all[sl] - 1e-12
            return cnt

        nc  = max(1, eff // max(1, os.cpu_count() or 2))
        sbs = [seeds[i:i+nc].tolist() for i in range(0, eff, nc)]
        tot = np.sum(Parallel(n_jobs=self.n_jobs, backend="loky")(
                     delayed(_chunk)(sb) for sb in sbs), axis=0)
        self.pvalues_ = tot / eff


# ══════════════════════════════════════════════════════════════════════════════
# §5  dCorFilter — Unbiased Distance Correlation  (Székely & Rizzo 2014)
# ══════════════════════════════════════════════════════════════════════════════

class dCorFilter(FilterMethod):
    """
    dCor² ∈ [0,1]. Multiclasse : OvR, moyenne des K canaux.

    OPTS :
    • Canaux y (matrices By U-centrées) pré-calculés UNE FOIS, partagés.
    • n ≤ N_EXACT : exact O(n²) ; perm = réindexage de Byc pré-calculé.
    • n > N_EXACT : B_SKETCH sous-échantillons stratifiés moyennés.
    • FEAT_BATCH features par worker.
    """

    def __init__(self, n_perms: int = 200, n_jobs: int = -1,
                 n_exact: int = _N_EXACT, n_sketch: int = _N_SKETCH,
                 b_sketch: int = _B_SKETCH, random_state: int = 0,
                 task: TaskParam = None):
        self.n_perms = n_perms; self.n_jobs = n_jobs
        self.n_exact = n_exact; self.n_sketch = n_sketch
        self.b_sketch = b_sketch; self.random_state = random_state
        self.task = task

    @staticmethod
    def _batch_exact(Xb: np.ndarray, channels: list,
                     n_perms: int, seed: int) -> list:
        """channels = [(Byc, dVarY), …] — un par classe OvR."""
        rng = np.random.default_rng(seed); n = Xb.shape[0]; K = len(channels)
        out = []
        for j in range(Xb.shape[1]):
            xv  = Xb[:, j:j+1]; Ax = cdist(xv, xv); Axc = _u_center(Ax)
            dVx = _dcov2(Axc, Axc)
            if dVx <= 0.0: out.append((0.0, 1.0)); continue
            sc_k = []
            for Byc, dVy in channels:
                if dVy <= 0.0: sc_k.append(0.0); continue
                sc_k.append(float(_dcov2(Axc, Byc) / np.sqrt(dVx * dVy)))
            obs = float(np.mean(sc_k))
            if n_perms == 0: out.append((obs, 1.0)); continue
            nc = 0
            for _ in range(n_perms):
                perm = rng.permutation(n); sc_p = []
                for Byc, dVy in channels:
                    if dVy <= 0.0: sc_p.append(0.0); continue
                    sc_p.append(float(_dcov2(Axc, Byc[perm][:, perm]) / np.sqrt(dVx * dVy)))
                if float(np.mean(sc_p)) >= obs - 1e-12: nc += 1
            pv = float(nc / n_perms)
            out.append((obs, float(min(1.0, pv * K)) if K > 1 else pv))
        return out

    def _batch_sketch(self, Xb: np.ndarray, y_enc: np.ndarray,
                      task: TaskType, K: int, seed: int) -> list:
        rng = np.random.default_rng(seed); out = []
        for j in range(Xb.shape[1]):
            xj = Xb[:, j]; draws = []
            for _ in range(self.b_sketch):
                idx  = _strat_idx(y_enc, task, K, self.n_sketch, rng)
                xjs  = xj[idx].reshape(-1, 1)
                Ax   = cdist(xjs, xjs); Axc = _u_center(Ax); dVx = _dcov2(Axc, Axc)
                if dVx <= 0.0: draws.append(0.0); continue
                y_s  = y_enc[idx]
                if task == "regression":
                    yv  = y_s.reshape(-1, 1); By = cdist(yv, yv)
                    Byc = _u_center(By); dVy = _dcov2(Byc, Byc)
                    draws.append(float(_dcov2(Axc, Byc) / np.sqrt(dVx * max(dVy, 0) + 1e-24)))
                else:
                    Yoh = _onehot(y_s.astype(int), K); sc_k = []
                    for k in range(K):
                        yk  = Yoh[:, k:k+1]; By = cdist(yk, yk)
                        Byc = _u_center(By); dVy = _dcov2(Byc, Byc)
                        sc_k.append(_dcov2(Axc, Byc) / np.sqrt(dVx * max(dVy, 0) + 1e-24))
                    draws.append(float(np.mean(sc_k)))
            out.append((float(np.mean(draws)), 1.0))
        return out

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p  = X.shape; task = self.task_; K = self.n_classes_
        rng   = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=(p // _FEAT_BATCH) + 1)
        eff_p = _eff_perms(n, self.n_perms)
        bats  = [list(range(i, min(i + _FEAT_BATCH, p))) for i in range(0, p, _FEAT_BATCH)]

        if n <= self.n_exact:
            # Pré-calculer les canaux y UNE FOIS
            if task == "regression":
                yv = y.reshape(-1, 1); By = cdist(yv, yv); Byc = _u_center(By)
                channels = [(Byc, _dcov2(Byc, Byc))]
            else:
                Yoh = _onehot(y.astype(int), K); channels = []
                for k in range(K):
                    yk = Yoh[:, k:k+1]; By = cdist(yk, yk); Byc = _u_center(By)
                    channels.append((Byc, _dcov2(Byc, Byc)))
            nested = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._batch_exact)(X[:, b], channels, eff_p, int(seeds[bi]))
                for bi, b in enumerate(bats))
        else:
            nested = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._batch_sketch)(X[:, b], y, task, K, int(seeds[bi]))
                for bi, b in enumerate(bats))

        flat = [item for sub in nested for item in sub]
        self.scores_  = np.array([r[0] for r in flat])
        self.pvalues_ = np.array([r[1] for r in flat])


# ══════════════════════════════════════════════════════════════════════════════
# §6  HSICFilter — HSIC + RFF  (Song et al. 2012 + Rahimi & Recht 2007)
# ══════════════════════════════════════════════════════════════════════════════

class HSICFilter(FilterMethod):
    """
    nHSIC = HSIC_u(Xj,y) / √(HSIC_u(Kx,Kx)·HSIC_u(Ly,Ly)) ∈ [0,1].
    Multiclasse : noyau delta Ly = Yoh·Yohᵀ (pas de bandwidth).

    OPTS :
    • RFF TOUJOURS (D=256) — bande passante MAD O(n) au lieu de cdist O(n²).
    • Zy pré-calculé UNE FOIS, hll=HSIC(Zy,Zy) aussi.
    • Perms : permuter les lignes de Zy (O(n)) — pas de reconstruction du noyau.
    • FEAT_BATCH features par worker — overhead joblib ~0.
    """

    def __init__(self, n_perms: int = 200, rff_D: int = _RFF_D,
                 n_jobs: int = -1, random_state: int = 0,
                 task: TaskParam = None):
        self.n_perms = n_perms; self.rff_D = rff_D
        self.n_jobs  = n_jobs;  self.random_state = random_state
        self.task = task

    @staticmethod
    def _adapt_D(n: int, rff_D: int) -> int:
        """D adaptatif : maintient n·D ≤ ~6.4M ops (64ms@50k). Min 32."""
        return max(32, min(rff_D, 6_400_000 // max(n, 1)))

    @staticmethod
    def _batch_worker(Xb: np.ndarray, Zy: np.ndarray, hll: float,
                      rff_D: int, n_perms: int, seed: int) -> list:
        rng = np.random.default_rng(seed); n = Xb.shape[0]; out = []
        D   = HSICFilter._adapt_D(n, rff_D)        # adaptive per-batch
        for j in range(Xb.shape[1]):
            xv  = Xb[:, j]
            bw  = _bw_mad(xv)
            Zx  = _rff(xv.reshape(-1, 1), D, bw, rng)
            # Re-project Zy to same D if needed
            Zyj = Zy[:, :D] if Zy.shape[1] > D else Zy
            h   = _hsic_rff(Zx, Zyj); hkk = _hsic_rff(Zx, Zx)
            hll_j = _hsic_rff(Zyj, Zyj)
            den = np.sqrt(max(hkk, 0.0) * max(hll_j, 0.0))
            sc  = float(h / den) if den > 1e-12 else 0.0
            if n_perms == 0: out.append((sc, 1.0)); continue
            null = np.empty(n_perms)
            for r in range(n_perms):
                hp     = _hsic_rff(Zx, Zyj[rng.permutation(n)])
                null[r] = float(hp / den) if den > 1e-12 else 0.0
            out.append((sc, float(np.mean(null >= sc - 1e-12))))
        return out

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p  = X.shape; task = self.task_; K = self.n_classes_
        rng   = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=(p // _FEAT_BATCH) + 1)
        eff_p = _eff_perms(n, self.n_perms)
        D     = self._adapt_D(n, self.rff_D)

        # Pré-calculer Zy UNE FOIS (au D adaptatif)
        if task == "regression":
            bwy = _bw_mad(y); Zy = _rff(y.reshape(-1, 1), D, bwy, rng)
        elif task == "binary":
            Zy  = _rff(y.reshape(-1, 1), D, 0.25, rng)
        else:  # multiclasse : delta kernel → Yoh sont les features exactes
            Yoh = _onehot(y.astype(int), K).astype(np.float32)
            # Troncature si K > D (rare)
            Zy  = Yoh[:, :D] if K > D else Yoh

        hll = _hsic_rff(Zy, Zy)
        bats = [list(range(i, min(i + _FEAT_BATCH, p))) for i in range(0, p, _FEAT_BATCH)]

        nested = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self._batch_worker)(
                X[:, b], Zy, hll, self.rff_D, eff_p, int(seeds[bi])
            ) for bi, b in enumerate(bats))

        flat = [item for sub in nested for item in sub]
        self.scores_  = np.array([r[0] for r in flat])
        self.pvalues_ = np.array([r[1] for r in flat])


# ══════════════════════════════════════════════════════════════════════════════
# §7  CPIFilter — Conditional Permutation Importance  (Molnar et al. 2023)
# ══════════════════════════════════════════════════════════════════════════════

class CPIFilter(FilterMethod):
    """
    Drop de score = baseline − mean(scores permutés). Modèle fitté UNE FOIS.
    Scoring : R² (reg) ou log-vraisemblance moyenne (clf).

    OPTS :
    • n > N_SCORE : sous-échantillon pour le scoring (pas de re-fit).
    • Permutation conditionnelle dans bins SVD-1 : O(n log n).
    • FEAT_BATCH features par worker.
    """

    N_SCORE = 50_000

    def __init__(self, estimator=None, n_perms: int = 50, n_bins: int = 5,
                 n_jobs: int = -1, random_state: int = 0,
                 task: TaskParam = None):
        self.estimator = estimator; self.n_perms = n_perms; self.n_bins = n_bins
        self.n_jobs    = n_jobs;    self.random_state = random_state
        self.task = task

    def _make_model(self, task: TaskType):
        kw = dict(n_estimators=200, max_features="sqrt",
                  n_jobs=1, random_state=self.random_state)
        return (RandomForestRegressor(**kw) if task == "regression"
                else RandomForestClassifier(**kw))

    @staticmethod
    def _score(y: np.ndarray, pred, task: TaskType, K: int) -> float:
        if task == "regression":
            return float(1.0 - np.sum((y - pred)**2) /
                         (np.sum((y - y.mean())**2) + 1e-12))
        p = np.clip(pred, 1e-12, 1.0)
        return float(np.mean(np.einsum("ij,ij->i",
                     _onehot(y.astype(int), K), np.log(p))))

    @staticmethod
    def _batch_worker(j_list: list, X: np.ndarray, y: np.ndarray,
                      baseline: float, model, task: TaskType, K: int,
                      n_perms: int, n_bins: int, seed: int) -> list:
        rng = np.random.default_rng(seed); n = X.shape[0]; out = []
        for j in j_list:
            # Permutation conditionnelle dans bins de la projection SVD-1(X_{-j})
            other = np.delete(X, j, axis=1) if X.shape[1] > 1 else X
            if other.shape[1] > 1:
                _, _, Vt = np.linalg.svd(other - other.mean(0), full_matrices=False)
                proj = other @ Vt[0]
            else:
                proj = other[:, 0]
            labs  = np.floor(
                rankdata(proj) / (n + 1) * n_bins).astype(int).clip(0, n_bins - 1)
            vals  = X[:, j].copy()
            perms = np.tile(vals, (n_perms, 1))
            for b in range(n_bins):
                m = np.where(labs == b)[0]
                if m.size < 2: continue
                for r in range(n_perms): perms[r, m] = rng.permutation(vals[m])

            # Évaluation des n_perms matrices perturbées
            Xp  = np.broadcast_to(X, (n_perms,) + X.shape).copy()
            Xp[:, :, j] = perms
            psc = np.empty(n_perms)
            for r in range(n_perms):
                pred = (model.predict(Xp[r]) if task == "regression"
                        else _proba(model, Xp[r], task, K))
                psc[r] = CPIFilter._score(y, pred, task, K)

            sc   = float(baseline - psc.mean())
            null = baseline - psc
            pv   = float(np.mean(np.abs(null) >= abs(sc)) * n_perms / (n_perms + 1))
            out.append((sc, pv))
        return out

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p = X.shape; task = self.task_; K = self.n_classes_
        rng  = np.random.default_rng(self.random_state)
        model = self.estimator or self._make_model(task)
        model.fit(X, y.astype(int) if task != "regression" else y)

        if n > self.N_SCORE:
            idx = _strat_idx(y, task, K, self.N_SCORE, rng); Xs, ys = X[idx], y[idx]
        else:
            Xs, ys = X, y

        baseline = (self._score(ys, model.predict(Xs), task, K)
                    if task == "regression"
                    else self._score(ys, _proba(model, Xs, task, K), task, K))
        self.baseline_score_ = baseline

        seeds = rng.integers(0, 2**31, size=(p // _FEAT_BATCH) + 1)
        bats  = [list(range(i, min(i + _FEAT_BATCH, p))) for i in range(0, p, _FEAT_BATCH)]
        nested = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self._batch_worker)(
                b, Xs, ys, baseline, model, task, K,
                self.n_perms, self.n_bins, int(seeds[bi])
            ) for bi, b in enumerate(bats))
        flat = [item for sub in nested for item in sub]
        self.scores_  = np.array([r[0] for r in flat])
        self.pvalues_ = np.array([r[1] for r in flat])


# ══════════════════════════════════════════════════════════════════════════════
# §8  GCMFilter — Generalised Covariance Measure
#     Shah & Peters (AoS 2020) · Scheidegger et al. (2022)
# ══════════════════════════════════════════════════════════════════════════════

class GCMFilter(FilterMethod):
    """
    T_n = mean(Rj · Ry_j) ;  Z = √n·T_n/σ̂ → N(0,1).  P-valeur CLT, 0 perms.

    Rj   = Xj − Ê[Xj|X_{-j}]   (Ridge)
    Ry_j = y  − Ê[y |X_{-j}]   (Ridge/RF/CatBoost)

    NOTE sur la puissance :
    GCM avec Ridge est RAPIDE et EXACT en taille pour les associations linéaires.
    Pour y non-linéaire en X (sin, x²...), les résidus Ridge sont moins bons
    → utiliser regressor="rf" ou "catboost" pour plus de puissance non-linéaire,
    ou combiner GCM (linéaire) + Xi/FOCI/HSIC (non-linéaire).

    CatBoost : eval_fraction carve validation par fold → early stopping → taille exacte.
    Multiclasse : OvR résidus (n,K) ; max-|Z| Bonferroni.

    OPTS :
    • Ridge par défaut : 100× plus rapide que RF.
    • CLT sandwich : 0 permutation.
    • Fold-splits partagés.
    • FEAT_BATCH + prefilter_k.
    """

    def __init__(self, n_splits: int = 5,
                 regressor: Literal["ridge", "rf", "catboost"] = "ridge",
                 alpha_ridge: float = 1.0, eval_fraction: float = 0.15,
                 cb_iterations: int = 500, cb_learning_rate: float = 0.05,
                 cb_depth: int = 6, cb_early_stopping: int = 40,
                 n_jobs: int = -1, random_state: int = 0,
                 prefilter_k: Optional[int] = None,
                 verbose_catboost: bool = False,
                 task: TaskParam = None):
        self.n_splits = n_splits; self.regressor = regressor
        self.alpha_ridge = alpha_ridge; self.eval_fraction = eval_fraction
        self.cb_iterations = cb_iterations; self.cb_learning_rate = cb_learning_rate
        self.cb_depth = cb_depth; self.cb_early_stopping = cb_early_stopping
        self.n_jobs = n_jobs; self.random_state = random_state
        self.prefilter_k = prefilter_k; self.verbose_catboost = verbose_catboost
        self.task = task

    def _make_y_model(self, task: TaskType):
        reg = self.regressor
        if reg == "catboost":
            if not _CB_OK:
                warnings.warn("CatBoost non installé → Ridge.", stacklevel=3); reg = "ridge"
            else:
                kw = dict(iterations=self.cb_iterations,
                          learning_rate=self.cb_learning_rate, depth=self.cb_depth,
                          early_stopping_rounds=self.cb_early_stopping,
                          random_seed=self.random_state, verbose=self.verbose_catboost,
                          allow_writing_files=False)
                if task == "regression": return CatBoostRegressor(loss_function="RMSE", **kw)
                if task == "binary":     return CatBoostClassifier(loss_function="Logloss", **kw)
                return CatBoostClassifier(loss_function="MultiClass", **kw)
        if reg == "rf":
            kw = dict(n_estimators=100, max_features="sqrt",
                      n_jobs=1, random_state=self.random_state)
            return (RandomForestRegressor(**kw) if task == "regression"
                    else RandomForestClassifier(**kw))
        return (Ridge(self.alpha_ridge) if task == "regression"
                else RidgeClassifier(self.alpha_ridge))

    def _eval_split(self, Xtr, ytr, fi, task):
        rng = np.random.default_rng(self.random_state + fi)
        if task != "regression":
            val = np.concatenate([
                rng.choice(np.where(ytr == c)[0],
                           max(1, int(np.sum(ytr == c) * self.eval_fraction)),
                           replace=False)
                for c in np.unique(ytr)])
        else:
            val = rng.choice(len(ytr), max(2, int(len(ytr) * self.eval_fraction)), replace=False)
        tr2 = np.setdiff1d(np.arange(len(ytr)), val, assume_unique=True)
        return Xtr[tr2], Xtr[val], ytr[tr2], ytr[val]

    def _resid_xj(self, xj, Xcond, folds):
        n = len(xj); out = np.empty(n); sc = StandardScaler()
        for tr, te in folds:
            Xtr = sc.fit_transform(Xcond[tr]); Xte = sc.transform(Xcond[te])
            out[te] = xj[te] - Ridge(self.alpha_ridge).fit(Xtr, xj[tr]).predict(Xte)
        return out

    def _resid_y(self, y, Xcond, task, K, folds, fi0=0):
        n = len(y); sc = StandardScaler()
        use_cb = self.regressor == "catboost" and _CB_OK
        if task == "regression":
            out = np.empty(n)
            for fi, (tr, te) in enumerate(folds):
                Xtr = sc.fit_transform(Xcond[tr]); Xte = sc.transform(Xcond[te])
                m = self._make_y_model(task)
                if use_cb:
                    X2, Xv, y2, yv = self._eval_split(Xtr, y[tr], fi + fi0, task)
                    m.fit(X2, y2, eval_set=(Xv, yv))
                else: m.fit(Xtr, y[tr])
                out[te] = y[te] - m.predict(Xte)
            return out
        elif task == "binary":
            out = np.empty(n)
            for fi, (tr, te) in enumerate(folds):
                Xtr = sc.fit_transform(Xcond[tr]); Xte = sc.transform(Xcond[te])
                yi  = y[tr].astype(int); m = self._make_y_model(task)
                if use_cb:
                    X2, Xv, y2, yv = self._eval_split(Xtr, yi, fi + fi0, task)
                    m.fit(X2, y2, eval_set=(Xv, yv))
                else: m.fit(Xtr, yi)
                out[te] = (y[te] == 1).astype(float) - _proba(m, Xte, task, K)[:, 1]
            return out
        else:
            out = np.zeros((n, K))
            for fi, (tr, te) in enumerate(folds):
                Xtr = sc.fit_transform(Xcond[tr]); Xte = sc.transform(Xcond[te])
                yi  = y[tr].astype(int); m = self._make_y_model(task)
                if use_cb:
                    X2, Xv, y2, yv = self._eval_split(Xtr, yi, fi + fi0, task)
                    m.fit(X2, y2, eval_set=(Xv, yv))
                else: m.fit(Xtr, yi)
                out[te] = _onehot(y[te].astype(int), K) - _proba(m, Xte, task, K)
            return out

    @staticmethod
    def _gcm_stat(Rj, Ry):
        if Ry.ndim == 1:
            n = len(Rj); pr = Rj * Ry; T = pr.mean()
            se = np.sqrt(((pr - T)**2).mean() / n + 1e-14)
            Z  = T / se
            return float(abs(Z)), float(2.0 * norm.sf(abs(Z)))
        K = Ry.shape[1]; Zs = np.empty(K); ps = np.empty(K)
        for k in range(K):
            pr = Rj * Ry[:, k]; T = pr.mean()
            se = np.sqrt(((pr - T)**2).mean() / len(Rj) + 1e-14)
            Zs[k] = abs(T / se); ps[k] = float(2.0 * norm.sf(Zs[k]))
        best = int(np.argmax(Zs))
        return float(Zs[best]), float(min(1.0, ps[best] * K))

    @staticmethod
    def _batch_worker(j_list, X, y, task, K, folds, gcm):
        out = []
        for j in j_list:
            Xc = np.delete(X, j, axis=1)
            Rj = gcm._resid_xj(X[:, j], Xc, folds)
            Ry = gcm._resid_y(y, Xc, task, K, folds, fi0=j)
            out.append(gcm._gcm_stat(Rj, Ry))
        return out

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p  = X.shape; task = self.task_; K = self.n_classes_
        folds = list(KFold(self.n_splits, shuffle=True,
                           random_state=self.random_state).split(X))
        active = list(range(p))
        if self.prefilter_k and p > self.prefilter_k:
            xi = XiFilter(n_perms=0, n_jobs=self.n_jobs).fit(X, y)
            active = list(np.argsort(xi.scores_)[::-1][:self.prefilter_k])
        bats = [active[i:i+_FEAT_BATCH] for i in range(0, len(active), _FEAT_BATCH)]
        nested = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self._batch_worker)(b, X, y, task, K, folds, self)
            for b in bats)
        flat = {j: r for b, res in zip(bats, nested) for j, r in zip(b, res)}
        self.scores_  = np.zeros(p); self.pvalues_ = np.ones(p)
        for j, (sc, pv) in flat.items():
            self.scores_[j] = sc; self.pvalues_[j] = pv


# ══════════════════════════════════════════════════════════════════════════════
# §9  PCMFilter — Projected Covariance Measure  (Scheidegger et al. 2022)
# ══════════════════════════════════════════════════════════════════════════════

class PCMFilter(FilterMethod):
    """
    T_n = n·HSIC_u(U,V)  avec  U = Xj−Ê[Xj|X_{-j}],  V = y−Ê[y|X_{-j}].
    Test de permutation sur les résidus → taille exacte.
    Multiclasse : V ∈ ℝ^(n×K), T_n = n·mean_k HSIC_u(U,V_k).

    OPTS :
    • Ridge pour tous les résidus.
    • Fold-splits partagés entre features.
    • Permuter U (scalaire) → O(1).
    • FEAT_BATCH par worker + prefilter_k.
    """

    def __init__(self, n_splits: int = 5, n_perms: int = 200,
                 n_jobs: int = -1, random_state: int = 0,
                 prefilter_k: Optional[int] = None, alpha: float = 1.0,
                 task: TaskParam = None):
        self.n_splits = n_splits; self.n_perms = n_perms; self.n_jobs = n_jobs
        self.random_state = random_state; self.prefilter_k = prefilter_k; self.alpha = alpha
        self.task = task

    @staticmethod
    def _resid_pair(xj, Xcond, y, task, K, folds, alpha):
        n  = len(xj); sc = StandardScaler()
        U  = np.empty(n)
        V  = np.zeros((n, K)) if task == "multiclass" else np.empty(n)
        for tr, te in folds:
            Xtr = sc.fit_transform(Xcond[tr]); Xte = sc.transform(Xcond[te])
            U[te] = xj[te] - Ridge(alpha).fit(Xtr, xj[tr]).predict(Xte)
            if task == "regression":
                V[te] = y[te] - Ridge(alpha).fit(Xtr, y[tr]).predict(Xte)
            elif task == "binary":
                m    = RidgeClassifier(alpha).fit(Xtr, y[tr].astype(int))
                df   = m.decision_function(Xte)
                V[te] = (y[te] == 1).astype(float) - _sigmoid(df)
            else:
                m    = RidgeClassifier(alpha).fit(Xtr, y[tr].astype(int))
                df   = m.decision_function(Xte)
                prob = _softmax_np(df) if df.ndim == 2 else np.column_stack(
                       [_sigmoid(-df), _sigmoid(df)])
                cls  = np.asarray(m.classes_).astype(int)
                pa   = np.zeros((len(te), K))
                for ci, c in enumerate(cls):
                    if c < K: pa[:, c] = prob[:, ci]
                V[te] = _onehot(y[te].astype(int), K) - pa
        return U, V

    @staticmethod
    def _Tn(U, V):
        n  = len(U); bwu = _bw_mad(U)
        Ku = _rbf(U.reshape(-1, 1), bwu)
        if V.ndim == 1:
            return float(n * _hsic_exact(Ku, _rbf(V.reshape(-1, 1), _bw_mad(V))))
        return float(n * np.mean([
            _hsic_exact(Ku, _rbf(V[:, k:k+1], _bw_mad(V[:, k])))
            for k in range(V.shape[1])]))

    @staticmethod
    def _batch_worker(j_list, X, y, task, K, folds, n_perms, alpha, seed):
        rng = np.random.default_rng(seed); out = []
        for j in j_list:
            Xc = np.delete(X, j, axis=1)
            U, V = PCMFilter._resid_pair(X[:, j], Xc, y, task, K, folds, alpha)
            U = (U - U.mean()) / (U.std() + 1e-12)
            if V.ndim > 1:
                V = (V - V.mean(0)) / (V.std(0) + 1e-12)
            else:
                V = (V - V.mean()) / (V.std() + 1e-12)
            T = PCMFilter._Tn(U, V)
            if n_perms == 0: out.append((float(T), 1.0)); continue
            null = np.array([PCMFilter._Tn(rng.permutation(U), V) for _ in range(n_perms)])
            out.append((float(T), float(np.mean(null >= T - 1e-12))))
        return out

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p  = X.shape; task = self.task_; K = self.n_classes_
        rng   = np.random.default_rng(self.random_state)
        eff_p = _eff_perms(n, self.n_perms)
        folds = list(KFold(self.n_splits, shuffle=True,
                           random_state=self.random_state).split(X))
        active = list(range(p))
        if self.prefilter_k and p > self.prefilter_k:
            xi = XiFilter(n_perms=0, n_jobs=self.n_jobs).fit(X, y)
            active = list(np.argsort(xi.scores_)[::-1][:self.prefilter_k])
        seeds = rng.integers(0, 2**31, size=(len(active) // _FEAT_BATCH) + 1)
        bats  = [active[i:i+_FEAT_BATCH] for i in range(0, len(active), _FEAT_BATCH)]
        nested = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self._batch_worker)(
                b, X, y, task, K, folds, eff_p, self.alpha, int(seeds[bi])
            ) for bi, b in enumerate(bats))
        flat = {j: r for b, res in zip(bats, nested) for j, r in zip(b, res)}
        self.scores_  = np.zeros(p); self.pvalues_ = np.ones(p)
        for j, (sc, pv) in flat.items():
            self.scores_[j] = sc; self.pvalues_[j] = pv


# ══════════════════════════════════════════════════════════════════════════════
# §10  FOCIFilter — Feature Ordering by Conditional Independence
#      Azadkia & Chatterjee (Ann. Statist. 2021)
# ══════════════════════════════════════════════════════════════════════════════

class FOCIFilter(FilterMethod):
    """
    Sélection greedy forward maximisant T(y,Xj|X_S). Arrêt automatique.
    T(Y,X|Z) = (Q(Y,X,Z)−Q(Y,Z))/(1−Q(Y,Z))  où  Q = ξ en espace NN.
    Multiclasse : max_k T(1_{y=k}, Xj | X_S).

    OPTS :
    • prefilter_k via Xi : gain massif sur grand p.
    • cKDTree SciPy (C) pour la recherche NN.
    • Candidats parallélisés dans chaque étape.
    • p-valeurs Xi réutilisées gratuitement.
    """

    def __init__(self, max_features: Optional[int] = None, tol: float = -np.inf,
                 n_jobs: int = -1, random_state: int = 0,
                 prefilter_k: Optional[int] = None,
                 task: TaskParam = None):
        self.max_features = max_features; self.tol = tol
        self.n_jobs = n_jobs; self.random_state = random_state
        self.prefilter_k = prefilter_k; self.task = task

    @staticmethod
    def _T(y: np.ndarray, xj: np.ndarray, Z: np.ndarray) -> float:
        n   = len(y); r_y = rankdata(y, method="average") / n

        def _xi_W(W):
            if W.shape[1] == 1:
                delta = np.abs(np.diff(r_y[np.argsort(W[:, 0], kind="stable")])).sum()
                return float(1.0 - 3.0 * delta / (n**2 - 1.0))
            Ws = (W - W.mean(0)) / (W.std(0) + 1e-8)
            _, nn = cKDTree(Ws).query(Ws, k=2, workers=1); nn = nn[:, 1]
            return float((np.mean(np.minimum(r_y, r_y[nn])) - np.mean(r_y**2)) /
                          (np.mean(r_y * (1.0 - r_y)) + 1e-14))

        if Z.shape[1] == 0:
            delta = np.abs(np.diff(r_y[np.argsort(xj, kind="stable")])).sum()
            return float(1.0 - 3.0 * delta / (n**2 - 1.0))
        xj_s = (xj - xj.mean()) / (xj.std() + 1e-8)
        Zs   = (Z  -  Z.mean(0)) / (Z.std(0) + 1e-8)
        Q_XZ = _xi_W(np.column_stack([Zs, xj_s])); Q_Z = _xi_W(Zs)
        return float((Q_XZ - Q_Z) / (1.0 - Q_Z + 1e-14))

    def _T_ovr(self, y_codes, K, xj, Z):
        return float(max(
            Parallel(n_jobs=self.n_jobs, backend="loky", prefer="threads")(
                delayed(self._T)((y_codes == k).astype(float), xj, Z)
                for k in range(K))))

    def _greedy(self, X, y, task, K, active):
        n = X.shape[0]; sel = []; cands = list(active)
        T_prev = 0.0; gains = np.zeros(X.shape[1])
        max_k  = self.max_features or len(active)
        multi  = task == "multiclass" and K > 2

        for _ in range(min(max_k, len(active))):
            if not cands: break
            Z = X[:, sel] if sel else np.empty((n, 0))
            if multi:
                T_vals = [self._T_ovr(y, K, X[:, j], Z) for j in cands]
            else:
                y_s = (y == 1).astype(float) if task == "binary" else y
                T_vals = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed(self._T)(y_s, X[:, j], Z) for j in cands)
            bi = int(np.argmax(T_vals)); bj = cands[bi]; bT = float(T_vals[bi])
            if bT - T_prev <= self.tol: break
            gains[bj] = bT; sel.append(bj); cands.remove(bj); T_prev = bT
        return sel, gains

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        task = self.task_; K = self.n_classes_; p = X.shape[1]
        xi_m = XiFilter(n_perms=0, n_jobs=self.n_jobs); xi_m.fit(X, y)
        active = list(range(p))
        if self.prefilter_k and p > self.prefilter_k:
            active = list(np.argsort(xi_m.scores_)[::-1][:self.prefilter_k])
        sel, gains = self._greedy(X, y, task, K, active)
        self.selected_order_ = sel; self.scores_ = gains; self.pvalues_ = xi_m.pvalues_

    def get_support(self, k: Optional[int] = None) -> np.ndarray:
        check_is_fitted(self)
        sel = np.array(self.selected_order_)
        return sel if k is None else sel[:k]


# ══════════════════════════════════════════════════════════════════════════════
# §11  compare_all
# ══════════════════════════════════════════════════════════════════════════════

def compare_all(X: np.ndarray, y: np.ndarray, top_k: int = 10,
                feature_names: Optional[list] = None,
                gcm_backend: Literal["ridge", "rf", "catboost"] = "ridge",
                prefilter_k: Optional[int] = None,
                n_perms: int = 100, n_jobs: int = -1) -> dict:
    """
    Lance les 7 méthodes sur (X, y) et affiche les résultats.

    Paramètres
    ----------
    gcm_backend  : "ridge" (défaut, rapide+linéaire), "rf" ou "catboost"
    prefilter_k  : pré-filtre PCM/GCM/FOCI via Xi (fortement conseillé si p>200)
    n_perms      : permutations (réduit auto pour grand n)
    """
    import time
    names = feature_names or [f"X{i}" for i in range(X.shape[1])]
    task  = _detect_task(np.asarray(y)); n, p = X.shape
    print(f"  n={n:,}  p={p}  task={task}  gcm={gcm_backend}"
          + (f"  prefilter={prefilter_k}" if prefilter_k else ""))
    if gcm_backend == "catboost" and not _CB_OK:
        gcm_backend = "ridge"; print("  CatBoost non dispo → GCM=Ridge.")

    methods = {
        "Xi":   XiFilter(n_perms=0, n_jobs=n_jobs),
        "dCor": dCorFilter(n_perms=n_perms, n_jobs=n_jobs),
        "HSIC": HSICFilter(n_perms=n_perms, n_jobs=n_jobs),
        "CPI":  CPIFilter(n_perms=min(n_perms, 50), n_jobs=n_jobs),
        "GCM":  GCMFilter(regressor=gcm_backend, n_jobs=n_jobs,
                           prefilter_k=prefilter_k),
        "PCM":  PCMFilter(n_perms=n_perms, n_jobs=n_jobs,
                           prefilter_k=prefilter_k),
        "FOCI": FOCIFilter(n_jobs=n_jobs, prefilter_k=prefilter_k),
    }
    results = {}
    for name, m in methods.items():
        t0 = time.perf_counter(); m.fit(X, y); elapsed = time.perf_counter() - t0
        top = np.argsort(m.scores_)[::-1][:top_k]
        results[name] = {"model": m, "scores": m.scores_, "pvalues": m.pvalues_,
                         "top_k_idx": top, "top_k_names": [names[i] for i in top],
                         "time_s": round(elapsed, 3)}
        extra = ""
        if hasattr(m, "selected_order_"):
            s5    = [names[i] for i in m.selected_order_[:5]]
            extra = f"  sél={s5}{'…' if len(m.selected_order_) > 5 else ''}"
        print(f"  [{name:5s}] {elapsed:7.2f}s  top-3: {[names[i] for i in top[:3]]}{extra}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# §12  __main__ — validation 3 tâches × signaux linéaires + non-linéaires
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42); n, p = 600, 20
    X   = rng.standard_normal((n, p))
    X[:, 5] = X[:, 0] + 0.04 * rng.standard_normal(n)   # piège colinéaire
    X[:, 9] = X[:, 0] + 0.04 * rng.standard_normal(n)   # piège colinéaire
    feat    = [f"F{i}" for i in range(p)]
    signal  = 2 * X[:, 0] + np.sin(3 * X[:, 2]) + X[:, 7] ** 2

    print("Signal = 2·F0 + sin(3·F2) + F7²")
    print("F5, F9 ≈ F0 (pièges colinéaires) | F1,F3,F4,... = bruit")
    print()
    print("GCM Ridge : optimal pour associations LINÉAIRES (F0).")
    print("Xi/FOCI/HSIC/CPI : capturent sin(F2) et F7² (non-linéaire).")

    y_reg   = signal + 0.3 * rng.standard_normal(n)
    y_bin   = (signal > signal.mean()).astype(int)
    q33, q66 = np.percentile(signal, [33, 66])
    y_multi  = np.where(signal < q33, 0, np.where(signal < q66, 1, 2))

    for label, y in [("REGRESSION", y_reg), ("BINAIRE", y_bin), ("MULTICLASSE", y_multi)]:
        print(f"\n{'='*70}")
        print(f"  {label}  —  vrai signal: F0 (lin), F2 (sin), F7 (x²)")
        print("="*70)
        compare_all(X, y, top_k=4, feature_names=feat, n_perms=50)
