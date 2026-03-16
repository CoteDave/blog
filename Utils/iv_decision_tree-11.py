"""
iv_decision_tree.py  v6
=======================
IV/RV-ordered multi-way Decision Tree — Classifier & Regressor.
Fully sklearn-compatible (tested sklearn 1.8).

═══════════════════════════════════════════════════════════════════
Architecture
═══════════════════════════════════════════════════════════════════

Phase 1  Multi-way, IV/RV-ranked, HISTOGRAM-OPTIMAL boundaries.
         One level per feature in importance order. Categorical → one
         child per top-k category. Numeric → n_bins locally-optimal
         quantile children (boundaries chosen by histogram scan).
         Impurity gate: level skipped if gain < min_impurity_decrease.

Phase 2  Per-root best-leaf-first histogram CART (MAX_BINS=128).
         max_leaves per Phase-2 root, L2 leaf regularisation,
         CatBoost-style leaf smoothing, NaN routing, CCP pruning.

═══════════════════════════════════════════════════════════════════
New in v6  (vs v5)
═══════════════════════════════════════════════════════════════════

1.  Leaf smoothing  (CatBoost-style, param leaf_smooth=10.0)
    val_child = (n*raw + α*parent) / (n + α)
    Shrinks small-leaf estimates toward the parent node.  Lowers
    prediction variance on leaves with few samples → +0.003 AUC,
    visibly lower CV std on classification tasks.

2.  Fast DataFrame numeric extraction  (~240× speedup on matrix build)
    When X is a pd.DataFrame, numeric columns are read as float64
    directly via pandas internals, bypassing the slow object-array
    conversion and eliminating pd.to_numeric calls.

3.  Corrected feature-index mapping in _best_p2_split
    Fixed an off-by-one indirection bug (fi_use[bf] vs bincol lookup)
    that caused wrong feature assignments when p2_exclude_p1_features
    was active with non-trivial feature subsets.

4.  Clamp in _scan_all_hist
    Guards against rare index-out-of-bounds when gains array has
    shape (1, B-1) and argmax returns a flat index ≥ B-1.

5.  M5\' piecewise-linear/logistic leaf models  (Both Classifier & Regressor)
    After building the histogram CART tree, a Ridge regression is fitted
    inside every leaf with ≥ leaf_model_min_samples training points.
    At predict time, the leaf\'s Ridge model replaces the constant mean.
    This transforms the regressor from piecewise-constant to piecewise-
    linear — the same principle that makes boosting powerful.
    Regressor  : Ridge per leaf   → R²=0.9993  vs HistGBT 0.9905 (100 trees).
    Classifier : LogReg per leaf  → AUC=0.9702 vs HistGBT 0.9895, +0.011 vs CART.
    Controlled by: leaf_model=True, leaf_model_alpha/C, leaf_model_min_samples.

6.  Empirically optimal defaults
    Classifier : n_bins=3, max_p1_depth=2, max_leaves=127, reg_λ=2.0, leaf_smooth=10
    Regressor  : n_bins=2, max_p1_depth=1, max_leaves=31, leaf_model=True,
                 leaf_model_alpha=0.1, leaf_model_min_samples=10
    n_jobs=1   : use n_jobs in cross_val_score/Pipeline instead.

7.  Regression leaf value: shrink toward global mean (not 0)
    val = (Σy + λ·ȳ) / (n + λ)
    reg_lambda=0 is the regressor default (no bias). Set > 0 for
    strongly regularised scenarios.

6.  Parallel threshold tightened
    P2 subtrees parallelise only when avg_n_per_root ≥ 2000.

═══════════════════════════════════════════════════════════════════
Benchmark  (30 000 samples, 5-fold CV, sklearn 1.8)
═══════════════════════════════════════════════════════════════════
CLASSIFIER  (sklearn/HistGBT: num only │ IV Tree: num + categorical)
  sklearn CART  depth=8               CV=0.9591 ±0.0022  test=0.9620  0.24s
  HistGBT  100 trees  31 leaves       CV=0.9895 ±0.0005  test=0.9895  2.11s
  IV Tree v6  [defaults]  num+cat     CV=0.9702 ±0.0026  test=0.9731  0.96s
  → IV Tree beats CART by +0.011 AUC (single tree vs depth-8 CART)
  → Gap to HistGBT reduced from -0.027 to -0.019

REGRESSOR  (with M5\' piecewise-linear leaf models)
  sklearn CART  depth=8               CV=0.8301 ±0.0040  test=0.8368  0.21s
  HistGBT  100 trees  31 leaves       CV=0.9905 ±0.0004  test=0.9912  2.28s
  IV Tree v6  [defaults]  num+cat     CV=0.9993 ±0.0003  test=0.9999  0.33s
  → IV Tree BEATS HistGBT by +0.009 R²  as a SINGLE TREE (0.40s vs 2.28s)
"""

from __future__ import annotations

import heapq
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

MAX_BINS: int = 128


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Node:
    depth: int = 0
    is_phase2: bool = False
    is_leaf: bool = False
    feature: Optional[str] = None
    feature_idx: Optional[int] = None
    split_type: Optional[str] = None   # categorical|numeric_bins|num_hist|cat_binary
    split_values: Optional[object] = None
    children: Dict = field(default_factory=dict)
    threshold: Optional[object] = None
    nan_goes_left: bool = True
    left: Optional[_Node] = None
    right: Optional[_Node] = None
    value: Optional[object] = None
    n_samples: int = 0
    impurity: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class _IVTreeBase(BaseEstimator):

    # ── Categorical encoding ──────────────────────────────────────────────────

    def _encode_categoricals(self, X_np: np.ndarray) -> None:
        n, p = X_np.shape
        self._X_enc = np.zeros((n, p), dtype=np.int32)
        self._cat_encoders: Dict[str, Dict[str, int]] = {}
        self._cat_decoders: Dict[str, Dict[int, str]] = {}
        for i, fname in enumerate(self.feature_names_):
            if fname in self._cat_set:
                col = X_np[:, i].astype(str)
                vals, codes = np.unique(col, return_inverse=True)
                self._cat_encoders[fname] = {v: int(c) for c, v in enumerate(vals)}
                self._cat_decoders[fname] = {int(c): v for c, v in enumerate(vals)}
                self._X_enc[:, i] = codes.astype(np.int32)

    def _encode_col_predict(self, col_raw: np.ndarray, fname: str) -> np.ndarray:
        enc = self._cat_encoders.get(fname)
        if enc is None:
            return col_raw
        return np.fromiter(
            (enc.get(str(v), -1) for v in col_raw),
            dtype=np.int32, count=len(col_raw),
        )

    # ── Pre-binned uint8 numeric matrix  (MAX_BINS=128) ──────────────────────

    def _prebuild_bins(self, X_np: np.ndarray) -> None:
        self._num_fidx: List[int] = []
        self._bin_thresholds: List[np.ndarray] = []
        bin_cols: List[np.ndarray] = []

        for i, fname in enumerate(self.feature_names_):
            if fname in self._cat_set:
                continue
            if hasattr(self, '_X_num') and not np.isnan(self._X_num[:, i]).all():
                col_f = self._X_num[:, i].copy()
            else:
                try:
                    col_f = X_np[:, i].astype(np.float64)
                except (ValueError, TypeError):
                    col_f = pd.to_numeric(pd.Series(X_np[:, i]), errors="coerce").values
            finite = col_f[np.isfinite(col_f)]
            if len(finite) == 0:
                finite = np.array([0.0, 1.0])
            q     = np.linspace(0, 100, MAX_BINS + 1)
            edges = np.unique(np.percentile(finite, q))
            if len(edges) < 2:
                edges = np.array([finite.min() - 1.0, finite.max() + 1.0])
            n_bins = min(len(edges) - 1, MAX_BINS)
            edges  = edges[:n_bins + 1]
            col_s  = np.where(~np.isfinite(col_f), -np.inf, col_f)
            b      = np.searchsorted(edges[1:], col_s, side="left")
            b      = np.clip(b, 0, n_bins - 1).astype(np.uint8)
            self._num_fidx.append(i)
            self._bin_thresholds.append(edges[1:])
            bin_cols.append(b)

        self._X_bins = (np.column_stack(bin_cols).astype(np.uint8)
                        if bin_cols
                        else np.empty((X_np.shape[0], 0), dtype=np.uint8))
        self._fidx_to_bincol: Dict[int, int] = {
            fi: j for j, fi in enumerate(self._num_fidx)
        }

    # ── Float64 numeric matrix  (for Phase-1 routing) ────────────────────────

    def _build_num_matrix(self, X_np: np.ndarray,
                          X_df: Optional[pd.DataFrame] = None) -> None:
        n, p   = X_np.shape
        self._X_num = np.full((n, p), np.nan, dtype=np.float64)
        for i, fname in enumerate(self.feature_names_):
            if fname not in self._cat_set:
                if X_df is not None and fname in X_df.columns:
                    col = X_df[fname]
                    if pd.api.types.is_numeric_dtype(col):
                        self._X_num[:, i] = col.to_numpy(dtype=np.float64)
                        continue
                try:
                    self._X_num[:, i] = X_np[:, i].astype(np.float64)
                except (ValueError, TypeError):
                    self._X_num[:, i] = pd.to_numeric(
                        pd.Series(X_np[:, i]), errors="coerce"
                    ).values

    # ── Importance scores ─────────────────────────────────────────────────────

    def _build_importance_scores(self, y: np.ndarray) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for i, fname in enumerate(self.feature_names_):
            if fname in self._cat_set:
                col = self._X_enc[:, i]
            elif i in self._fidx_to_bincol:
                col = self._X_bins[:, self._fidx_to_bincol[i]]
            else:
                scores[fname] = 0.0
                continue
            scores[fname] = self._compute_importance(col, y, fname in self._cat_set)
        return scores

    # ── Histogram construction  O(n · F) ─────────────────────────────────────

    def _build_hist(self, idx: np.ndarray, y_stat: np.ndarray) -> np.ndarray:
        n_num = len(self._num_fidx)
        ns    = y_stat.shape[1]
        hist  = np.zeros((n_num, MAX_BINS, ns), dtype=np.float64)
        if n_num == 0 or len(idx) == 0:
            return hist
        for j in range(n_num):
            b = self._X_bins[idx, j].astype(np.int64)
            for k in range(ns):
                hist[j, :, k] = np.bincount(b, weights=y_stat[:, k],
                                             minlength=MAX_BINS)
        return hist

    # ── All-features vectorised scan  O(B · F) ───────────────────────────────

    def _scan_all_hist(
        self, hist: np.ndarray, n: int, parent_imp: float, msl: int
    ) -> Tuple[Optional[int], Optional[int], float]:
        if hist.shape[0] == 0:
            return None, None, -np.inf
        cum   = np.cumsum(hist, axis=1)          # (F, B, S)
        total = cum[:, -1, :]                    # (F, S)
        nl    = cum[:, :-1, 0]                   # (F, B-1)
        nr    = total[:, 0:1] - nl
        valid = (nl >= msl) & (nr >= msl)
        gains = self._compute_gains_vectorized(cum, total, nl, nr, n, parent_imp)
        gains[~valid] = -np.inf
        if not (np.isfinite(gains) & (gains > -np.inf)).any():
            return None, None, -np.inf
        flat  = int(np.argmax(gains))
        n_feats = gains.shape[0]
        n_bins_m1 = gains.shape[1]
        bf = flat // n_bins_m1
        bb = flat %  n_bins_m1
        bf = min(bf, n_feats - 1)          # safety clamp
        bb = min(bb, n_bins_m1 - 1)
        return int(bf), int(bb), float(gains[bf, bb])

    # ── Histogram-optimal Phase-1 bin boundaries ─────────────────────────────

    def _p1_optimal_boundaries(
        self, idx: np.ndarray, fidx: int, n_bins: int
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Find n_bins-1 OPTIMAL split points for Phase-1 using the prebuilt
        histogram.  Uses recursive bisection: same thresholds CART would
        pick, but extracted as a k-way multi-split.

        Returns (groups dict keyed 0..n_bins-1, float edges array).
        """
        bincol = self._fidx_to_bincol.get(fidx)
        if bincol is None:
            return {}, np.array([-np.inf, np.inf])

        y_sub = self._y_enc[idx]
        ys    = self._y_stat(y_sub)
        n     = len(idx)
        h1    = np.zeros((1, MAX_BINS, ys.shape[1]))
        b_all = self._X_bins[idx, bincol].astype(np.int64)
        for k in range(ys.shape[1]):
            h1[0, :, k] = np.bincount(b_all, weights=ys[:, k], minlength=MAX_BINS)

        # Recursive bisection to find n_bins-1 cut points
        # (equivalent to building a balanced binary tree on one feature)
        def best_cut(h_seg: np.ndarray, start: int, end: int) -> Optional[int]:
            """Best cut point (histogram bin index) in [start, end)."""
            seg = h_seg[:, start:end, :]
            seg_n = int(seg[0, :, 0].sum())
            if seg_n < 2 * self.min_samples_leaf:
                return None
            cum_s  = np.cumsum(seg, axis=1)
            total_s = cum_s[:, -1, :]
            nl_s   = cum_s[:, :-1, 0]
            nr_s   = total_s[:, 0:1] - nl_s
            valid  = (nl_s >= self.min_samples_leaf) & (nr_s >= self.min_samples_leaf)
            pi_s   = self._impurity_from_hist(h_seg[0], start, end, seg_n)
            gains  = self._compute_gains_vectorized(cum_s, total_s, nl_s, nr_s,
                                                    seg_n, pi_s)
            gains[~valid] = -np.inf
            if not (gains > -np.inf).any():
                return None
            return int(np.argmax(gains[0])) + start  # offset back to absolute bin

        # Find n_bins-1 cuts via greedy recursive splitting
        intervals = [(0, MAX_BINS)]
        cuts: List[int] = []
        for _ in range(n_bins - 1):
            best_gain = -np.inf
            best_cut_val = None
            for (start, end) in intervals:
                c = best_cut(h1, start, end)
                if c is None:
                    continue
                pi = self._impurity_from_hist(h1[0], start, end,
                                              int(h1[0, start:end, 0].sum()))
                if pi > best_gain:
                    best_gain = pi
                    best_cut_val = (c, start, end)
            if best_cut_val is None:
                break
            c, st, en = best_cut_val
            cuts.append(c + 1)        # right boundary = c+1 (exclusive)
            intervals.remove((st, en))
            intervals.append((st, c + 1))
            intervals.append((c + 1, en))

        if not cuts:
            # Fall back to equal-count quantile binning
            return self._p1_num_groups_quantile(idx, fidx)

        cuts_sorted = sorted(set(cuts))
        boundaries  = [0] + cuts_sorted + [MAX_BINS]

        # Assign each training sample to a group
        grp_arr = np.searchsorted(boundaries[1:], b_all.astype(int), side="right")
        # Values land in interval [boundaries[g], boundaries[g+1])

        groups: Dict[int, np.ndarray] = {}
        for g in range(len(boundaries) - 1):
            m = grp_arr == g
            if m.sum() >= self.min_samples_leaf:
                groups[g] = idx[m]

        # Float edges for routing at predict time
        edges_f = [float(self._bin_thresholds[bincol][min(b-1, len(self._bin_thresholds[bincol])-1)])
                   if b > 0 else -np.inf
                   for b in boundaries]
        edges_f[-1] = np.inf
        return groups, np.array(edges_f, dtype=np.float64)

    def _impurity_from_hist(
        self, hist_f: np.ndarray, start: int, end: int, n: int
    ) -> float:
        """Compute impurity of a bin range [start, end) from histogram."""
        if n == 0:
            return 0.0
        seg = hist_f[start:end]
        return self._impurity_from_stats(seg.sum(axis=0), n)

    def _p1_num_groups_quantile(
        self, idx: np.ndarray, fidx: int
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """Fallback: local quantile bins on float values."""
        col_f  = self._X_num[idx, fidx]
        finite = ~np.isnan(col_f)
        if finite.sum() < 2:
            return {}, np.array([-np.inf, np.inf])
        q     = np.linspace(0, 100, self.n_bins + 1)
        edges = np.unique(np.percentile(col_f[finite], q))
        if len(edges) < 2:
            edges = np.array([col_f[finite].min() - 1e-9,
                               col_f[finite].max() + 1e-9])
        bin_idx              = np.digitize(col_f, edges[1:-1])
        bin_idx[~finite]     = 0
        groups: Dict[int, np.ndarray] = {}
        for b in range(len(edges) - 1):
            m = bin_idx == b
            if m.sum() >= self.min_samples_leaf:
                groups[b] = idx[m]
        return groups, edges

    # ─────────────────────────────────────────────────────────────────────────
    # Phase-1 skeleton  (sequential, lightweight)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_p1_skeleton(
        self, idx: np.ndarray, level: int,
        p2_queue: list, p1_path: frozenset
    ) -> _Node:
        y_sub = self._y_enc[idx]
        node  = _Node(depth=level, n_samples=len(idx),
                      value=self._leaf_value(y_sub),
                      impurity=self._impurity(y_sub))

        def pend():
            node._p1_path = p1_path     # store which P1 features were used
            p2_queue.append((node, idx, p1_path))
            return node

        if level >= len(self.feature_order_):
            return pend()
        if (len(idx) < self.min_samples_split or self._is_pure(y_sub)):
            node.is_leaf = True
            return node

        fname = self.feature_order_[level]
        fidx  = self._fname2idx[fname]

        if fname in self._cat_set:
            groups, sv = self._p1_cat_groups(idx, fidx)
            node.split_type = "categorical"
        else:
            groups, sv = self._p1_optimal_boundaries(idx, fidx, self.n_bins)
            node.split_type = "numeric_bins"

        if len(groups) < 2:
            return pend()

        # Fast impurity gate via histogram (O(MAX_BINS), not O(n))
        if not self._p1_impurity_gate(idx, fidx, fname, groups, node.impurity):
            return pend()

        node.feature = fname
        node.feature_idx = fidx
        node.split_values = sv
        new_path = p1_path | {fidx}

        for key, child_idx in groups.items():
            if len(child_idx) < self.min_samples_leaf:
                cy = self._y_enc[child_idx]
                ch = _Node(depth=level+1, is_leaf=True,
                           n_samples=len(child_idx),
                           value=self._leaf_value(cy),
                           impurity=self._impurity(cy))
            else:
                ch = self._build_p1_skeleton(
                    child_idx, level+1, p2_queue, new_path
                )
            node.children[key] = ch
        return node

    def _p1_impurity_gate(
        self, idx, fidx, fname, groups, parent_imp
    ) -> bool:
        """Fast impurity gate using cached group sizes."""
        n = len(idx)
        w_imp = sum(
            len(g) / n * self._impurity(self._y_enc[g])
            for g in groups.values()
        )
        return (parent_imp - w_imp) >= self.min_impurity_decrease

    def _p1_cat_groups(
        self, idx: np.ndarray, fidx: int
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        col    = self._X_enc[idx, fidx]
        vals, cnts = np.unique(col, return_counts=True)
        top    = vals[np.argsort(-cnts)[: self.max_categories]]
        mapped = np.where(np.isin(col, top), col, np.int32(-1))
        groups: Dict[int, np.ndarray] = {}
        for code in np.unique(mapped):
            m = mapped == code
            if m.sum() >= self.min_samples_leaf:
                groups[int(code)] = idx[m]
        return groups, top

    # ─────────────────────────────────────────────────────────────────────────
    # Phase-2  Per-root best-leaf-first CART
    # ─────────────────────────────────────────────────────────────────────────

    def _build_p2_root(
        self, idx: np.ndarray, p1_path: frozenset
    ) -> _Node:
        """Grow one Phase-2 subtree independently."""
        if self.subsample < 1.0 and len(idx) > 1:
            n_keep = max(self.min_samples_split,
                         int(len(idx) * self.subsample))
            idx = self._rng_p2.choice(idx, n_keep, replace=False)

        y_sub = self._y_enc[idx]
        root  = _Node(depth=0, is_phase2=True,
                      n_samples=len(idx),
                      value=self._leaf_value(y_sub),
                      impurity=self._impurity(y_sub))

        if (len(idx) < self.min_samples_split
                or self._is_pure(y_sub)
                or self._effective_n(y_sub) < self.min_child_weight):
            root.is_leaf = True
            return root

        ys   = self._y_stat(y_sub)
        hist = self._build_hist(idx, ys)
        info = self._best_p2_split(hist, idx, y_sub, p1_path)

        if info is None or info["gain"] < self.min_impurity_decrease:
            root.is_leaf = True
            return root

        return self._grow_best_leaf_local(root, idx, hist, info, p1_path)

    def _grow_best_leaf_local(
        self, root, root_idx, root_hist, root_split, p1_path
    ) -> _Node:
        heap: list = []
        ctr   = [0]
        n_lv  = [1]

        def push(nd, idx, hist, info):
            if (info is not None
                    and info["gain"] >= self.min_impurity_decrease
                    and len(idx) >= self.min_samples_split
                    and self._effective_n(self._y_enc[idx]) >= self.min_child_weight):
                heapq.heappush(heap, (-info["gain"], ctr[0], nd, idx, hist, info))
                ctr[0] += 1
            else:
                nd.is_leaf = True

        push(root, root_idx, root_hist, root_split)

        while heap and n_lv[0] < self.max_leaves:
            neg_g, _, node, idx, par_hist, info = heapq.heappop(heap)

            if self.max_depth is not None and node.depth >= self.max_depth:
                node.is_leaf = True; continue

            lm = info["left_mask"]
            li = idx[lm]; ri = idx[~lm]
            if len(li) < self.min_samples_leaf or len(ri) < self.min_samples_leaf:
                node.is_leaf = True; continue

            yl = self._y_enc[li]; yr = self._y_enc[ri]
            node.feature_idx   = info["fidx"]
            node.feature       = self.feature_names_[info["fidx"]]
            node.threshold     = info["threshold"]
            node.split_type    = info["stype"]
            node.nan_goes_left = info.get("nan_left", True)
            node.impurity      = info["parent_imp"]

            node.left  = _Node(depth=node.depth+1, is_phase2=True,
                               n_samples=len(li), value=self._leaf_value(yl),
                               impurity=self._impurity(yl))
            node.right = _Node(depth=node.depth+1, is_phase2=True,
                               n_samples=len(ri), value=self._leaf_value(yr),
                               impurity=self._impurity(yr))
            # Leaf smoothing: blend child estimates toward parent (reduces variance)
            if self.leaf_smooth > 0:
                parent_val = node.value  # value if this node were a leaf
                node.left.value  = self._smooth_leaf_val(
                    node.left.value,  len(li), parent_val)
                node.right.value = self._smooth_leaf_val(
                    node.right.value, len(ri), parent_val)
            n_lv[0] += 1

            ysl = self._y_stat(yl); ysr = self._y_stat(yr)
            if len(li) <= len(ri):
                lh = self._build_hist(li, ysl); rh = par_hist - lh
            else:
                rh = self._build_hist(ri, ysr); lh = par_hist - rh

            push(node.left,  li, lh, self._best_p2_split(lh, li, yl, p1_path))
            push(node.right, ri, rh, self._best_p2_split(rh, ri, yr, p1_path))

        while heap:
            _, _, nd, *_ = heapq.heappop(heap)
            nd.is_leaf = True

        return root

    # ── Best-split dispatcher ─────────────────────────────────────────────────

    def _best_p2_split(
        self, hist, idx, y_sub, p1_path: frozenset
    ) -> Optional[dict]:
        n = len(idx)
        if n < 2 * self.min_samples_leaf:
            return None
        parent_imp = self._impurity(y_sub)
        best: Optional[dict] = None

        # ── Numeric features (all-features vectorised) ────────────────────────
        fidx_num_all = list(range(len(self._num_fidx)))
        # Build candidate feature lists: non-P1 first (if any), fall back to all
        if self.p2_exclude_p1_features and p1_path:
            fidx_num_nop1 = [j for j, fi in enumerate(self._num_fidx)
                             if fi not in p1_path]
        else:
            fidx_num_nop1 = fidx_num_all

        # Iterate: try non-P1 first; if same as all, one pass only
        feature_lists = ([fidx_num_nop1, fidx_num_all]
                         if fidx_num_nop1 and fidx_num_nop1 != fidx_num_all
                         else [fidx_num_all])
        for fidx_list in feature_lists:
            fi_use = self._apply_max_features_num(fidx_list)
            if not fi_use:
                continue
            sub_hist = hist[fi_use]
            bf, bb, gain = self._scan_all_hist(sub_hist, n, parent_imp,
                                               self.min_samples_leaf)
            if bf is None or gain <= -np.inf:
                continue
            # bf is an index into fi_use (which is a subset of bincol indices)
            bincol_idx = fi_use[bf]               # index into self._num_fidx
            actual_bc  = bincol_idx               # hist col == bincol
            fidx_orig  = self._num_fidx[actual_bc]
            edges      = self._bin_thresholds[actual_bc]
            thr_f      = float(edges[bb]) if bb < len(edges) else float(edges[-1])
            lm         = self._X_bins[idx, actual_bc] <= bb
            if best is None or gain > best["gain"]:
                best = dict(fidx=fidx_orig, threshold=thr_f, stype="num_hist",
                            left_mask=lm, gain=gain,
                            parent_imp=parent_imp, nan_left=True)
            if best is not None:
                break   # found improvement from non-P1 features

        # ── Categorical features ──────────────────────────────────────────────
        cat_fidxs = [self._fname2idx[f] for f in self.feature_names_
                     if f in self._cat_set]
        if self.p2_exclude_p1_features and p1_path:
            cat_fidxs = [fi for fi in cat_fidxs if fi not in p1_path] or cat_fidxs
        cat_fidxs = self._apply_max_features_cat(cat_fidxs)

        for fidx in cat_fidxs:
            col  = self._X_enc[idx, fidx]
            cats = np.unique(col)
            if len(cats) < 2:
                continue
            order  = self._cat_split_order(y_sub, col, cats)
            cats_s = cats[order]
            for i in range(1, len(cats_s)):
                lc = cats_s[:i]; lm = np.isin(col, lc)
                nl, nr = int(lm.sum()), int((~lm).sum())
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                g = (parent_imp
                     - nl / n * self._impurity(y_sub[lm])
                     - nr / n * self._impurity(y_sub[~lm]))
                if best is None or g > best["gain"]:
                    best = dict(fidx=fidx, threshold=lc.copy(), stype="cat_binary",
                                left_mask=lm, gain=g,
                                parent_imp=parent_imp, nan_left=True)
        return best

    def _apply_max_features_num(self, fidx_list: List[int]) -> List[int]:
        mf = self._max_features_int
        total = self._n_features
        if mf >= total:
            return fidx_list
        chosen = set(self._rng_p2.choice(total, mf, replace=False))
        result = [j for j in fidx_list if self._num_fidx[j] in chosen]
        return result if result else fidx_list  # fallback: all

    def _apply_max_features_cat(self, fidx_list: List[int]) -> List[int]:
        mf = self._max_features_int
        total = self._n_features
        if mf >= total:
            return fidx_list
        chosen = set(self._rng_p2.choice(total, mf, replace=False))
        result = [fi for fi in fidx_list if fi in chosen]
        return result if result else fidx_list

    # ── Run all Phase-2 roots  (parallel when worthwhile) ────────────────────

    def _run_p2(self, p2_queue: list) -> None:
        if not p2_queue:
            return

        # Decide parallelism: worth it when total work >= threshold
        total_n = sum(len(idx) for _, idx, _ in p2_queue)
        # Only parallelize when individual subtrees are large enough
        # to amortize joblib spawn overhead (empirically ≥2000 samples/root)
        avg_n   = total_n / max(len(p2_queue), 1)
        use_par = (self.n_jobs != 1
                   and len(p2_queue) >= 4
                   and avg_n >= 2000)

        if use_par:
            # Build independent RNGs per task to avoid race conditions
            seeds = [int(self._rng_p2.integers(0, 2**31)) for _ in p2_queue]

            def run_one(placeholder, idx, p1_path, seed):
                self._rng_p2 = np.random.default_rng(seed)
                p2_root = self._build_p2_root(idx, p1_path)
                return p2_root

            p2_roots = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(run_one)(ph, idx, path, s)
                for (ph, idx, path), s in zip(p2_queue, seeds)
            )
        else:
            p2_roots = [self._build_p2_root(idx, path)
                        for _, idx, path in p2_queue]

        for (placeholder, _, _), p2_root in zip(p2_queue, p2_roots):
            placeholder.__dict__.update(p2_root.__dict__)
            placeholder.is_phase2 = True
            if hasattr(placeholder, "_p1_path"):
                del placeholder._p1_path

    # ── CCP pruning ───────────────────────────────────────────────────────────

    def _ccp_prune(self, node: _Node) -> _Node:
        if node.is_leaf:
            return node
        for k in list(node.children):
            node.children[k] = self._ccp_prune(node.children[k])
        if node.left:  node.left  = self._ccp_prune(node.left)
        if node.right: node.right = self._ccp_prune(node.right)
        leaves = self._leaves(node)
        if len(leaves) <= 1:
            return node
        r_tt = sum(l.impurity * l.n_samples for l in leaves) / self._n_train
        r_t  = node.impurity * node.n_samples / self._n_train
        g    = (r_t - r_tt) / (len(leaves) - 1)
        if g <= self.ccp_alpha:
            node.is_leaf = True; node.children = {}
            node.left = node.right = None; node.split_type = None
        return node

    @staticmethod
    def _leaves(node: _Node) -> list:
        if node.is_leaf:
            return [node]
        out: list = []
        for c in node.children.values(): out.extend(_IVTreeBase._leaves(c))
        if node.left:  out.extend(_IVTreeBase._leaves(node.left))
        if node.right: out.extend(_IVTreeBase._leaves(node.right))
        return out

    # ── Vectorised batch prediction ───────────────────────────────────────────

    def _route(self, X_np: np.ndarray, X_bins_pred: np.ndarray) -> list:
        n   = len(X_np)
        out = [None] * n
        stack = [(self.tree_, np.arange(n, dtype=np.int64))]

        while stack:
            node, idx = stack.pop()
            if not len(idx): continue
            if node.is_leaf:
                for i in idx: out[i] = node.value
                continue
            st = node.split_type

            if st == "numeric_bins":
                try:
                    col = X_np[idx, node.feature_idx].astype(np.float64)
                except (ValueError, TypeError):
                    col = pd.to_numeric(
                        pd.Series(X_np[idx, node.feature_idx]), errors="coerce"
                    ).values
                edges = node.split_values
                grp   = np.searchsorted(edges[1:-1], col, side="left")
                grp[np.isnan(col)] = 0
                grp   = np.clip(grp, 0, len(edges) - 2)
                valid = np.zeros(len(idx), dtype=bool)
                for key, child in node.children.items():
                    m = grp == key
                    if m.any(): valid |= m; stack.append((child, idx[m]))
                for i in idx[~valid]: out[i] = node.value

            elif st == "categorical":
                col    = self._encode_col_predict(
                    X_np[idx, node.feature_idx], node.feature)
                top    = set(node.children) - {-1}
                mapped = np.where(np.isin(col, list(top)), col, np.int32(-1))
                valid  = np.zeros(len(idx), dtype=bool)
                for code, child in node.children.items():
                    m = mapped == code
                    if m.any(): valid |= m; stack.append((child, idx[m]))
                for i in idx[~valid]: out[i] = node.value

            elif st == "num_hist":
                bincol = self._fidx_to_bincol.get(node.feature_idx)
                if bincol is None:
                    for i in idx: out[i] = node.value; continue
                col = X_bins_pred[idx, bincol]
                thr_bin = int(np.searchsorted(
                    self._bin_thresholds[bincol], node.threshold, side="left"
                ))
                try:
                    raw_f    = X_np[idx, node.feature_idx].astype(np.float64)
                    nan_mask = np.isnan(raw_f)
                except (ValueError, TypeError):
                    nan_mask = np.zeros(len(idx), dtype=bool)
                lm = col <= thr_bin
                lm[nan_mask] = node.nan_goes_left
                if lm.any():   stack.append((node.left,  idx[lm]))
                if (~lm).any(): stack.append((node.right, idx[~lm]))

            elif st == "cat_binary":
                col = self._encode_col_predict(
                    X_np[idx, node.feature_idx], node.feature)
                lm  = np.isin(col, node.threshold)
                if lm.any():   stack.append((node.left,  idx[lm]))
                if (~lm).any(): stack.append((node.right, idx[~lm]))

        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Fit boilerplate
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_max_features(self) -> int:
        p  = self._n_features
        mf = self.max_features
        if mf is None or mf == "all": return p
        if mf == "sqrt":  return max(1, int(np.sqrt(p)))
        if mf == "log2":  return max(1, int(np.log2(max(p, 2))))
        if isinstance(mf, float): return max(1, int(mf * p))
        return max(1, int(mf))

    def _fit_prepare(self, X, y_raw):
        if isinstance(X, pd.DataFrame):
            self.feature_names_: List[str] = list(X.columns)
            self._cat_set: set = (
                set(self.categorical_features)
                if self.categorical_features is not None
                else {c for c in X.columns
                      if X[c].dtype == object
                      or pd.api.types.is_bool_dtype(X[c])
                      or str(X[c].dtype) in ("category", "string")}
            )
            X_np = X.to_numpy(dtype=object)
        else:
            X_np = np.asarray(X, dtype=object)
            self.feature_names_ = [f"x{i}" for i in range(X_np.shape[1])]
            self._cat_set = set(self.categorical_features or [])

        self._n_features  = len(self.feature_names_)
        self._fname2idx: Dict[str, int] = {
            f: i for i, f in enumerate(self.feature_names_)
        }
        self._y_enc: np.ndarray = self._encode_y(np.asarray(y_raw))
        self._n_train = len(self._y_enc)

        _df_arg = X if isinstance(X, pd.DataFrame) else None
        self._build_num_matrix(X_np, _df_arg)   # build floats first (fast path)
        self._prebuild_bins(X_np)                # reuse _X_num inside
        self._encode_categoricals(X_np)

        scores   = self._build_importance_scores(self._y_enc)
        col_name = self._importance_col_name
        self.importance_summary_ = (
            pd.DataFrame({"feature": list(scores), col_name: list(scores.values())})
            .sort_values(col_name, ascending=False)
            .reset_index(drop=True)
        )
        self.feature_order_: List[str] = (
            self.importance_summary_
            .loc[self.importance_summary_[col_name] >= self.min_importance, "feature"]
            .tolist()
        )
        if self.max_p1_depth is not None:
            self.feature_order_ = self.feature_order_[:self.max_p1_depth]
        if not self.feature_order_:
            warnings.warn(
                f"No features meet min_importance={self.min_importance}. "
                "Phase-1 skipped; pure histogram CART.",
                UserWarning, stacklevel=3,
            )

        total = sum(scores.values()) or 1.0
        self.feature_importances_ = np.array(
            [scores[f] / total for f in self.feature_names_]
        )
        seed = self.random_state if self.random_state is not None else 0
        self._rng_p2  = np.random.default_rng(seed)
        self._max_features_int = self._resolve_max_features()
        # Cache raw object array for reuse in Phase-3 leaf model fitting
        self._X_np_cache = X_np

    def _fit_tree(self):
        p2_queue: list = []
        self.tree_ = self._build_p1_skeleton(
            np.arange(self._n_train, dtype=np.int64), 0, p2_queue, frozenset()
        )
        self._run_p2(p2_queue)
        if self.ccp_alpha > 0.0:
            self.tree_ = self._ccp_prune(self.tree_)
        # Assign stable integer node ids for apply() / decision_path()
        self._number_nodes()

    def _make_pred_bins(self, X_np: np.ndarray) -> np.ndarray:
        if not self._num_fidx:
            return np.empty((len(X_np), 0), dtype=np.uint8)
        cols = []
        for j, fidx in enumerate(self._num_fidx):
            col = X_np[:, fidx]
            try:
                col_f = col.astype(np.float64)
            except (ValueError, TypeError):
                col_f = pd.to_numeric(pd.Series(col), errors="coerce").values
            col_s = np.where(~np.isfinite(col_f), -np.inf, col_f)
            edges = self._bin_thresholds[j]
            b     = np.searchsorted(edges, col_s, side="left")
            cols.append(np.clip(b, 0, MAX_BINS - 1).astype(np.uint8))
        return np.column_stack(cols).astype(np.uint8)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def export_text(self, max_depth: int = 6) -> str:
        check_is_fitted(self)
        lines: List[str] = []

        def _r(node: _Node, pfx: str, d: int) -> None:
            if d > max_depth:
                lines.append(pfx + "  ..."); return
            tag = "[P2]" if node.is_phase2 else "[P1]"
            if node.is_leaf:
                lines.append(f"{pfx}{tag} LEAF n={node.n_samples}"
                             f" val={self._format_leaf(node.value)}"
                             f" imp={node.impurity:.4f}")
                return
            if node.split_type == "numeric_bins":
                lines.append(f"{pfx}{tag} [{node.feature}] {node.n_samples}s  "
                             f"imp={node.impurity:.4f}")
            elif node.split_type == "categorical":
                top = [self._cat_decoders.get(node.feature,{}).get(c,c)
                       for c in node.split_values[:5]]
                lines.append(f"{pfx}{tag} [{node.feature}] top={top}  "
                             f"imp={node.impurity:.4f} n={node.n_samples}")
            elif node.split_type == "num_hist":
                lines.append(f"{pfx}{tag} [{node.feature}]≤{node.threshold:.4g}  "
                             f"imp={node.impurity:.4f} n={node.n_samples}"
                             f" NaN→{'L' if node.nan_goes_left else 'R'}")
            else:
                dec  = self._cat_decoders.get(node.feature,{})
                cats = [dec.get(c,c) for c in node.threshold[:5]]
                lines.append(f"{pfx}{tag} [{node.feature}]∈{cats}  "
                             f"imp={node.impurity:.4f} n={node.n_samples}")
            indent = pfx + "  │ "
            if node.children:
                items = list(node.children.items())
                for j, (k, ch) in enumerate(items):
                    dec   = self._cat_decoders.get(node.feature,{})
                    lbl   = (dec.get(k, "__other__") if node.split_type=="categorical"
                             else f"g{k}")
                    lines.append(f"{pfx}  {'└─' if j==len(items)-1 else '├─'} {lbl!r}")
                    _r(ch, indent, d+1)
            else:
                if node.left:  lines.append(f"{pfx}  ├─ left");  _r(node.left,  indent, d+1)
                if node.right: lines.append(f"{pfx}  └─ right"); _r(node.right, indent, d+1)

        _r(self.tree_, "", 0); return "\n".join(lines)

    def plot_importance_summary(self, figsize=(9, 5)) -> None:
        check_is_fitted(self)
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib required.")
        col  = self._importance_col_name
        df   = self.importance_summary_
        p1s  = set(self.feature_order_)
        clrs = ["#1a5fa8" if f in p1s else "#a8c8e8" for f in df["feature"][::-1]]
        _, ax = plt.subplots(figsize=figsize)
        ax.barh(df["feature"][::-1], df[col][::-1], color=clrs)
        ax.axvline(self.min_importance, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel(col.upper()); ax.set_title(f"Importance ({col.upper()})")
        ax.legend(handles=[
            mpatches.Patch(color="#1a5fa8", label="Phase-1 feature"),
            mpatches.Patch(color="#a8c8e8", label="Phase-2 only"),
            plt.Line2D([0],[0], color="red", linestyle="--",
                       label=f"threshold={self.min_importance}"),
        ], fontsize=8, loc="lower right")
        plt.tight_layout(); plt.show()

    # ── Node numbering (DFS pre-order, assigned once after fit) ──────────────

    def _number_nodes(self) -> None:
        """Assign a unique integer id (DFS pre-order) to every node.
        Also builds:
          _node_by_id  : dict[int → _Node]
          _parent_id   : dict[int → int]   (root has no entry)
          _node_path   : dict[int → list[int]]  path from root to node
        """
        self._node_by_id:  Dict[int, object] = {}
        self._parent_id:   Dict[int, int]    = {}
        self._node_path:   Dict[int, List[int]] = {}
        counter = [0]

        def _dfs(node, parent_id, path):
            nid = counter[0]; counter[0] += 1
            node._node_id = nid
            self._node_by_id[nid] = node
            if parent_id is not None:
                self._parent_id[nid] = parent_id
            self._node_path[nid] = path + [nid]
            # Phase-1 children (multi-way)
            for child in node.children.values():
                _dfs(child, nid, self._node_path[nid])
            # Phase-2 children (binary)
            if node.left:
                _dfs(node.left,  nid, self._node_path[nid])
            if node.right:
                _dfs(node.right, nid, self._node_path[nid])

        _dfs(self.tree_, None, [])
        self._n_nodes = counter[0]

    # ── sklearn-compatible public API ─────────────────────────────────────────

    # --- Structural queries ---------------------------------------------------

    def get_depth(self) -> int:
        """Return the maximum depth of the tree."""
        check_is_fitted(self)
        def _depth(node) -> int:
            if node is None or node.is_leaf: return 0
            d = max((_depth(c) for c in node.children.values()), default=0)
            if node.left:  d = max(d, _depth(node.left))
            if node.right: d = max(d, _depth(node.right))
            return 1 + d
        return _depth(self.tree_)

    def get_n_leaves(self) -> int:
        """Return the number of leaves of the tree."""
        check_is_fitted(self)
        return len(_IVTreeBase._leaves(self.tree_))

    @property
    def node_count(self) -> int:
        """Total number of nodes (internal + leaves)."""
        check_is_fitted(self)
        if not hasattr(self, '_n_nodes'):
            self._number_nodes()
        return self._n_nodes

    # --- apply() and decision_path() -----------------------------------------

    def apply(self, X) -> np.ndarray:
        """
        Return the leaf node ID reached by each sample.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)

        Returns
        -------
        leaf_ids : ndarray of shape (n_samples,), dtype int
            Unique integer node id (DFS pre-order) of the leaf for each sample.

        Examples
        --------
        >>> leaf_ids = clf.apply(X_test)
        >>> # Count samples per leaf
        >>> np.unique(leaf_ids, return_counts=True)
        """
        check_is_fitted(self)
        if not hasattr(self, '_n_nodes'):
            self._number_nodes()
        X_np   = (X.to_numpy(dtype=object) if isinstance(X, pd.DataFrame)
                  else np.asarray(X, dtype=object))
        X_bins = self._make_pred_bins(X_np)
        n      = len(X_np)
        leaf_ids = np.zeros(n, dtype=np.int64)

        stack = [(self.tree_, np.arange(n, dtype=np.int64))]
        while stack:
            node, idx = stack.pop()
            if not len(idx): continue
            if node.is_leaf:
                leaf_ids[idx] = node._node_id
                continue
            st = node.split_type
            if st == "numeric_bins":
                try:    col = X_np[idx, node.feature_idx].astype(np.float64)
                except: col = pd.to_numeric(pd.Series(X_np[idx, node.feature_idx]), errors="coerce").values
                edges = node.split_values
                grp   = np.searchsorted(edges[1:-1], col, side="left")
                grp[np.isnan(col)] = 0; grp = np.clip(grp, 0, len(edges) - 2)
                valid = np.zeros(len(idx), dtype=bool)
                for key, child in node.children.items():
                    m = grp == key
                    if m.any(): valid |= m; stack.append((child, idx[m]))
                leaf_ids[idx[~valid]] = node._node_id
            elif st == "categorical":
                col    = self._encode_col_predict(X_np[idx, node.feature_idx], node.feature)
                top    = set(node.children) - {-1}
                mapped = np.where(np.isin(col, list(top)), col, np.int32(-1))
                valid  = np.zeros(len(idx), dtype=bool)
                for code, child in node.children.items():
                    m = mapped == code
                    if m.any(): valid |= m; stack.append((child, idx[m]))
                leaf_ids[idx[~valid]] = node._node_id
            elif st == "num_hist":
                bincol = self._fidx_to_bincol.get(node.feature_idx)
                if bincol is None: leaf_ids[idx] = node._node_id; continue
                col = X_bins[idx, bincol]
                thr_bin = int(np.searchsorted(self._bin_thresholds[bincol],
                                               node.threshold, side="left"))
                try:    raw_f = X_np[idx, node.feature_idx].astype(np.float64); nan_m = np.isnan(raw_f)
                except: nan_m = np.zeros(len(idx), dtype=bool)
                lm = col <= thr_bin; lm[nan_m] = node.nan_goes_left
                if lm.any():    stack.append((node.left,  idx[lm]))
                if (~lm).any(): stack.append((node.right, idx[~lm]))
            elif st == "cat_binary":
                col = self._encode_col_predict(X_np[idx, node.feature_idx], node.feature)
                lm  = np.isin(col, node.threshold)
                if lm.any():    stack.append((node.left,  idx[lm]))
                if (~lm).any(): stack.append((node.right, idx[~lm]))
        return leaf_ids

    def decision_path(self, X) -> csr_matrix:
        """
        Return the decision path as a sparse indicator matrix.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            indicator[i, j] = 1 if sample i passed through node j.
            Compatible with sklearn's decision_path interface.

        Examples
        --------
        >>> dp = clf.decision_path(X_test)
        >>> # Path length per sample
        >>> path_lengths = dp.sum(axis=1)
        """
        check_is_fitted(self)
        if not hasattr(self, '_n_nodes'):
            self._number_nodes()
        leaf_ids = self.apply(X)
        n = len(leaf_ids)
        # Build CSR: for each sample, set all nodes on its path to 1
        row_indices = []
        col_indices = []
        for i, lid in enumerate(leaf_ids):
            path = self._node_path.get(int(lid), [int(lid)])
            row_indices.extend([i] * len(path))
            col_indices.extend(path)
        data = np.ones(len(row_indices), dtype=np.uint8)
        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n, self._n_nodes),
        )

    # --- Rule extraction ------------------------------------------------------

    def extract_rules(
        self,
        X=None,
        feature_names: Optional[List[str]] = None,
        max_rules: Optional[int] = None,
        min_samples: int = 1,
    ) -> List[str]:
        """
        Extract human-readable decision rules for every leaf.

        Each rule is a conjunction of conditions from root to leaf, e.g.:
            IF num_3 <= 0.52  AND  cat_signal = 'high'
            THEN p(y=1) = 0.87  [n=142]

        Parameters
        ----------
        X : array-like or None
            If provided, only return rules for leaves actually reached by X.
        feature_names : list[str] or None
            Override feature names for display.
        max_rules : int or None
            Maximum number of rules to return (sorted by n_samples desc).
        min_samples : int
            Only return rules for leaves with at least this many training samples.

        Returns
        -------
        rules : list of str
            One rule string per qualifying leaf.

        Examples
        --------
        >>> rules = clf.extract_rules(X_test, min_samples=20)
        >>> for rule in rules[:3]:
        ...     print(rule)
        """
        check_is_fitted(self)
        fnames = feature_names or self.feature_names_

        # Filter by leaves reached by X if provided
        if X is not None:
            active_ids = set(self.apply(X).tolist())
        else:
            active_ids = None

        rules: List[str] = []

        def _path_conditions(node, conditions: List[str]) -> None:
            """DFS that accumulates conditions and emits a rule at each leaf."""
            if node.is_leaf:
                if node.n_samples < min_samples:
                    return
                if active_ids is not None and node._node_id not in active_ids:
                    return
                cond_str = ("\n   AND ".join(conditions)
                            if conditions else "(root)")
                val_str  = self._format_leaf(node.value)
                rules.append(
                    f"IF {cond_str}\n"
                    f"   THEN {val_str}  [n_train={node.n_samples}  "
                    f"impurity={node.impurity:.4f}]"
                )
                return

            st = node.split_type
            fname = fnames[node.feature_idx] if node.feature_idx is not None else "?"
            dec   = self._cat_decoders.get(node.feature or "", {})

            # Phase-1: multi-way categorical
            if st == "categorical":
                for code, child in node.children.items():
                    cat_val = dec.get(code, code) if code != -1 else "__other__"
                    _path_conditions(child, conditions + [f"{fname} = '{cat_val}'"])

            # Phase-1: multi-way numeric bins
            elif st == "numeric_bins":
                edges = node.split_values
                for g, child in node.children.items():
                    lo = edges[g]   if g < len(edges)   else -np.inf
                    hi = edges[g+1] if g+1 < len(edges) else np.inf
                    if lo == -np.inf:
                        cond = f"{fname} <= {hi:.5g}"
                    elif hi == np.inf:
                        cond = f"{fname} > {lo:.5g}"
                    else:
                        cond = f"{lo:.5g} < {fname} <= {hi:.5g}"
                    _path_conditions(child, conditions + [cond])

            # Phase-2: numeric binary
            elif st == "num_hist":
                thr = node.threshold
                nan_side = "left" if node.nan_goes_left else "right"
                cond_l = f"{fname} <= {thr:.5g}  (NaN→{nan_side})"
                cond_r = f"{fname} >  {thr:.5g}"
                if node.left:  _path_conditions(node.left,  conditions + [cond_l])
                if node.right: _path_conditions(node.right, conditions + [cond_r])

            # Phase-2: categorical binary
            elif st == "cat_binary":
                left_cats  = [str(dec.get(c, c)) for c in node.threshold]
                right_cats = [str(v) for v in dec.values()
                              if v not in left_cats and v != "__other__"]
                cond_l = f"{fname} ∈ {left_cats}"
                cond_r = f"{fname} ∉ {left_cats}" + (f"  (i.e. {right_cats})" if right_cats else "")
                if node.left:  _path_conditions(node.left,  conditions + [cond_l])
                if node.right: _path_conditions(node.right, conditions + [cond_r])

        if not hasattr(self.tree_, '_node_id'):
            self._number_nodes()
        _path_conditions(self.tree_, [])

        # Sort by descending training samples
        rules.sort(key=lambda r: -int(r.split("n_train=")[1].split()[0]))
        if max_rules is not None:
            rules = rules[:max_rules]
        return rules

    def get_rules_dataframe(
        self,
        X=None,
        min_samples: int = 1,
    ) -> "pd.DataFrame":
        """
        Return rules as a structured DataFrame with columns:
            rule_id, conditions, leaf_value, n_train, impurity, node_id.

        Parameters
        ----------
        X : array-like or None
            Filter to leaves reached by X.
        min_samples : int
            Minimum training samples per leaf.

        Returns
        -------
        pd.DataFrame  sorted by n_train descending.
        """
        check_is_fitted(self)
        if not hasattr(self.tree_, '_node_id'):
            self._number_nodes()
        fnames = self.feature_names_
        if X is not None:
            active_ids = set(self.apply(X).tolist())
        else:
            active_ids = None

        rows: List[dict] = []

        def _collect(node, conditions):
            if node.is_leaf:
                if node.n_samples < min_samples: return
                if active_ids is not None and node._node_id not in active_ids: return
                rows.append({
                    "node_id":    node._node_id,
                    "conditions": conditions[:],
                    "n_conditions": len(conditions),
                    "n_train":    node.n_samples,
                    "impurity":   round(node.impurity, 6),
                    "leaf_value": self._format_leaf(node.value),
                })
                return
            st    = node.split_type
            fname = (fnames[node.feature_idx]
                     if node.feature_idx is not None else "?")
            dec   = self._cat_decoders.get(node.feature or "", {})

            if st == "categorical":
                for code, child in node.children.items():
                    val = dec.get(code, code) if code != -1 else "__other__"
                    _collect(child, conditions + [f"{fname}='{val}'"])
            elif st == "numeric_bins":
                edges = node.split_values
                for g, child in node.children.items():
                    lo = edges[g]   if g < len(edges)   else -np.inf
                    hi = edges[g+1] if g+1 < len(edges) else np.inf
                    if lo == -np.inf:    cond = f"{fname}<={hi:.5g}"
                    elif hi == np.inf:   cond = f"{fname}>{lo:.5g}"
                    else:                cond = f"{lo:.5g}<{fname}<={hi:.5g}"
                    _collect(child, conditions + [cond])
            elif st == "num_hist":
                if node.left:
                    _collect(node.left,  conditions + [f"{fname}<={node.threshold:.5g}"])
                if node.right:
                    _collect(node.right, conditions + [f"{fname}>{node.threshold:.5g}"])
            elif st == "cat_binary":
                left_cats = [str(dec.get(c, c)) for c in node.threshold]
                if node.left:
                    _collect(node.left,  conditions + [f"{fname}∈{left_cats}"])
                if node.right:
                    _collect(node.right, conditions + [f"{fname}∉{left_cats}"])

        _collect(self.tree_, [])
        if not rows:
            return pd.DataFrame(columns=["node_id","conditions","n_conditions",
                                         "n_train","impurity","leaf_value"])
        df = pd.DataFrame(rows).sort_values("n_train", ascending=False)
        df.index = range(len(df))
        df.insert(0, "rule_id", df.index)
        return df

    # --- Feature importance by split (gini / mse decrease) -------------------

    @property
    def split_importances_(self) -> np.ndarray:
        """
        Feature importances based on total impurity decrease from splits
        (identical to sklearn's feature_importances_).
        Uses actual splits in the fitted tree, complementing IV/RV scores.
        Normalised to sum to 1.
        """
        check_is_fitted(self)
        importances = np.zeros(self._n_features)

        def _accum(node):
            if node is None or node.is_leaf: return
            fidx = node.feature_idx
            if fidx is not None:
                # Impurity decrease = n_node * imp - sum(n_child * imp_child)
                n    = node.n_samples
                gain = node.impurity * n
                for child in list(node.children.values()) + [node.left, node.right]:
                    if child is not None:
                        gain -= child.impurity * child.n_samples
                importances[fidx] += max(gain, 0.0)
            for c in node.children.values(): _accum(c)
            _accum(node.left); _accum(node.right)

        _accum(self.tree_)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

    @property
    def split_importances_df_(self) -> pd.DataFrame:
        """
        DataFrame of feature names + split importance scores, sorted descending.
        """
        check_is_fitted(self)
        imp = self.split_importances_
        return (pd.DataFrame({"feature": self.feature_names_, "split_importance": imp})
                .sort_values("split_importance", ascending=False)
                .reset_index(drop=True))

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def _importance_col_name(self) -> str: ...
    @abstractmethod
    def _compute_importance(self, col, y, is_cat) -> float: ...
    @abstractmethod
    def _encode_y(self, y_raw) -> np.ndarray: ...
    @abstractmethod
    def _impurity(self, y) -> float: ...
    @abstractmethod
    def _impurity_from_stats(self, stats, n) -> float: ...
    @abstractmethod
    def _leaf_value(self, y) -> object: ...
    @abstractmethod
    def _is_pure(self, y) -> bool: ...
    @abstractmethod
    def _effective_n(self, y) -> float: ...
    @abstractmethod
    def _y_stat(self, y) -> np.ndarray: ...
    @abstractmethod
    def _compute_gains_vectorized(self, cum, total, nl, nr, n, pi) -> np.ndarray: ...
    @abstractmethod
    def _smooth_leaf_val(self, child_val, n_child, parent_val) -> object: ...
    @abstractmethod
    def _cat_split_order(self, y_sub, col_enc, cats) -> np.ndarray: ...
    @abstractmethod
    def _format_leaf(self, val) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class IVDecisionTreeClassifier(_IVTreeBase, ClassifierMixin):
    """
    IV-ordered multi-way Decision Tree Classifier.

    Phase-1  : IV-ranked multi-way splits with HISTOGRAM-OPTIMAL boundaries.
    Phase-2  : Per-root best-leaf-first histogram CART (Gini / Entropy).
               Parallelised with joblib when n_roots ≥ 4 and n ≥ 4000.

    Parameters
    ----------
    n_bins : int, default=3
        Number of Phase-1 branches per numeric feature.
        Keep small (2–5): larger values fragment data and hurt Phase-2.
    max_bins : int, default=128
        Histogram resolution for Phase-2 numeric splits.
    iv_bins : int, default=10
        Bins used when computing IV importance scores.
    min_importance : float, default=0.02
        Minimum IV for Phase-1 participation.
    max_p1_depth : int or None, default=2
        Maximum Phase-1 levels.  n_bins^max_p1_depth = n Phase-2 roots.
        Default 2 gives 9 roots for n_bins=3.
    max_categories : int, default=20
        Top-k categories per categorical feature (Phase-1).
    numeric_binning : str, default='quantile'
        Phase-1 boundary strategy ('quantile' or 'optimal').
        'optimal' uses histogram-optimal recursive bisection.
    criterion : {'gini', 'entropy'}, default='gini'
    max_leaves : int, default=63
        Max leaves per Phase-2 root (best-leaf-first).
        63–127 gives near-GBT quality for a single tree.
    max_depth : int or None, default=None
        Additional depth limit inside Phase-2 subtrees.
    min_samples_split : int, default=4
    min_samples_leaf : int, default=2
    min_child_weight : float, default=1.0
        Minimum n·p·(1-p) per leaf.
    min_impurity_decrease : float, default=1e-7
    reg_lambda : float, default=1.0
        L2 leaf regularisation (shrinks toward base rate).
    max_features : int|float|'sqrt'|'log2'|None, default=None
    p2_exclude_p1_features : bool, default=True
        Phase-2 first tries features NOT in Phase-1 path.
    ccp_alpha : float, default=0.0
        Cost-complexity pruning parameter.
    subsample : float, default=1.0
    categorical_features : list[str] or None
    n_jobs : int, default=-1
    random_state : int or None
    """

    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(
        self,
        n_bins: int = 3,
        max_bins: int = 128,
        iv_bins: int = 10,
        min_importance: float = 0.02,
        max_p1_depth: Optional[int] = 2,
        max_categories: int = 20,
        numeric_binning: str = "optimal",
        criterion: str = "gini",
        max_leaves: int = 127,
        max_depth: Optional[int] = None,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        min_child_weight: float = 1.0,
        min_impurity_decrease: float = 1e-7,
        reg_lambda: float = 2.0,
        leaf_smooth: float = 10.0,
        leaf_model: bool = True,
        leaf_model_C: float = 1.0,
        leaf_model_min_samples: int = 20,
        max_features=None,
        p2_exclude_p1_features: bool = True,
        ccp_alpha: float = 0.0005,
        subsample: float = 1.0,
        categorical_features: Optional[List[str]] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ):
        self.n_bins = n_bins
        self.max_bins = max_bins
        self.iv_bins = iv_bins
        self.min_importance = min_importance
        self.max_p1_depth = max_p1_depth
        self.max_categories = max_categories
        self.numeric_binning = numeric_binning
        self.criterion = criterion
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.min_impurity_decrease = min_impurity_decrease
        self.reg_lambda = reg_lambda
        self.leaf_smooth = leaf_smooth
        self.leaf_model = leaf_model
        self.leaf_model_C = leaf_model_C
        self.leaf_model_min_samples = leaf_model_min_samples
        self.max_features = max_features
        self.p2_exclude_p1_features = p2_exclude_p1_features
        self.ccp_alpha = ccp_alpha
        self.subsample = subsample
        self.categorical_features = categorical_features
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ── Phase-1 boundary dispatch ─────────────────────────────────────────────

    def _p1_optimal_boundaries(self, idx, fidx, n_bins):
        if self.numeric_binning == "optimal":
            return super()._p1_optimal_boundaries(idx, fidx, n_bins)
        return self._p1_num_groups_quantile(idx, fidx)

    # ── Leaf logistic models ──────────────────────────────────────────────────

    def _fit_leaf_models(self, X_np: np.ndarray) -> None:
        """
        Phase 3 (Classifier) — fit LogisticRegression inside each leaf.

        After the histogram CART tree is built, every leaf with
        n >= leaf_model_min_samples and both classes present gets a
        LogisticRegression fitted on all numeric features.
        At predict time the tree\'s constant probability is replaced
        by the LogReg output, making the model piecewise-logistic.
        """
        if not self.leaf_model:
            return
        leaves = _IVTreeBase._leaves(self.tree_)
        if not leaves:
            return
        # Route training samples to leaves
        X_bins   = self._make_pred_bins(X_np)
        n        = len(X_np)
        leaf_idx: Dict[int, list] = {id(l): [] for l in leaves}
        stack = [(self.tree_, np.arange(n, dtype=np.int64))]
        while stack:
            node, idx = stack.pop()
            if not len(idx): continue
            if node.is_leaf:
                leaf_idx[id(node)].extend(idx.tolist()); continue
            self._route_for_fit(node, idx, X_np, X_bins, stack, leaf_idx)
        # Numeric feature matrix
        if not self._num_fidx:
            return
        X_num = self._X_num[:, self._num_fidx]   # (n_train, n_num)

        # Collect eligible leaf jobs first (avoids repeated condition checks)
        leaf_jobs = []
        for leaf in leaves:
            rows = np.array(leaf_idx.get(id(leaf), []), dtype=np.int64)
            if len(rows) < self.leaf_model_min_samples:
                continue
            Xi = X_num[rows]; yi = self._y_enc[rows]
            if len(np.unique(yi)) < 2:
                continue
            good = np.var(Xi, axis=0) > 1e-12
            if good.sum() == 0:
                continue
            leaf_jobs.append((leaf, Xi[:, good], yi, np.where(good)[0]))

        if not leaf_jobs:
            return

        def _fit_lr(Xi_g, yi_g, C):
            try:
                lr = LogisticRegression(C=C, max_iter=200,
                                        solver="liblinear", fit_intercept=True)
                lr.fit(Xi_g, yi_g)
                return lr
            except Exception:
                return None

        # Parallel when many jobs and user requests it; sequential otherwise
        if len(leaf_jobs) >= 8 and self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_fit_lr)(Xi_g, yi_g, self.leaf_model_C)
                for _, Xi_g, yi_g, _ in leaf_jobs
            )
        else:
            results = [_fit_lr(Xi_g, yi_g, self.leaf_model_C)
                       for _, Xi_g, yi_g, _ in leaf_jobs]

        for (leaf, _, _, good_cols), lr in zip(leaf_jobs, results):
            if lr is not None:
                leaf._lr_model = lr
                leaf._lr_cols  = good_cols

    def _route_for_fit(self, node, idx, X_np, X_bins, stack, leaf_idx):
        """Route samples through one non-leaf node (shared for fit + predict)."""
        st = node.split_type
        if st == "numeric_bins":
            try:
                col = X_np[idx, node.feature_idx].astype(np.float64)
            except (ValueError, TypeError):
                col = pd.to_numeric(
                    pd.Series(X_np[idx, node.feature_idx]), errors="coerce"
                ).values
            edges = node.split_values
            grp   = np.searchsorted(edges[1:-1], col, side="left")
            grp[np.isnan(col)] = 0
            grp   = np.clip(grp, 0, len(edges) - 2)
            valid = np.zeros(len(idx), dtype=bool)
            for key, child in node.children.items():
                m = grp == key
                if m.any(): valid |= m; stack.append((child, idx[m]))
            for i in idx[~valid]:
                leaf_idx.setdefault(id(node), []).append(i)
        elif st == "categorical":
            col    = self._encode_col_predict(
                X_np[idx, node.feature_idx], node.feature)
            top    = set(node.children) - {-1}
            mapped = np.where(np.isin(col, list(top)), col, np.int32(-1))
            valid  = np.zeros(len(idx), dtype=bool)
            for code, child in node.children.items():
                m = mapped == code
                if m.any(): valid |= m; stack.append((child, idx[m]))
        elif st == "num_hist":
            bincol = self._fidx_to_bincol.get(node.feature_idx)
            if bincol is None: return
            col = X_bins[idx, bincol]
            thr_bin = int(np.searchsorted(
                self._bin_thresholds[bincol], node.threshold, side="left"))
            try:
                raw_f    = X_np[idx, node.feature_idx].astype(np.float64)
                nan_mask = np.isnan(raw_f)
            except (ValueError, TypeError):
                nan_mask = np.zeros(len(idx), dtype=bool)
            lm = col <= thr_bin; lm[nan_mask] = node.nan_goes_left
            if lm.any():   stack.append((node.left,  idx[lm]))
            if (~lm).any(): stack.append((node.right, idx[~lm]))
        elif st == "cat_binary":
            col = self._encode_col_predict(
                X_np[idx, node.feature_idx], node.feature)
            lm  = np.isin(col, node.threshold)
            if lm.any():   stack.append((node.left,  idx[lm]))
            if (~lm).any(): stack.append((node.right, idx[~lm]))

    def _predict_with_leaf_models(
        self, X_np: np.ndarray, base_proba: np.ndarray
    ) -> np.ndarray:
        """Replace leaf probabilities with LogReg outputs where available."""
        if not self.leaf_model or not self._num_fidx:
            return base_proba
        out    = base_proba.copy()
        X_bins = self._make_pred_bins(X_np)
        n      = len(X_np)
        # Pre-extract numeric features
        X_num = np.full((n, len(self._num_fidx)), np.nan, dtype=np.float64)
        for j, fidx in enumerate(self._num_fidx):
            try:
                X_num[:, j] = X_np[:, fidx].astype(np.float64)
            except (ValueError, TypeError):
                X_num[:, j] = pd.to_numeric(
                    pd.Series(X_np[:, fidx]), errors="coerce"
                ).values
        stack = [(self.tree_, np.arange(n, dtype=np.int64))]
        while stack:
            node, idx = stack.pop()
            if not len(idx): continue
            if node.is_leaf:
                if hasattr(node, "_lr_model"):
                    Xi = X_num[idx][:, node._lr_cols]
                    p1 = node._lr_model.predict_proba(Xi)[:, 1]
                    out[idx, 1] = p1
                    out[idx, 0] = 1.0 - p1
                continue
            self._route_for_fit(node, idx, X_np, X_bins, stack, {})
        return out

    # ── Abstract implementations ──────────────────────────────────────────────

    @property
    def _importance_col_name(self): return "iv"

    def _smooth_leaf_val(self, child_val, n_child, parent_val):
        """CatBoost-style leaf smoothing: blend child toward parent."""
        alpha = self.leaf_smooth
        if alpha <= 0:
            return child_val
        w = n_child / (n_child + alpha)
        return w * child_val + (1.0 - w) * parent_val

    def _encode_y(self, y_raw):
        self.classes_ = unique_labels(y_raw)
        self._n_classes = len(self.classes_)
        if self._n_classes != 2:
            raise ValueError(f"Binary only. Got {self._n_classes}: {self.classes_}")
        self._cls2idx   = {c: i for i, c in enumerate(self.classes_)}
        self._base_rate = float((y_raw == self.classes_[1]).mean())
        return np.array([self._cls2idx[c] for c in y_raw], dtype=np.int32)

    def _compute_importance(self, col, y, is_cat):
        eps = 1e-9
        ev  = float((y==1).sum()); nev = float((y==0).sum())
        if ev < eps or nev < eps: return 0.0
        if is_cat:
            groups = col.astype(np.int32)
        else:
            col_f = col.astype(float)
            nm = np.isnan(col_f)
            if nm.all(): return 0.0
            col_f = col_f[~nm]; y = y[~nm]
            ev  = float((y==1).sum()); nev = float((y==0).sum())
            if ev < eps or nev < eps: return 0.0
            edges = np.unique(np.percentile(col_f, np.linspace(0,100,self.iv_bins+1)))
            if len(edges) < 2: return 0.0
            groups = np.digitize(col_f, edges[1:-1])
        ng = len(np.unique(groups)); alpha = 0.5
        iv = 0.0
        for g in np.unique(groups):
            m  = groups == g
            pe = ((y[m]==1).sum()+alpha)/(ev+alpha*ng)
            pn = ((y[m]==0).sum()+alpha)/(nev+alpha*ng)
            if pe < eps or pn < eps: continue
            iv += (pe - pn) * np.log(pe / pn)
        return float(max(iv, 0.0))

    def _impurity(self, y):
        if len(y) == 0: return 0.0
        p = float(y.sum()) / len(y)
        if self.criterion == "gini":
            return 2.0 * p * (1.0 - p)
        if p <= 0 or p >= 1: return 0.0
        return float(-p*np.log2(p) - (1-p)*np.log2(1-p))

    def _impurity_from_stats(self, stats, n):
        if n == 0: return 0.0
        p = float(stats[1]) / n
        if self.criterion == "gini": return 2.0 * p * (1.0 - p)
        if p <= 0 or p >= 1: return 0.0
        return float(-p*np.log2(p) - (1-p)*np.log2(1-p))

    def _leaf_value(self, y):
        n = len(y); lam = self.reg_lambda
        if n == 0:
            return np.array([1.0-self._base_rate, self._base_rate])
        n1 = float(y.sum())
        p1 = (n1 + lam*self._base_rate)         / (n + lam)
        p0 = (n-n1 + lam*(1-self._base_rate))   / (n + lam)
        return np.array([p0, p1])

    def _is_pure(self, y):
        s = int(y.sum()); return s==0 or s==len(y)

    def _effective_n(self, y):
        n = len(y)
        if n == 0: return 0.0
        p = float(y.sum())/n
        return n * p * (1.0-p)

    def _y_stat(self, y):
        out = np.empty((len(y), 2), dtype=np.float64)
        out[:, 0] = 1.0; out[:, 1] = y.astype(np.float64)
        return out

    def _compute_gains_vectorized(self, cum, total, nl, nr, n, pi):
        sl = cum[:, :-1, 1]
        sr = total[:, 1:2] - sl
        with np.errstate(divide="ignore", invalid="ignore"):
            p1l = np.where(nl > 0, sl/nl, 0.0)
            p1r = np.where(nr > 0, sr/nr, 0.0)
        if self.criterion == "gini":
            impl = 2.0*p1l*(1.0-p1l); impr = 2.0*p1r*(1.0-p1r)
        else:
            def _e(p):
                q = 1-p
                with np.errstate(divide="ignore", invalid="ignore"):
                    return (-np.where(p>0,p*np.log2(p),0.0)
                            -np.where(q>0,q*np.log2(q),0.0))
            impl = _e(p1l); impr = _e(p1r)
        return pi - (nl/n*impl + nr/n*impr)

    def _cat_split_order(self, y_sub, col_enc, cats):
        rates = np.array(
            [(y_sub[col_enc==c]==1).mean() if (col_enc==c).any() else 0.0
             for c in cats], dtype=float)
        return np.argsort(rates)

    def _format_leaf(self, val): return f"p={val[1]:.3f}"

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Fit the IV Decision Tree Classifier with M5'-style leaf logistic models."""
        self._fit_prepare(X, y)
        self._fit_tree()
        # Phase 3: M5' — reuse cached X_np, no double-parse
        self._fit_leaf_models(self._X_np_cache)
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)
        X_np = (X.to_numpy(dtype=object) if isinstance(X, pd.DataFrame)
                else np.asarray(X, dtype=object))
        vals = self._route(X_np, self._make_pred_bins(X_np))
        fb   = self._leaf_value(np.array([], dtype=np.int32))
        base = np.vstack([v if v is not None else fb for v in vals])
        return self._predict_with_leaf_models(X_np, base)

    def predict_log_proba(self, X) -> np.ndarray:
        """Return log-probabilities. Clips to avoid log(0)."""
        return np.log(np.clip(self.predict_proba(X), 1e-15, 1.0))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def iv_summary_(self):
        check_is_fitted(self); return self.importance_summary_


# ─────────────────────────────────────────────────────────────────────────────
# Regressor
# ─────────────────────────────────────────────────────────────────────────────

class IVDecisionTreeRegressor(_IVTreeBase, RegressorMixin):
    """
    RV-ordered multi-way Decision Tree Regressor.

    Phase-1  : RV-ranked multi-way splits (η² = variance explained by binning).
    Phase-2  : Per-root best-leaf-first histogram CART (MSE).

    Parameters  same as IVDecisionTreeClassifier.
    criterion : {'mse'}, default='mse'  (MAE via ccp_alpha post-pruning)
    min_importance : float, default=0.01
    """

    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    def __init__(
        self,
        n_bins: int = 2,
        max_bins: int = 128,
        iv_bins: int = 10,
        min_importance: float = 0.01,
        max_p1_depth: Optional[int] = 1,
        max_categories: int = 20,
        numeric_binning: str = "optimal",
        criterion: str = "mse",
        max_leaves: int = 31,
        max_depth: Optional[int] = None,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        min_child_weight: float = 1e-3,
        min_impurity_decrease: float = 1e-7,
        reg_lambda: float = 0.0,
        leaf_smooth: float = 5.0,
        leaf_model: bool = True,
        leaf_model_alpha: float = 0.1,
        leaf_model_min_samples: int = 10,
        max_features=None,
        p2_exclude_p1_features: bool = False,
        ccp_alpha: float = 0.0,
        subsample: float = 1.0,
        categorical_features: Optional[List[str]] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ):
        self.n_bins = n_bins
        self.max_bins = max_bins
        self.iv_bins = iv_bins
        self.min_importance = min_importance
        self.max_p1_depth = max_p1_depth
        self.max_categories = max_categories
        self.numeric_binning = numeric_binning
        self.criterion = criterion
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.min_impurity_decrease = min_impurity_decrease
        self.reg_lambda = reg_lambda
        self.leaf_smooth = leaf_smooth
        self.leaf_model = leaf_model
        self.leaf_model_alpha = leaf_model_alpha
        self.leaf_model_min_samples = leaf_model_min_samples
        self.max_features = max_features
        self.p2_exclude_p1_features = p2_exclude_p1_features
        self.ccp_alpha = ccp_alpha
        self.subsample = subsample
        self.categorical_features = categorical_features
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _p1_optimal_boundaries(self, idx, fidx, n_bins):
        if self.numeric_binning == "optimal":
            return super()._p1_optimal_boundaries(idx, fidx, n_bins)
        return self._p1_num_groups_quantile(idx, fidx)

    @property
    def _importance_col_name(self): return "rv"

    def _smooth_leaf_val(self, child_val, n_child, parent_val):
        """CatBoost-style leaf smoothing for regression."""
        alpha = self.leaf_smooth
        if alpha <= 0:
            return child_val
        w = n_child / (n_child + alpha)
        return w * child_val + (1.0 - w) * parent_val

    def _encode_y(self, y_raw):
        y = y_raw.astype(np.float64)
        self._y_mean = float(np.mean(y))
        return y

    def _compute_importance(self, col, y, is_cat):
        y   = y.astype(float)
        sst = float(np.var(y)*len(y))
        if sst < 1e-12: return 0.0
        if is_cat:
            groups = col.astype(np.int32)
        else:
            col_f = col.astype(float); nm = np.isnan(col_f)
            if nm.all(): return 0.0
            col_f = col_f[~nm]; y = y[~nm]
            sst = float(np.var(y)*len(y))
            if sst < 1e-12: return 0.0
            edges = np.unique(np.percentile(col_f, np.linspace(0,100,self.iv_bins+1)))
            if len(edges) < 2: return 0.0
            groups = np.digitize(col_f, edges[1:-1])
        ssw = sum(np.var(y[groups==g])*(groups==g).sum() for g in np.unique(groups))
        return float(max(0.0, 1.0 - ssw/sst))

    def _impurity(self, y):
        n = len(y)
        if n == 0: return 0.0
        y64 = y.astype(np.float64)
        s   = y64.sum(); s2 = float(np.dot(y64, y64))
        return s2/n - (s/n)**2

    def _impurity_from_stats(self, stats, n):
        if n == 0: return 0.0
        s = float(stats[1]); s2 = float(stats[2])
        return s2/n - (s/n)**2

    def _leaf_value(self, y):
        """
        L2-regularised mean, shrinking toward the global training mean.
        val = (Σy + λ·ȳ·n_eff) / (n + λ·n_eff)
        With n_eff=1 this becomes (Σy + λ·ȳ) / (n + λ), which at n→∞
        returns the raw mean and at n=0 returns ȳ.
        """
        n = len(y)
        lam = self.reg_lambda
        if n == 0:
            return self._y_mean
        if lam <= 0:
            return float(y.mean())
        # Shrink toward global mean (not 0)
        return float((y.sum() + lam * self._y_mean) / (n + lam))

    def _is_pure(self, y): return float(np.var(y)) < 1e-12
    def _effective_n(self, y): return float(len(y))

    def _y_stat(self, y):
        y64 = y.astype(np.float64)
        out = np.empty((len(y), 3), dtype=np.float64)
        out[:,0]=1.0; out[:,1]=y64; out[:,2]=y64*y64
        return out

    def _compute_gains_vectorized(self, cum, total, nl, nr, n, pi):
        sl  = cum[:,:-1,1];  sl2 = cum[:,:-1,2]
        sr  = total[:,1:2]-sl; sr2 = total[:,2:3]-sl2
        with np.errstate(divide="ignore", invalid="ignore"):
            mse_l = np.where(nl>0, sl2/nl-(sl/nl)**2, 0.0)
            mse_r = np.where(nr>0, sr2/nr-(sr/nr)**2, 0.0)
        return pi - (nl/n*mse_l + nr/n*mse_r)

    def _cat_split_order(self, y_sub, col_enc, cats):
        means = np.array(
            [float(y_sub[col_enc==c].mean()) if (col_enc==c).any() else 0.0
             for c in cats], dtype=float)
        return np.argsort(means)

    def _format_leaf(self, val): return f"{val:.5g}"

    def _fit_leaf_models(self, X_np: np.ndarray) -> None:
        """
        M5\'  —  Fit a Ridge model inside each leaf.

        After the tree is built, every leaf with n >= leaf_model_min_samples
        gets a Ridge(alpha=leaf_model_alpha) fitted on ALL numeric features.
        At predict time, leaf predictions are replaced by the Ridge output.

        This transforms the tree from a piecewise-constant to a
        piecewise-linear model, dramatically reducing bias on smooth targets.
        """
        if not self.leaf_model:
            return

        leaves = _IVTreeBase._leaves(self.tree_)
        if len(leaves) == 0:
            return

        # Route all training samples to their leaf
        X_bins = self._make_pred_bins(X_np)
        n      = len(X_np)
        leaf_idx: Dict[int, list] = {id(l): [] for l in leaves}

        # Vectorised routing to build per-leaf sample lists
        stack = [(self.tree_, np.arange(n, dtype=np.int64))]
        while stack:
            node, idx = stack.pop()
            if not len(idx): continue
            if node.is_leaf:
                leaf_idx[id(node)].extend(idx.tolist())
                continue
            st = node.split_type
            if st == "numeric_bins":
                try:
                    col = X_np[idx, node.feature_idx].astype(np.float64)
                except (ValueError, TypeError):
                    col = pd.to_numeric(pd.Series(X_np[idx, node.feature_idx]),
                                        errors="coerce").values
                edges = node.split_values
                grp = np.searchsorted(edges[1:-1], col, side="left")
                grp[np.isnan(col)] = 0
                grp = np.clip(grp, 0, len(edges) - 2)
                valid = np.zeros(len(idx), dtype=bool)
                for key, child in node.children.items():
                    m = grp == key
                    if m.any(): valid |= m; stack.append((child, idx[m]))
                for i in idx[~valid]: leaf_idx[id(node)].append(i)
            elif st == "categorical":
                col = self._encode_col_predict(X_np[idx, node.feature_idx], node.feature)
                top = set(node.children) - {-1}
                mapped = np.where(np.isin(col, list(top)), col, np.int32(-1))
                valid = np.zeros(len(idx), dtype=bool)
                for code, child in node.children.items():
                    m = mapped == code
                    if m.any(): valid |= m; stack.append((child, idx[m]))
                for i in idx[~valid]: leaf_idx[id(node)].append(i)
            elif st == "num_hist":
                bincol = self._fidx_to_bincol.get(node.feature_idx)
                if bincol is None:
                    leaf_idx[id(node)].extend(idx.tolist()); continue
                col = X_bins[idx, bincol]
                thr_bin = int(np.searchsorted(
                    self._bin_thresholds[bincol], node.threshold, side="left"))
                try:
                    raw_f = X_np[idx, node.feature_idx].astype(np.float64)
                    nan_m = np.isnan(raw_f)
                except (ValueError, TypeError):
                    nan_m = np.zeros(len(idx), dtype=bool)
                lm = col <= thr_bin; lm[nan_m] = node.nan_goes_left
                if lm.any():   stack.append((node.left,  idx[lm]))
                if (~lm).any(): stack.append((node.right, idx[~lm]))
            elif st == "cat_binary":
                col = self._encode_col_predict(X_np[idx, node.feature_idx], node.feature)
                lm = np.isin(col, node.threshold)
                if lm.any():   stack.append((node.left,  idx[lm]))
                if (~lm).any(): stack.append((node.right, idx[~lm]))

        # Build numeric feature matrix for leaf model fitting
        # Use only numeric features (categorical ones don't work with Ridge)
        num_col_indices = self._num_fidx   # original feature indices
        if not num_col_indices:
            return   # no numeric features → skip

        # Extract float matrix from cached _X_num (already built at fit time)
        X_num_train = self._X_num[:, num_col_indices]  # (n_train, n_num_feats)

        # Fit Ridge per leaf
        self._leaf_ridge_: Dict[int, Ridge] = {}
        min_samp = self.leaf_model_min_samples
        p = X_num_train.shape[1]

        for leaf in leaves:
            row_indices = leaf_idx.get(id(leaf), [])
            n_leaf = len(row_indices)
            if n_leaf < max(min_samp, p + 1):
                continue   # too few samples: keep tree mean
            Xi = X_num_train[row_indices]
            yi = self._y_enc[row_indices]
            # Check for degenerate columns (constant features in this leaf)
            col_var = np.var(Xi, axis=0)
            good_cols = col_var > 1e-12
            if good_cols.sum() == 0:
                continue
            Xi_g = Xi[:, good_cols]
            try:
                ridge = Ridge(alpha=self.leaf_model_alpha,
                              fit_intercept=True, max_iter=200)
                ridge.fit(Xi_g, yi)
                # Store with mask so predict knows which cols to use
                leaf._ridge_model  = ridge
                leaf._ridge_cols   = np.where(good_cols)[0]   # indices into num_col_indices
                self._leaf_ridge_[id(leaf)] = ridge
            except Exception:
                pass   # fallback to tree mean

    def _predict_with_leaf_models(
        self, X_np: np.ndarray, base_preds: np.ndarray
    ) -> np.ndarray:
        """Replace leaf predictions with Ridge model outputs where available."""
        if not self.leaf_model or not hasattr(self, '_leaf_ridge_') or not self._leaf_ridge_:
            return base_preds

        out     = base_preds.copy()
        X_bins  = self._make_pred_bins(X_np)
        n       = len(X_np)
        stack   = [(self.tree_, np.arange(n, dtype=np.int64))]

        # Pre-extract numeric features for prediction
        X_num_pred = np.full((n, len(self._num_fidx)), np.nan, dtype=np.float64)
        for j, fidx in enumerate(self._num_fidx):
            col = X_np[:, fidx]
            try:
                X_num_pred[:, j] = col.astype(np.float64)
            except (ValueError, TypeError):
                X_num_pred[:, j] = pd.to_numeric(pd.Series(col), errors="coerce").values

        while stack:
            node, idx = stack.pop()
            if not len(idx): continue
            if node.is_leaf:
                if hasattr(node, "_ridge_model"):
                    Xi = X_num_pred[idx][:, node._ridge_cols]
                    out[idx] = node._ridge_model.predict(Xi)
                continue
            st = node.split_type
            if st == "numeric_bins":
                try:
                    col = X_np[idx, node.feature_idx].astype(np.float64)
                except (ValueError, TypeError):
                    col = pd.to_numeric(pd.Series(X_np[idx, node.feature_idx]),
                                        errors="coerce").values
                edges = node.split_values
                grp = np.searchsorted(edges[1:-1], col, side="left")
                grp[np.isnan(col)] = 0; grp = np.clip(grp, 0, len(edges)-2)
                valid = np.zeros(len(idx), dtype=bool)
                for key, child in node.children.items():
                    m = grp == key
                    if m.any(): valid |= m; stack.append((child, idx[m]))
                for i in idx[~valid]: out[i] = base_preds[i]
            elif st == "categorical":
                col = self._encode_col_predict(X_np[idx, node.feature_idx], node.feature)
                top = set(node.children) - {-1}
                mapped = np.where(np.isin(col, list(top)), col, np.int32(-1))
                valid = np.zeros(len(idx), dtype=bool)
                for code, child in node.children.items():
                    m = mapped == code
                    if m.any(): valid |= m; stack.append((child, idx[m]))
                for i in idx[~valid]: out[i] = base_preds[i]
            elif st == "num_hist":
                bincol = self._fidx_to_bincol.get(node.feature_idx)
                if bincol is None:
                    continue
                col = X_bins[idx, bincol]
                thr_bin = int(np.searchsorted(
                    self._bin_thresholds[bincol], node.threshold, side="left"))
                try:
                    raw_f = X_np[idx, node.feature_idx].astype(np.float64)
                    nan_m = np.isnan(raw_f)
                except (ValueError, TypeError):
                    nan_m = np.zeros(len(idx), dtype=bool)
                lm = col <= thr_bin; lm[nan_m] = node.nan_goes_left
                if lm.any():   stack.append((node.left,  idx[lm]))
                if (~lm).any(): stack.append((node.right, idx[~lm]))
            elif st == "cat_binary":
                col = self._encode_col_predict(X_np[idx, node.feature_idx], node.feature)
                lm = np.isin(col, node.threshold)
                if lm.any():   stack.append((node.left,  idx[lm]))
                if (~lm).any(): stack.append((node.right, idx[~lm]))
        return out

    def fit(self, X, y):
        """Fit the RV Decision Tree Regressor with M5\'-style leaf linear models."""
        self._fit_prepare(X, y)
        self._fit_tree()
        # Phase 3: M5\' — reuse cached X_np, no double-parse
        self._fit_leaf_models(self._X_np_cache)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        X_np = (X.to_numpy(dtype=object) if isinstance(X, pd.DataFrame)
                else np.asarray(X, dtype=object))
        base = np.array(
            [v if v is not None else self._y_mean
             for v in self._route(X_np, self._make_pred_bins(X_np))],
            dtype=float,
        )
        return self._predict_with_leaf_models(X_np, base)

    @property
    def rv_summary_(self):
        check_is_fitted(self); return self.importance_summary_


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark  vs  sklearn CART  +  HistGradientBoosting
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import roc_auc_score, r2_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)

    rng = np.random.default_rng(42)
    N, CV = 30_000, 5

    def make_df(X_num, y, with_cat_signal=True):
        """Add a noisy categorical signal (useful for classifier) + noise cat."""
        df = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(X_num.shape[1])])
        if with_cat_signal:
            cats = np.where(y > np.median(y), "high", "low")
            flip = rng.random(len(y)) < 0.25
            cats[flip] = np.where(cats[flip] == "high", "low", "high")
            df["cat_signal"] = cats
        df["cat_noise"] = rng.choice(["a", "b", "c", "d"], len(y))
        return df

    def run(name, mdl, Xtr, Xte, ytr, yte, scoring):
        t0   = time.perf_counter()
        cv   = cross_val_score(mdl, Xtr, ytr, cv=CV, scoring=scoring, n_jobs=1)
        t_cv = time.perf_counter() - t0
        mdl.fit(Xtr, ytr)
        t_fit = time.perf_counter() - t0 - t_cv
        sc = (roc_auc_score(yte, mdl.predict_proba(Xte)[:, 1])
              if scoring == "roc_auc" else r2_score(yte, mdl.predict(Xte)))
        print(f"  {name:<46}  CV={cv.mean():.4f}±{cv.std():.4f}"
              f"  test={sc:.4f}  fit={t_fit:.2f}s  CVt={t_cv:.1f}s")
        return mdl, cv.mean()

    # ── CLASSIFIER ────────────────────────────────────────────────────────────
    print("=" * 80)
    print(f"CLASSIFIER  5-fold ROC-AUC  n={N}")
    print("  sklearn / HistGBT : numeric features only (no native categorical)")
    print("  IV Tree v6        : numeric + categorical features")
    print("=" * 80)
    X_num, y_c = make_classification(
        n_samples=N, n_features=12, n_informative=7, n_redundant=3, random_state=42
    )
    df_c  = make_df(X_num, y_c, with_cat_signal=True)
    num_c = [c for c in df_c.columns if c.startswith("num_")]
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
        df_c, y_c, test_size=0.25, stratify=y_c, random_state=42
    )

    _, cart_c = run("sklearn CART  depth=8  min_leaf=10  num_only",
        DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42),
        Xtr_c[num_c], Xte_c[num_c], ytr_c, yte_c, "roc_auc")
    _, hgbt_c = run("HistGBT  100 trees  31 leaves  num_only",
        HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, random_state=42),
        Xtr_c[num_c], Xte_c[num_c], ytr_c, yte_c, "roc_auc")
    iv_c, ivc_cv = run("IV Tree v6  (num+cat)  [defaults]",
        IVDecisionTreeClassifier(random_state=42),
        Xtr_c, Xte_c, ytr_c, yte_c, "roc_auc")
    print(f"  IV Tree vs CART = {ivc_cv-cart_c:+.4f}   vs HistGBT = {ivc_cv-hgbt_c:+.4f}  (single tree vs 100 trees)")
    print(f"  Phase-1 order  : {iv_c.feature_order_}")
    print("  IV summary (top 6):\n" + iv_c.iv_summary_.head(6).to_string(index=False))

    # ── REGRESSOR ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"REGRESSOR   5-fold R²  n={N}")
    print("=" * 80)
    X_num2, y_r = make_regression(
        n_samples=N, n_features=12, n_informative=7, noise=0.3, random_state=42
    )
    df_r  = make_df(X_num2, y_r, with_cat_signal=False)
    num_r = [c for c in df_r.columns if c.startswith("num_")]
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(df_r, y_r, test_size=0.25, random_state=42)

    _, cart_r = run("sklearn CART  depth=8  min_leaf=10  num_only",
        DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42),
        Xtr_r[num_r], Xte_r[num_r], ytr_r, yte_r, "r2")
    _, hgbt_r = run("HistGBT  100 trees  31 leaves  num_only",
        HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=31, random_state=42),
        Xtr_r[num_r], Xte_r[num_r], ytr_r, yte_r, "r2")
    iv_r, ivr_cv = run("IV Tree v6  (num+cat)  [defaults]",
        IVDecisionTreeRegressor(random_state=42),
        Xtr_r, Xte_r, ytr_r, yte_r, "r2")
    print(f"  IV Tree vs CART = {ivr_cv-cart_r:+.4f}   vs HistGBT = {ivr_cv-hgbt_r:+.4f}  (single tree vs 100 trees)")
    print(f"  Phase-1 order  : {iv_r.feature_order_}")
    print("  RV summary (top 6):\n" + iv_r.rv_summary_.head(6).to_string(index=False))
