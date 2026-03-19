"""
drift_detector.py  ─  Ultra-fast raw data drift analyzer
=========================================================
Détecte le drift univarié, multivarié et structurel entre
un DataFrame de référence (df1) et un DataFrame de production (df2).

Usage rapide
------------
    from drift_detector import analyze_drift, print_drift_report

    report = analyze_drift(df1, df2)
    print_drift_report(report)

    # Accès programmatique
    report["univariate"]     # DataFrame trié par drift_score desc
    report["summary"]        # dict concis
    report["multivariate"]   # AUC classifier
    report["correlation"]    # drift structure de corrélations
    report["schema"]         # colonnes ajoutées/supprimées/dtype changé

Dépendances : numpy, pandas, scipy, joblib, sklearn (multivarié optionnel)
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  1. INFÉRENCE DU TYPE SÉMANTIQUE DE COLONNE
# ══════════════════════════════════════════════════════════════════════════════

_BOOL_VALUES = frozenset(
    {"true", "false", "yes", "no", "oui", "non", "1", "0", "t", "f", "y", "n"}
)


def _infer_col_type(
    col: pd.Series,
    id_card_thresh: float = 0.90,
    cat_card_thresh: float = 0.05,
    max_cat: int = 200,
) -> str:
    """Déduit le type sémantique d'une colonne brute."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime"

    n_total = len(col.dropna())
    if n_total == 0:
        return "all_null"

    n_unique = col.nunique(dropna=True)
    dtype = col.dtype

    # ── booléen ──────────────────────────────────────────────────────────────
    if dtype == bool:
        return "boolean"
    if n_unique <= 2:
        return "boolean"
    if dtype == object and n_unique <= 4:
        sample_low = col.dropna().astype(str).str.lower()
        if sample_low.isin(_BOOL_VALUES).all():
            return "boolean"

    # ── numérique ────────────────────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(col):
        card_ratio = n_unique / n_total
        if card_ratio > id_card_thresh and n_unique > 200:
            return "id_numeric"
        return "numeric"

    # ── object / string ──────────────────────────────────────────────────────
    if dtype == object:
        card_ratio = n_unique / n_total
        sample = col.dropna().astype(str).head(200)

        # Heuristique ID : haute cardinalité, pas d'espaces, longueur homogène
        avg_spaces = sample.str.count(" ").mean()
        avg_len = sample.str.len().mean()
        len_std = sample.str.len().std()
        if (
            card_ratio > id_card_thresh
            and n_unique > 100
            and avg_spaces < 0.1
            and (len_std < 3 or avg_len < 6)
        ):
            return "id_string"

        # Texte libre : espaces nombreux
        if avg_spaces > 2:
            return "text"

        # Catégorielle
        if n_unique <= max_cat or card_ratio <= cat_card_thresh:
            return "categorical"

        return "high_card_string"

    return "other"


# ══════════════════════════════════════════════════════════════════════════════
#  2. MÉTRIQUES DRIFT ÉLÉMENTAIRES
# ══════════════════════════════════════════════════════════════════════════════

def _psi_numeric(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index pour variable numérique."""
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 5 or len(cur) < 5:
        return np.nan
    breaks = np.unique(np.percentile(ref, np.linspace(0, 100, n_bins + 1)))
    if len(breaks) < 3:
        return np.nan
    r = np.clip(np.histogram(ref, bins=breaks)[0] / len(ref), 1e-6, None)
    c = np.clip(np.histogram(cur, bins=breaks)[0] / len(cur), 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    """PSI pour variable catégorielle / booléenne."""
    ref = ref.dropna().astype(str)
    cur = cur.dropna().astype(str)
    if len(ref) < 5 or len(cur) < 5:
        return np.nan
    cats = list(set(ref.unique()) | set(cur.unique()))
    r = np.clip(ref.value_counts(normalize=True).reindex(cats, fill_value=0).values, 1e-6, None)
    c = np.clip(cur.value_counts(normalize=True).reindex(cats, fill_value=0).values, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _wasserstein_normalized(ref: np.ndarray, cur: np.ndarray) -> float:
    """Wasserstein distance normalisée par l'écart-type de référence."""
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 5 or len(cur) < 5:
        return np.nan
    w = stats.wasserstein_distance(ref, cur)
    scale = np.std(ref)
    return float(w / scale) if scale > 0 else float(w)


# ══════════════════════════════════════════════════════════════════════════════
#  3. ANALYSE PAR COLONNE (exécutée en parallèle)
# ══════════════════════════════════════════════════════════════════════════════

def _analyze_column(
    col_name: str,
    s1: pd.Series,
    s2: pd.Series,
    col_type: str,
    alpha: float,
    psi_bins: int,
) -> dict:
    """Retourne un dict de métriques drift pour une colonne."""
    null_ref = float(s1.isna().mean())
    null_cur = float(s2.isna().mean())
    null_delta = null_cur - null_ref

    base = {
        "column": col_name,
        "type": col_type,
        "null_rate_ref": round(null_ref, 4),
        "null_rate_cur": round(null_cur, 4),
        "null_delta": round(null_delta, 4),
        "null_drift": abs(null_delta) >= 0.05,
        "drift_flag": False,
        "drift_score": 0.0,
    }

    # ── colonnes à ignorer ───────────────────────────────────────────────────
    if col_type.startswith("id") or col_type in ("all_null", "other"):
        base["note"] = "skipped"
        return base

    # ── numérique ────────────────────────────────────────────────────────────
    if col_type == "numeric":
        v1 = s1.dropna().values.astype(np.float64)
        v2 = s2.dropna().values.astype(np.float64)
        if len(v1) < 10 or len(v2) < 10:
            base["note"] = "insufficient data"
            return base

        ks_stat, ks_p   = stats.ks_2samp(v1, v2)
        psi             = _psi_numeric(v1, v2, n_bins=psi_bins)
        wass            = _wasserstein_normalized(v1, v2)
        std1            = float(np.std(v1))
        mean_shift_z    = (np.mean(v2) - np.mean(v1)) / (std1 + 1e-9)
        std_ratio       = np.std(v2) / (std1 + 1e-9)
        p5_d            = np.percentile(v2, 5)  - np.percentile(v1, 5)
        p50_d           = np.percentile(v2, 50) - np.percentile(v1, 50)
        p95_d           = np.percentile(v2, 95) - np.percentile(v1, 95)

        base.update({
            "ks_stat":         round(ks_stat, 4),
            "ks_pvalue":       round(ks_p, 6),
            "psi":             round(psi, 4) if not np.isnan(psi) else np.nan,
            "wasserstein":     round(wass, 4) if not np.isnan(wass) else np.nan,
            "mean_ref":        round(float(np.mean(v1)), 6),
            "mean_cur":        round(float(np.mean(v2)), 6),
            "mean_shift_z":    round(mean_shift_z, 4),
            "std_ratio":       round(std_ratio, 4),
            "p5_delta":        round(p5_d, 4),
            "p50_delta":       round(p50_d, 4),
            "p95_delta":       round(p95_d, 4),
        })

        score = 0.0
        if ks_p < alpha:               score += 0.35
        if not np.isnan(psi):
            if psi > 0.25:             score += 0.40
            elif psi > 0.10:           score += 0.20
        if abs(mean_shift_z) > 3:      score += 0.20
        elif abs(mean_shift_z) > 1.5:  score += 0.10
        if std_ratio < 0.5 or std_ratio > 2.0:
                                       score += 0.10

    # ── catégorielle / booléenne / high_card_string ──────────────────────────
    elif col_type in ("categorical", "boolean", "high_card_string"):
        v1 = s1.dropna().astype(str)
        v2 = s2.dropna().astype(str)
        if len(v1) < 5 or len(v2) < 5:
            base["note"] = "insufficient data"
            return base

        cats_ref = set(v1.unique())
        cats_cur = set(v2.unique())
        new_cats     = cats_cur - cats_ref
        missing_cats = cats_ref - cats_cur

        all_cats = list(cats_ref | cats_cur)
        obs1 = v1.value_counts().reindex(all_cats, fill_value=0).values
        obs2 = v2.value_counts().reindex(all_cats, fill_value=0).values
        try:
            chi2_stat, chi2_p, *_ = stats.chi2_contingency(np.vstack([obs1, obs2]))
        except Exception:
            chi2_stat, chi2_p = np.nan, np.nan

        psi = _psi_categorical(s1, s2)

        # Top catégories les plus déplacées
        freq1 = v1.value_counts(normalize=True)
        freq2 = v2.value_counts(normalize=True)
        all_freq = freq1.reindex(all_cats, fill_value=0).subtract(
                   freq2.reindex(all_cats, fill_value=0), fill_value=0)
        top_shifted = all_freq.abs().nlargest(5).index.tolist()

        base.update({
            "chi2_stat":              round(chi2_stat, 4) if not np.isnan(chi2_stat) else np.nan,
            "chi2_pvalue":            round(chi2_p,    6) if not np.isnan(chi2_p)    else np.nan,
            "psi":                    round(psi, 4) if not np.isnan(psi) else np.nan,
            "n_cat_ref":              len(cats_ref),
            "n_cat_cur":              len(cats_cur),
            "new_categories_n":       len(new_cats),
            "missing_categories_n":   len(missing_cats),
            "new_categories_ex":      sorted(new_cats)[:5],
            "missing_categories_ex":  sorted(missing_cats)[:5],
            "top_shifted_categories": top_shifted,
        })

        score = 0.0
        if not np.isnan(chi2_p) and chi2_p < alpha:   score += 0.35
        if not np.isnan(psi):
            if psi > 0.25:                             score += 0.35
            elif psi > 0.10:                           score += 0.20
        score += min(0.20, len(new_cats)     / 10.0)
        score += min(0.10, len(missing_cats) / 10.0)

    # ── texte libre ──────────────────────────────────────────────────────────
    elif col_type == "text":
        len1 = s1.dropna().astype(str).str.len().values
        len2 = s2.dropna().astype(str).str.len().values
        if len(len1) < 5 or len(len2) < 5:
            base["note"] = "insufficient data"
            return base

        ks_stat, ks_p = stats.ks_2samp(len1, len2)
        base.update({
            "avg_len_ref":   round(float(np.mean(len1)), 2),
            "avg_len_cur":   round(float(np.mean(len2)), 2),
            "ks_stat_len":   round(ks_stat, 4),
            "ks_pvalue_len": round(ks_p, 6),
        })
        score = 0.40 if ks_p < alpha else 0.0

    # ── datetime ─────────────────────────────────────────────────────────────
    elif col_type == "datetime":
        ts1 = pd.to_datetime(s1, errors="coerce").dropna().view(np.int64)
        ts2 = pd.to_datetime(s2, errors="coerce").dropna().view(np.int64)
        if len(ts1) < 5 or len(ts2) < 5:
            base["note"] = "insufficient data"
            return base

        ks_stat, ks_p = stats.ks_2samp(ts1, ts2)
        base.update({
            "ks_stat":   round(ks_stat, 4),
            "ks_pvalue": round(ks_p, 6),
            "min_ref":   str(pd.to_datetime(s1, errors="coerce").min()),
            "min_cur":   str(pd.to_datetime(s2, errors="coerce").min()),
            "max_ref":   str(pd.to_datetime(s1, errors="coerce").max()),
            "max_cur":   str(pd.to_datetime(s2, errors="coerce").max()),
        })
        score = 0.60 if ks_p < alpha else 0.0

    else:
        score = 0.0

    # Bonus null drift
    if abs(null_delta) >= 0.05:
        score += 0.10
    if abs(null_delta) >= 0.20:
        score += 0.10

    base["drift_score"] = round(min(score, 1.0), 4)
    base["drift_flag"]  = base["drift_score"] > 0.15 or base["null_drift"]

    return base


# ══════════════════════════════════════════════════════════════════════════════
#  4. DRIFT MULTIVARIÉ  (classifier df1 vs df2)
# ══════════════════════════════════════════════════════════════════════════════

def _multivariate_drift(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    numeric_cols: list[str],
    n_jobs: int,
    n_sample: int = 50_000,
) -> dict:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return {"status": "skipped", "reason": "sklearn non disponible"}

    cols = [c for c in numeric_cols if c in df1.columns and c in df2.columns]
    if len(cols) < 2:
        return {"status": "skipped", "reason": "pas assez de colonnes numériques communes"}

    cols = cols[: min(50, len(cols))]
    n = min(n_sample, len(df1), len(df2))

    rng = np.random.default_rng(42)
    idx1 = rng.choice(len(df1), n, replace=False)
    idx2 = rng.choice(len(df2), n, replace=False)

    X1 = df1.iloc[idx1][cols].fillna(-9999).values
    X2 = df2.iloc[idx2][cols].fillna(-9999).values
    X  = np.vstack([X1, X2])
    y  = np.array([0] * n + [1] * n, dtype=np.int8)

    clf = RandomForestClassifier(
        n_estimators=80, max_depth=5, min_samples_leaf=20,
        n_jobs=n_jobs, random_state=42
    )
    aucs = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
    auc  = float(np.mean(aucs))

    # Feature importances → colonnes qui séparent le mieux df1 / df2
    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=cols).nlargest(10)

    severity = (
        "CRITICAL" if auc > 0.80
        else "HIGH"   if auc > 0.70
        else "MEDIUM" if auc > 0.60
        else "LOW"    if auc > 0.55
        else "NONE"
    )
    return {
        "method":        "RandomForest classifier (label: df1=0, df2=1)",
        "roc_auc":       round(auc, 4),
        "severity":      severity,
        "n_cols_used":   len(cols),
        "n_samples":     n,
        "top_separator_features": imp.round(4).to_dict(),
        "interpretation": (
            f"AUC={auc:.3f} → le modèle distingue df1/df2 "
            + ("FACILEMENT → drift majeur" if auc > 0.75
               else "MODÉRÉMENT → drift notable" if auc > 0.60
               else "DIFFICILEMENT → faible drift global")
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  5. DRIFT STRUCTURE DE CORRÉLATION
# ══════════════════════════════════════════════════════════════════════════════

def _correlation_drift(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    numeric_cols: list[str],
) -> dict:
    cols = [c for c in numeric_cols if c in df1.columns and c in df2.columns]
    if len(cols) < 2:
        return {"status": "skipped", "reason": "pas assez de colonnes numériques"}

    cols = cols[: min(40, len(cols))]
    corr1 = df1[cols].corr().values
    corr2 = df2[cols].corr().values

    with np.errstate(invalid="ignore"):
        diff = np.abs(corr1 - corr2)

    upper = np.triu(np.ones_like(diff, dtype=bool), k=1)
    nan_mask  = np.isnan(diff)                       # 2D, même shape que diff
    diff_vals = diff[upper & ~nan_mask]              # les deux masques sont 2D → OK

    if len(diff_vals) == 0:
        return {"status": "skipped", "reason": "corrélations toutes NaN"}

    mean_delta = float(np.mean(diff_vals))
    max_delta  = float(np.max(diff_vals))

    # Top paires les plus driftées
    diff_masked = np.where(upper, diff, 0)
    flat_top = np.argsort(diff_masked.ravel())[::-1]
    top_pairs = []
    for idx in flat_top:
        i, j = divmod(idx, len(cols))
        if not upper[i, j]:
            continue
        if np.isnan(corr1[i, j]) or np.isnan(corr2[i, j]):
            continue
        top_pairs.append({
            "col_a":    cols[i],
            "col_b":    cols[j],
            "corr_ref": round(corr1[i, j], 4),
            "corr_cur": round(corr2[i, j], 4),
            "delta":    round(diff[i, j], 4),
        })
        if len(top_pairs) >= 8:
            break

    severity = (
        "HIGH"   if mean_delta > 0.20
        else "MEDIUM" if mean_delta > 0.10
        else "LOW"
    )
    return {
        "n_cols_used":        len(cols),
        "mean_abs_corr_delta": round(mean_delta, 4),
        "max_abs_corr_delta":  round(max_delta, 4),
        "severity":           severity,
        "top_drifted_pairs":  top_pairs,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  6. POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def analyze_drift(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    sample_size: Optional[int] = 100_000,
    n_jobs: int = -1,
    psi_bins: int = 10,
    alpha: float = 0.05,
    id_card_thresh: float = 0.90,
    cat_card_thresh: float = 0.05,
    max_categories: int = 200,
    skip_types: tuple[str, ...] = ("id_string", "id_numeric", "all_null", "other"),
    skip_multivariate: bool = False,
    skip_correlation: bool = False,
) -> dict:
    """
    Analyse ultra-rapide du drift entre df1 (référence) et df2 (production).

    Paramètres
    ----------
    df1, df2          : DataFrames à comparer (données raw, avant nettoyage)
    sample_size       : échantillonnage max par DF (None = tout utiliser)
    n_jobs            : parallélisme joblib (-1 = tous les CPU)
    psi_bins          : bins PSI pour variables numériques
    alpha             : seuil de signification statistique
    id_card_thresh    : ratio cardinalité → détection colonne ID
    cat_card_thresh   : ratio cardinalité → détection catégorielle
    max_categories    : seuil max de catégories uniques
    skip_types        : types sémantiques à exclure de l'analyse
    skip_multivariate : désactiver l'analyse multivarié (plus rapide)
    skip_correlation  : désactiver l'analyse de corrélation

    Retourne
    --------
    dict avec clés : summary, schema, univariate, multivariate, correlation, col_types
    """
    t0 = time.perf_counter()

    # ── 1. Drift de schéma ───────────────────────────────────────────────────
    cols_ref = set(df1.columns)
    cols_cur = set(df2.columns)
    added    = sorted(cols_cur - cols_ref)
    removed  = sorted(cols_ref - cols_cur)
    common   = cols_ref & cols_cur

    dtype_changes = {
        c: {"ref": str(df1[c].dtype), "cur": str(df2[c].dtype)}
        for c in common if df1[c].dtype != df2[c].dtype
    }

    schema = {
        "ref_shape":        df1.shape,
        "cur_shape":        df2.shape,
        "row_delta_pct":    round((df2.shape[0] - df1.shape[0]) / max(df1.shape[0], 1) * 100, 2),
        "added_columns":    added,
        "removed_columns":  removed,
        "dtype_changes":    dtype_changes,
        "n_common_columns": len(common),
    }

    # ── 2. Échantillonnage ───────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    def _sample(df: pd.DataFrame) -> pd.DataFrame:
        if sample_size and len(df) > sample_size:
            idx = rng.choice(len(df), sample_size, replace=False)
            return df.iloc[idx].reset_index(drop=True)
        return df

    df1s, df2s = _sample(df1), _sample(df2)

    # ── 3. Inférence des types ───────────────────────────────────────────────
    col_types: dict[str, str] = {
        c: _infer_col_type(df1s[c], id_card_thresh, cat_card_thresh, max_categories)
        for c in common
    }

    cols_to_analyze = [c for c in common if col_types[c] not in skip_types]

    # ── 4. Analyse univariée (parallèle, backend threading) ──────────────────
    def _safe(col: str) -> dict:
        try:
            return _analyze_column(
                col, df1s[col], df2s[col], col_types[col], alpha, psi_bins
            )
        except Exception as exc:
            return {
                "column": col, "type": col_types[col],
                "drift_flag": False, "drift_score": np.nan,
                "error": str(exc),
            }

    raw_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_safe)(c) for c in cols_to_analyze
    )

    univariate = (
        pd.DataFrame(raw_results)
        .sort_values("drift_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    # ── 5. Drift multivarié ──────────────────────────────────────────────────
    numeric_cols = [c for c in cols_to_analyze if col_types[c] == "numeric"]
    multivariate = (
        _multivariate_drift(df1s, df2s, numeric_cols, n_jobs)
        if not skip_multivariate
        else {"status": "skipped", "reason": "skip_multivariate=True"}
    )

    # ── 6. Drift de corrélations ─────────────────────────────────────────────
    correlation = (
        _correlation_drift(df1s, df2s, numeric_cols)
        if not skip_correlation
        else {"status": "skipped", "reason": "skip_correlation=True"}
    )

    # ── 7. Résumé ────────────────────────────────────────────────────────────
    n_analyzed = len(cols_to_analyze)
    drifted_mask = univariate["drift_flag"] == True
    n_drifted    = int(drifted_mask.sum())
    pct_drifted  = round(n_drifted / max(n_analyzed, 1) * 100, 1)

    null_mask = univariate.get("null_drift", pd.Series(False, index=univariate.index))
    n_null_drift = int((null_mask == True).sum())

    mv_auc = multivariate.get("roc_auc", 0.5)
    overall = (
        "CRITICAL" if pct_drifted > 30 or mv_auc > 0.80
        else "HIGH"   if pct_drifted > 15 or mv_auc > 0.70
        else "MEDIUM" if n_drifted > 0     or mv_auc > 0.60
        else "LOW"
    )

    summary = {
        "overall_severity":       overall,
        "elapsed_sec":            round(time.perf_counter() - t0, 2),
        "n_columns_analyzed":     n_analyzed,
        "n_columns_drifted":      n_drifted,
        "pct_columns_drifted":    pct_drifted,
        "n_columns_null_drift":   n_null_drift,
        "n_schema_issues":        len(added) + len(removed) + len(dtype_changes),
        "multivariate_severity":  multivariate.get("severity", "N/A"),
        "multivariate_auc":       multivariate.get("roc_auc",  np.nan),
        "correlation_severity":   correlation.get("severity",  "N/A"),
        "top_drifted_columns":    univariate.loc[drifted_mask, "column"].head(15).tolist(),
    }

    return {
        "summary":     summary,
        "schema":      schema,
        "univariate":  univariate,
        "multivariate": multivariate,
        "correlation": correlation,
        "col_types":   pd.Series(col_types).value_counts().to_dict(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  7. RAPPORT TEXTUEL CONCIS
# ══════════════════════════════════════════════════════════════════════════════

_SEV_ICON = {
    "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡",
    "LOW": "🟢", "NONE": "🟢", "N/A": "⚪",
}


def print_drift_report(report: dict, top_n: int = 25) -> None:
    """Affiche un rapport de drift concis et lisible."""
    s  = report["summary"]
    sc = report["schema"]
    mv = report["multivariate"]
    cr = report["correlation"]
    uv = report["univariate"]

    sev_icon = _SEV_ICON.get(s["overall_severity"], "⚪")

    print("═" * 72)
    print(f"  RAPPORT DE DRIFT   {sev_icon} Sévérité globale : {s['overall_severity']}")
    print(f"  Calculé en {s['elapsed_sec']}s  |  "
          f"Ref {sc['ref_shape']} → Prod {sc['cur_shape']} "
          f"(lignes {sc['row_delta_pct']:+.1f}%)")
    print("═" * 72)

    # ── Schéma ───────────────────────────────────────────────────────────────
    print("\n📐  SCHÉMA")
    if sc["added_columns"]:
        print(f"  ➕ Colonnes ajoutées   : {sc['added_columns']}")
    if sc["removed_columns"]:
        print(f"  ➖ Colonnes supprimées : {sc['removed_columns']}")
    if sc["dtype_changes"]:
        print(f"  ⚠️   Changements dtype ({len(sc['dtype_changes'])}) :")
        for col, d in list(sc["dtype_changes"].items())[:8]:
            print(f"      {col}: {d['ref']} → {d['cur']}")
    if not (sc["added_columns"] or sc["removed_columns"] or sc["dtype_changes"]):
        print("  ✅ Pas de changement de schéma")

    print(f"  Types détectés : {report['col_types']}")

    # ── Univarié ─────────────────────────────────────────────────────────────
    print(f"\n📊  UNIVARIÉ  —  {s['n_columns_drifted']}/{s['n_columns_analyzed']} cols driftées "
          f"({s['pct_columns_drifted']}%)  |  null-drift : {s['n_columns_null_drift']} cols")

    drifted = uv[uv["drift_flag"] == True].head(top_n)
    if drifted.empty:
        print("  ✅ Aucun drift univarié détecté")
    else:
        for _, row in drifted.iterrows():
            icon = "🔴" if row["drift_score"] > 0.65 else "🟠" if row["drift_score"] > 0.35 else "🟡"
            extras: list[str] = []

            for key, label in [("psi", "PSI"), ("wasserstein", "Wass")]:
                v = row.get(key)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    extras.append(f"{label}={v:.3f}")

            for key, label in [("ks_pvalue", "KS_p"), ("chi2_pvalue", "χ²_p"), ("ks_pvalue_len", "KS_p")]:
                v = row.get(key)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    extras.append(f"{label}={v:.4f}")

            for key, label in [("mean_shift_z", "Δμ")]:
                v = row.get(key)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    extras.append(f"{label}={v:+.2f}σ")

            std_r = row.get("std_ratio")
            if std_r is not None and not (isinstance(std_r, float) and np.isnan(std_r)):
                if std_r < 0.6 or std_r > 1.8:
                    extras.append(f"std×{std_r:.2f}")

            for key, sym in [("new_categories_n", "+"), ("missing_categories_n", "−")]:
                v = row.get(key, 0)
                if v and v > 0:
                    ex_key = key.replace("_n", "_ex")
                    ex = row.get(ex_key, [])
                    extras.append(f"{sym}{v} cat ({', '.join(str(x) for x in ex[:3])})")

            nd = row.get("null_delta", 0) or 0
            if abs(nd) >= 0.05:
                extras.append(f"null Δ={nd:+.1%}")

            col_str  = f"{row['column']:<32s}"
            type_str = f"[{row['type']:16s}]"
            score    = f"score={row['drift_score']:.3f}"
            print(f"  {icon} {type_str} {col_str} {score}  {' | '.join(extras)}")

    # ── Multivarié ───────────────────────────────────────────────────────────
    mv_sev  = mv.get("severity", "N/A")
    mv_icon = _SEV_ICON.get(mv_sev, "⚪")
    print(f"\n🧠  MULTIVARIÉ  {mv_icon} {mv_sev}")
    if "roc_auc" in mv:
        print(f"  {mv['interpretation']}")
        if mv.get("top_separator_features"):
            top_feats = list(mv["top_separator_features"].items())[:5]
            feat_str  = "  |  ".join(f"{k}={v:.3f}" for k, v in top_feats)
            print(f"  Top features séparatrices : {feat_str}")
    else:
        print(f"  {mv.get('reason', 'N/A')}")

    # ── Corrélation ──────────────────────────────────────────────────────────
    cr_sev  = cr.get("severity", "N/A")
    cr_icon = _SEV_ICON.get(cr_sev, "⚪")
    print(f"\n🔗  CORRÉLATION  {cr_icon} {cr_sev}")
    if "mean_abs_corr_delta" in cr:
        print(f"  Mean|ΔCorr|={cr['mean_abs_corr_delta']:.4f}   Max|ΔCorr|={cr['max_abs_corr_delta']:.4f}   "
              f"sur {cr['n_cols_used']} colonnes numériques")
        if cr.get("top_drifted_pairs"):
            print("  Paires les plus driftées :")
            for p in cr["top_drifted_pairs"][:5]:
                print(f"    {p['col_a']} ↔ {p['col_b']}: "
                      f"{p['corr_ref']:+.3f} → {p['corr_cur']:+.3f}  (Δ={p['delta']:.3f})")

    print("\n" + "═" * 72)


# ══════════════════════════════════════════════════════════════════════════════
#  8. HELPERS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def get_drifted_columns(report: dict, min_score: float = 0.0) -> pd.DataFrame:
    """Retourne un DataFrame des colonnes driftées, triées par score."""
    uv = report["univariate"]
    mask = (uv["drift_flag"] == True) & (uv["drift_score"] >= min_score)
    return uv[mask].reset_index(drop=True)


def get_drift_summary_df(report: dict) -> pd.DataFrame:
    """Résumé tabulaire pour export / dashboard."""
    rows = []
    s  = report["summary"]
    sc = report["schema"]
    mv = report["multivariate"]
    cr = report["correlation"]

    rows.append(("overall_severity",       s["overall_severity"],         "global"))
    rows.append(("pct_columns_drifted",    s["pct_columns_drifted"],      "univariate"))
    rows.append(("n_columns_drifted",      s["n_columns_drifted"],        "univariate"))
    rows.append(("n_null_drift_cols",      s["n_columns_null_drift"],     "univariate"))
    rows.append(("multivariate_auc",       mv.get("roc_auc", np.nan),    "multivariate"))
    rows.append(("multivariate_severity",  mv.get("severity", "N/A"),    "multivariate"))
    rows.append(("corr_mean_delta",        cr.get("mean_abs_corr_delta", np.nan), "correlation"))
    rows.append(("corr_severity",          cr.get("severity", "N/A"),    "correlation"))
    rows.append(("schema_issues",          s["n_schema_issues"],          "schema"))
    rows.append(("added_columns",          sc["added_columns"],           "schema"))
    rows.append(("removed_columns",        sc["removed_columns"],         "schema"))

    return pd.DataFrame(rows, columns=["metric", "value", "category"])


# ══════════════════════════════════════════════════════════════════════════════
#  9. EXPLICATION NARRATIVE PAR COLONNE
# ══════════════════════════════════════════════════════════════════════════════

# ── Seuils de sévérité par métrique ──────────────────────────────────────────

_SEV_THRESHOLDS = {
    # (LOW, MODERATE, HIGH, CRITICAL)
    "psi":             (0.10, 0.20, 0.35, 0.50),
    "wasserstein":     (0.10, 0.30, 0.60, 1.00),
    "mean_shift_z":    (0.50, 1.00, 2.00, 3.50),   # absolu
    "std_ratio_delta": (0.15, 0.30, 0.50, 0.80),   # |ratio - 1|
    "null_delta":      (0.03, 0.08, 0.15, 0.30),   # absolu
    "ks_stat":         (0.05, 0.10, 0.20, 0.35),
    "new_cat_pct":     (0.02, 0.05, 0.10, 0.25),   # part de nouvelles catégories
    "missing_cat_pct": (0.02, 0.05, 0.10, 0.25),
    "chi2_norm":       (0.10, 0.25, 0.50, 0.80),   # chi2 normalisé [0..1] approx
}

_SEV_LABELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
_SEV_RANK   = {s: i for i, s in enumerate(_SEV_LABELS)}


def _score_to_sev(value: float, thresholds: tuple) -> str:
    lo, mo, hi, cr = thresholds
    if value >= cr: return "CRITICAL"
    if value >= hi: return "HIGH"
    if value >= mo: return "MODERATE"
    if value >= lo: return "LOW"
    return "NONE"


def _max_sev(*sevs: str) -> str:
    valid = [s for s in sevs if s in _SEV_RANK]
    return max(valid, key=lambda s: _SEV_RANK[s]) if valid else "NONE"


# ── Impact métier selon le type de drift ────────────────────────────────────

_IMPACT_TEMPLATES = {
    # Numériques
    "mean_shift": (
        "Le centre de la distribution a glissé de {mean_shift_z:+.2f}σ "
        "(ref μ={mean_ref:.4g} → prod μ={mean_cur:.4g}). "
        "Les modèles linéaires, arbres et réseaux de neurones utilisent tous la valeur brute : "
        "un décalage de {abs_z:.2f}σ signifie que {pct_out:.0f}% des observations "
        "de production tombent en dehors de la plage typique d'entraînement."
    ),
    "std_change": (
        "La variance a changé (σ_prod/σ_ref = {std_ratio:.3f}). "
        "{direction} "
        "Les features normalisées pendant le preprocessing seront recalibées différemment ; "
        "les modèles sensibles à l'échelle (SVM, KNN, régression Ridge) sont directement affectés."
    ),
    "tail_shift": (
        "Les queues de distribution se sont déplacées : "
        "P5 Δ={p5_delta:+.4g}, P50 Δ={p50_delta:+.4g}, P95 Δ={p95_delta:+.4g}. "
        "Les valeurs extrêmes sont particulièrement impactantes pour les arbres de décision "
        "(splits sur percentiles) et la détection d'anomalies."
    ),
    "shape_change": (
        "La forme générale de la distribution a changé (KS={ks_stat:.3f}, p={ks_pvalue:.2e} ; "
        "PSI={psi:.3f}). "
        "Tous les modèles calibrés sur df1 voient leur espace d'entrée altéré. "
        "PSI > 0.25 est le seuil standard de recalibration forcée en risque crédit."
    ),
    # Catégorielles
    "new_categories": (
        "{new_n} nouvelle(s) catégorie(s) apparue(s) en production "
        "({new_ex}), représentant ~{new_pct:.1%} des valeurs. "
        "Ces modalités sont INCONNUES du modèle : encodage one-hot → colonne manquante, "
        "label encoder → exception KeyError, embeddings → vecteur hors-distribution."
    ),
    "missing_categories": (
        "{miss_n} catégorie(s) du train absente(s) en production "
        "({miss_ex}). "
        "Si le modèle a appris des effets spécifiques à ces modalités (coefficients, splits), "
        "ces effets ne s'appliquent plus ; les prédictions pour les classes voisines "
        "absorbent des observations qu'elles n'ont jamais vues."
    ),
    "freq_shift": (
        "Les fréquences relatives des catégories existantes ont changé significativement "
        "(PSI={psi:.3f}, χ²_p={chi2_pvalue:.2e}). "
        "Les features catégorielles encodées en fréquence ou target-encodées "
        "vont produire des valeurs décalées, biaisant les prédictions."
    ),
    # Nulls
    "null_increase": (
        "Le taux de valeurs nulles a AUGMENTÉ de {null_ref:.1%} → {null_cur:.1%} "
        "(Δ={null_delta:+.1%}). "
        "L'imputation appliquée au train (médiane, mode, modèle) va injecter "
        "une valeur artificielle pour {null_extra:.0f}% de nouveaux cas, "
        "créant un biais systématique."
    ),
    "null_decrease": (
        "Le taux de valeurs nulles a DIMINUÉ de {null_ref:.1%} → {null_cur:.1%} "
        "(Δ={null_delta:+.1%}). "
        "Si le modèle a appris à utiliser l'indicateur de nullité comme feature implicite, "
        "la sémantique de la donnée manquante a changé."
    ),
    # Texte
    "text_len_shift": (
        "La longueur moyenne des textes a changé "
        "({avg_len_ref:.0f} → {avg_len_cur:.0f} caractères, KS={ks_stat_len:.3f}). "
        "Les modèles TF-IDF, bag-of-words ou les embeddings de longueur fixe "
        "seront affectés par le changement de densité informationnelle."
    ),
    # Datetime
    "datetime_shift": (
        "La plage temporelle a changé "
        "(ref : {min_ref} → {max_ref}, prod : {min_cur} → {max_cur}). "
        "Les features dérivées du temps (âge, jour de semaine, distance à date de ref) "
        "vont produire des valeurs hors-distribution ou négatives."
    ),
}


def _nan_safe(v, default=np.nan):
    """Retourne default si v est None ou NaN flottant."""
    if v is None:
        return default
    if isinstance(v, float) and np.isnan(v):
        return default
    return v


def _explain_numeric(row: pd.Series) -> tuple[list[str], str]:
    """Retourne (liste de phrases d'explication, sévérité max)."""
    findings: list[str] = []
    sevs:     list[str] = []

    psi        = _nan_safe(row.get("psi"))
    ks_stat    = _nan_safe(row.get("ks_stat"))
    ks_p       = _nan_safe(row.get("ks_pvalue"))
    wass       = _nan_safe(row.get("wasserstein"))
    mz         = _nan_safe(row.get("mean_shift_z"))
    std_ratio  = _nan_safe(row.get("std_ratio"))
    p5d        = _nan_safe(row.get("p5_delta"), 0.0)
    p50d       = _nan_safe(row.get("p50_delta"), 0.0)
    p95d       = _nan_safe(row.get("p95_delta"), 0.0)
    mean_ref   = _nan_safe(row.get("mean_ref"), 0.0)
    mean_cur   = _nan_safe(row.get("mean_cur"), 0.0)

    # ── Forme globale ────────────────────────────────────────────────────────
    if not np.isnan(psi) and psi >= 0.10:
        sev = _score_to_sev(psi, _SEV_THRESHOLDS["psi"])
        sevs.append(sev)
        findings.append(
            _IMPACT_TEMPLATES["shape_change"].format(
                ks_stat=_nan_safe(ks_stat, 0),
                ks_pvalue=_nan_safe(ks_p, 1),
                psi=psi,
            )
        )
    elif not np.isnan(ks_stat) and ks_stat >= 0.05:
        sev = _score_to_sev(ks_stat, _SEV_THRESHOLDS["ks_stat"])
        sevs.append(sev)
        findings.append(
            f"Distribution significativement différente (KS={ks_stat:.3f}, p={ks_p:.2e}) "
            f"mais PSI non calculable — surveiller de près."
        )

    # ── Décalage de moyenne ───────────────────────────────────────────────────
    if not np.isnan(mz) and abs(mz) >= 0.5:
        sev = _score_to_sev(abs(mz), _SEV_THRESHOLDS["mean_shift_z"])
        sevs.append(sev)
        # % observations prod en dehors de [μ±2σ] ref, approx gaussien
        pct_out = min(100.0, abs(mz) / 4.0 * 100)
        findings.append(
            _IMPACT_TEMPLATES["mean_shift"].format(
                mean_shift_z=mz, mean_ref=mean_ref, mean_cur=mean_cur,
                abs_z=abs(mz), pct_out=pct_out,
            )
        )

    # ── Changement de variance ────────────────────────────────────────────────
    if not np.isnan(std_ratio):
        delta_std = abs(std_ratio - 1.0)
        if delta_std >= 0.15:
            sev = _score_to_sev(delta_std, _SEV_THRESHOLDS["std_ratio_delta"])
            sevs.append(sev)
            direction = (
                "La variance a RÉTRÉCI (données plus concentrées) : "
                if std_ratio < 1 else
                "La variance a EXPLOSÉ (données plus dispersées) : "
            )
            findings.append(
                _IMPACT_TEMPLATES["std_change"].format(
                    std_ratio=std_ratio, direction=direction
                )
            )

    # ── Déplacement des queues ────────────────────────────────────────────────
    max_tail = max(abs(p5d), abs(p50d), abs(p95d))
    scale    = abs(mean_ref) if abs(mean_ref) > 1e-9 else 1.0
    rel_tail = max_tail / scale
    if rel_tail >= 0.05 and max_tail >= 0.001:
        findings.append(
            _IMPACT_TEMPLATES["tail_shift"].format(
                p5_delta=p5d, p50_delta=p50d, p95_delta=p95d
            )
        )

    return findings, _max_sev(*sevs) if sevs else "NONE"


def _explain_categorical(row: pd.Series) -> tuple[list[str], str]:
    findings: list[str] = []
    sevs:     list[str] = []

    psi      = _nan_safe(row.get("psi"))
    chi2_p   = _nan_safe(row.get("chi2_pvalue"))
    chi2_s   = _nan_safe(row.get("chi2_stat"))
    new_n    = int(_nan_safe(row.get("new_categories_n"),  0))
    miss_n   = int(_nan_safe(row.get("missing_categories_n"), 0))
    new_ex   = row.get("new_categories_ex",     []) or []
    miss_ex  = row.get("missing_categories_ex", []) or []
    n_ref    = max(int(_nan_safe(row.get("n_cat_ref"), 1)), 1)

    # ── Nouvelles catégories inconnues ────────────────────────────────────────
    if new_n > 0:
        new_pct = new_n / (n_ref + new_n)
        sev = _score_to_sev(new_pct, _SEV_THRESHOLDS["new_cat_pct"])
        sevs.append(sev)
        findings.append(
            _IMPACT_TEMPLATES["new_categories"].format(
                new_n=new_n,
                new_ex=", ".join(str(x) for x in new_ex[:4]) or "—",
                new_pct=new_pct,
            )
        )

    # ── Catégories disparues ──────────────────────────────────────────────────
    if miss_n > 0:
        miss_pct = miss_n / n_ref
        sev = _score_to_sev(miss_pct, _SEV_THRESHOLDS["missing_cat_pct"])
        sevs.append(sev)
        findings.append(
            _IMPACT_TEMPLATES["missing_categories"].format(
                miss_n=miss_n,
                miss_ex=", ".join(str(x) for x in miss_ex[:4]) or "—",
            )
        )

    # ── Glissement de fréquences ──────────────────────────────────────────────
    if not np.isnan(psi) and psi >= 0.10:
        sev = _score_to_sev(psi, _SEV_THRESHOLDS["psi"])
        sevs.append(sev)
        findings.append(
            _IMPACT_TEMPLATES["freq_shift"].format(
                psi=psi,
                chi2_pvalue=_nan_safe(chi2_p, 1.0),
            )
        )
    elif not np.isnan(chi2_p) and chi2_p < 0.05:
        sevs.append("LOW")
        findings.append(
            f"Chi² significatif (p={chi2_p:.2e}) mais PSI faible — "
            "glissement mineur dans les fréquences des catégories existantes."
        )

    return findings, _max_sev(*sevs) if sevs else "NONE"


def _explain_null(row: pd.Series) -> tuple[list[str], str]:
    null_ref   = float(_nan_safe(row.get("null_rate_ref"), 0.0))
    null_cur   = float(_nan_safe(row.get("null_rate_cur"), 0.0))
    null_delta = float(_nan_safe(row.get("null_delta"),    0.0))

    if abs(null_delta) < 0.03:
        return [], "NONE"

    sev = _score_to_sev(abs(null_delta), _SEV_THRESHOLDS["null_delta"])
    null_extra = abs(null_cur - null_ref) * 100

    if null_delta > 0:
        msg = _IMPACT_TEMPLATES["null_increase"].format(
            null_ref=null_ref, null_cur=null_cur,
            null_delta=null_delta, null_extra=null_extra,
        )
    else:
        msg = _IMPACT_TEMPLATES["null_decrease"].format(
            null_ref=null_ref, null_cur=null_cur,
            null_delta=null_delta,
        )
    return [msg], sev


def _explain_text(row: pd.Series) -> tuple[list[str], str]:
    ks_stat   = _nan_safe(row.get("ks_stat_len"))
    ks_p      = _nan_safe(row.get("ks_pvalue_len"))
    avg_ref   = _nan_safe(row.get("avg_len_ref"), 0)
    avg_cur   = _nan_safe(row.get("avg_len_cur"), 0)

    if np.isnan(ks_stat) or ks_stat < 0.05:
        return [], "NONE"

    sev = _score_to_sev(ks_stat, _SEV_THRESHOLDS["ks_stat"])
    msg = _IMPACT_TEMPLATES["text_len_shift"].format(
        avg_len_ref=avg_ref, avg_len_cur=avg_cur,
        ks_stat_len=ks_stat,
    )
    return [msg], sev


def _explain_datetime(row: pd.Series) -> tuple[list[str], str]:
    ks_stat = _nan_safe(row.get("ks_stat"))
    ks_p    = _nan_safe(row.get("ks_pvalue"))

    if np.isnan(ks_stat) or ks_stat < 0.05:
        return [], "NONE"

    sev = _score_to_sev(ks_stat, _SEV_THRESHOLDS["ks_stat"])
    msg = _IMPACT_TEMPLATES["datetime_shift"].format(
        min_ref=row.get("min_ref", "?"), max_ref=row.get("max_ref", "?"),
        min_cur=row.get("min_cur", "?"), max_cur=row.get("max_cur", "?"),
    )
    return [msg], sev


# ── Analyse de corrélation par colonne ───────────────────────────────────────

def _corr_mentions(col: str, correlation: dict) -> list[str]:
    """Retourne les mentions de corrélations driftées impliquant cette colonne."""
    mentions = []
    for pair in correlation.get("top_drifted_pairs", []):
        if pair["col_a"] == col or pair["col_b"] == col:
            partner = pair["col_b"] if pair["col_a"] == col else pair["col_a"]
            mentions.append(
                f"Corrélation avec '{partner}' a changé : "
                f"{pair['corr_ref']:+.3f} → {pair['corr_cur']:+.3f} "
                f"(Δ={pair['delta']:.3f}). "
                "Ceci peut indiquer qu'une relation apprise conjointement "
                "n'est plus valide en production."
            )
    return mentions


# ── Point d'entrée public ────────────────────────────────────────────────────

def explain_drift(
    report: dict,
    *,
    only_drifted: bool = True,
    min_severity: str = "LOW",
    top_n: Optional[int] = None,
    print_report: bool = True,
) -> pd.DataFrame:
    """
    Génère une explication narrative du drift pour chaque colonne.

    Paramètres
    ----------
    report        : sortie de analyze_drift()
    only_drifted  : si True, n'affiche que les colonnes avec drift_flag=True
    min_severity  : filtre minimum ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')
    top_n         : limite aux N premières colonnes (par drift_score)
    print_report  : affiche le rapport formaté

    Retourne
    --------
    DataFrame avec colonnes : column, type, drift_score, severity, explanation
    """
    uv          = report["univariate"].copy()
    correlation = report["correlation"]
    min_rank    = _SEV_RANK.get(min_severity, 0)

    if only_drifted:
        uv = uv[uv["drift_flag"] == True]
    if top_n:
        uv = uv.head(top_n)

    records: list[dict] = []

    for _, row in uv.iterrows():
        col_type = str(row.get("type", ""))

        # ── Construire les findings par domaine ───────────────────────────────
        all_findings: list[str] = []
        all_sevs:     list[str] = []

        # Nulls (tous types)
        null_findings, null_sev = _explain_null(row)
        all_findings.extend(null_findings)
        all_sevs.append(null_sev)

        # Spécifique au type
        if col_type == "numeric":
            f, s = _explain_numeric(row)
        elif col_type in ("categorical", "boolean", "high_card_string"):
            f, s = _explain_categorical(row)
        elif col_type == "text":
            f, s = _explain_text(row)
        elif col_type == "datetime":
            f, s = _explain_datetime(row)
        else:
            f, s = [], "NONE"

        all_findings.extend(f)
        all_sevs.append(s)

        # Corrélations impliquant cette colonne
        corr_f = _corr_mentions(str(row["column"]), correlation)
        if corr_f:
            all_findings.extend(corr_f)
            all_sevs.append("MODERATE")

        col_sev = _max_sev(*all_sevs) if all_sevs else "NONE"

        # Filtre par sévérité
        if _SEV_RANK.get(col_sev, -1) < min_rank:
            continue

        records.append({
            "column":      row["column"],
            "type":        col_type,
            "drift_score": row.get("drift_score", np.nan),
            "severity":    col_sev,
            "n_findings":  len(all_findings),
            "explanation": all_findings,  # liste ordonnée
        })

    result_df = pd.DataFrame(records)
    if result_df.empty:
        if print_report:
            print("✅ Aucun drift à expliquer avec les filtres courants.")
        return result_df

    # Tri : sévérité desc, puis score desc
    sev_order = pd.CategoricalDtype(
        ["CRITICAL", "HIGH", "MODERATE", "LOW", "NONE"], ordered=True
    )
    result_df["_sev_ord"] = pd.Categorical(result_df["severity"], dtype=sev_order)
    result_df = (
        result_df
        .sort_values(["_sev_ord", "drift_score"], ascending=[True, False])
        .drop(columns="_sev_ord")
        .reset_index(drop=True)
    )

    if print_report:
        _print_explain_report(result_df, report)

    return result_df


# ── Affichage narratif ────────────────────────────────────────────────────────

_SEV_COLOR = {
    "CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢", "NONE": "⚪",
}
_SEV_BAR = {
    "CRITICAL": "████████████ CRITICAL",
    "HIGH":     "█████████    HIGH",
    "MODERATE": "██████       MODERATE",
    "LOW":      "███          LOW",
    "NONE":     "─            NONE",
}


def _print_explain_report(df: pd.DataFrame, report: dict) -> None:
    s  = report["summary"]
    mv = report["multivariate"]
    cr = report["correlation"]

    print("\n" + "═" * 76)
    print("  RAPPORT D'EXPLICATION DU DRIFT — IMPACT MODÈLE")
    print(f"  Sévérité globale : {_SEV_COLOR.get(s['overall_severity'], '⚪')} {s['overall_severity']}")
    print("═" * 76)

    # ── Contexte global ───────────────────────────────────────────────────────
    print("\n┌─ CONTEXTE GLOBAL " + "─" * 58)
    print(f"│  {s['n_columns_drifted']}/{s['n_columns_analyzed']} colonnes driftées ({s['pct_columns_drifted']}%)")

    if "roc_auc" in mv:
        auc_msg = (
            f"AUC={mv['roc_auc']:.3f} → le modèle peut distinguer ref vs prod "
            f"[{mv['severity']}]"
        )
        print(f"│  Drift multivarié : {auc_msg}")

    if "mean_abs_corr_delta" in cr:
        print(
            f"│  Drift corrélation : mean|ΔCorr|={cr['mean_abs_corr_delta']:.4f} [{cr['severity']}]"
        )
    print("└" + "─" * 75)

    # ── Colonnes CRITICAL ─────────────────────────────────────────────────────
    for sev in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        subset = df[df["severity"] == sev]
        if subset.empty:
            continue

        icon  = _SEV_COLOR[sev]
        bar   = _SEV_BAR[sev]
        print(f"\n{'─'*76}")
        print(f"  {icon}  {bar}  ({len(subset)} colonne{'s' if len(subset) > 1 else ''})")
        print(f"{'─'*76}")

        for _, row in subset.iterrows():
            col_label = f"[{row['type']:16s}] {row['column']}"
            print(f"\n  ▶ {col_label}  (score={row['drift_score']:.3f})")
            print(f"    {'─'*70}")
            for i, finding in enumerate(row["explanation"], 1):
                # Wrap à 72 chars avec indentation
                words = finding.split()
                lines = []
                current = f"    {i}. "
                pad     = " " * len(current)
                for word in words:
                    if len(current) + len(word) + 1 > 76:
                        lines.append(current)
                        current = pad + word
                    else:
                        current = current + word + " "
                lines.append(current)
                print("\n".join(lines))

    # ── Recommandations synthétiques ──────────────────────────────────────────
    critical_cols = df[df["severity"] == "CRITICAL"]["column"].tolist()
    high_cols     = df[df["severity"] == "HIGH"]["column"].tolist()

    print(f"\n{'═'*76}")
    print("  RECOMMANDATIONS")
    print(f"{'═'*76}")

    if critical_cols:
        print(f"\n  🔴 ACTION IMMÉDIATE requis sur : {', '.join(critical_cols[:8])}")
        print("     → Re-entraîner ou exclure ces features avant déploiement.")

    if high_cols:
        print(f"\n  🟠 SURVEILLER de près : {', '.join(high_cols[:8])}")
        print("     → Envisager recalibration, monitoring alertes, shadow mode.")

    mv_auc = mv.get("roc_auc", 0.5)
    if mv_auc > 0.70:
        top_seps = list(mv.get("top_separator_features", {}).keys())[:4]
        print(f"\n  🧠 Drift multivarié fort (AUC={mv_auc:.3f}).")
        if top_seps:
            print(f"     Features les plus séparatrices : {', '.join(top_seps)}")
        print("     → Le modèle opère dans une région hors de sa distribution d'entraînement.")

    cr_sev = cr.get("severity", "NONE")
    if cr_sev in ("HIGH", "CRITICAL"):
        print(f"\n  🔗 Structure de corrélation altérée [{cr_sev}].")
        print("     → Les interactions entre features apprises par le modèle peuvent")
        print("        être invalides. Inspecter les paires listées dans report['correlation'].")

    if not critical_cols and not high_cols and mv_auc <= 0.65 and cr_sev not in ("HIGH", "CRITICAL"):
        print("\n  ✅ Drift présent mais contrôlé. Monitoring standard suffisant.")

    print("\n" + "═" * 76 + "\n")
