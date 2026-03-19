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
    diff_vals = diff[upper & ~np.isnan(diff[upper])]

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
