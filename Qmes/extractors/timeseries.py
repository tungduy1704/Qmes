"""qmatch/extractors/timeseries.py

Meta-feature extraction cho univariate time series classification.
Dùng pycatch22 (22 canonical time-series characteristics, Lubba et al. 2019).

Input: X (n_samples, n_timesteps), y (n_samples,)
Strategy: tính catch22 trên từng series, rồi aggregate (mean) across samples.
Plus: simple descriptors (n_samples, n_timesteps, n_classes).

Scaling: catch22 cần z-scored series. Extractor tự normalize internally.

Dependencies: pycatch22
"""
from __future__ import annotations

import numpy as np

from Qmes.extractors.base import BaseExtractor

# ── catch22 feature names (canonical ordering) ──────────────────────────────
_CATCH22_NAMES: list[str] = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
    "FC_LocalSimple_mean1_tauresrat",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
]

# Simple dataset descriptors
_SIMPLE_NAMES: list[str] = [
    "n_samples",
    "n_timesteps",
    "n_classes",
]

_ALL_NAMES: list[str] = _CATCH22_NAMES + _SIMPLE_NAMES


def _zscore_series(X: np.ndarray) -> np.ndarray:
    """Z-score normalize từng series independently."""
    X = X.copy().astype(np.float64)
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-12, 1.0, stds)
    return (X - means) / stds


def _compute_catch22_aggregated(X: np.ndarray) -> np.ndarray:
    """Tính catch22 trên từng series, trả mean vector (22,).

    Strategy: per-series extraction → mean aggregation.
    Robust: nếu một series fail, bỏ qua nó.
    """
    import pycatch22

    n_samples = X.shape[0]
    all_features = []

    for i in range(n_samples):
        try:
            result = pycatch22.catch22_all(X[i].tolist())
            all_features.append(result["values"])
        except Exception:
            continue  # skip failed series

    if not all_features:
        return np.zeros(22, dtype=np.float64)

    arr = np.array(all_features, dtype=np.float64)
    # Replace non-finite per-series values before averaging
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr.mean(axis=0)


class TimeSeriesExtractor(BaseExtractor):
    """Meta-feature extractor cho univariate time series classification.

    Produces vector:
        [22 catch22 features (mean over samples)] + [n_samples, n_timesteps, n_classes]
    Total: 25-dim
    """

    @property
    def task_type(self) -> str:
        return "timeseries"

    @property
    def _feature_names(self) -> list[str]:
        return _ALL_NAMES

    def _extract_raw(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        if y is None:
            raise ValueError("Time series extractor requires y")

        # Z-score normalize for catch22
        X_norm = _zscore_series(X)

        # catch22 aggregated over all series
        catch22_vec = _compute_catch22_aggregated(X_norm)

        # Simple descriptors
        n_samples = float(X.shape[0])
        n_timesteps = float(X.shape[1])
        n_classes = float(len(np.unique(y)))

        return np.concatenate([
            catch22_vec,
            [n_samples, n_timesteps, n_classes],
        ])