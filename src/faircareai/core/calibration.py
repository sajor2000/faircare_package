"""
FairCareAI Calibration Metrics Module

Implements calibration assessment for healthcare ML models:
1. ACE (Adaptive Calibration Error) with quantile binning
2. Cluster bootstrap CIs for calibration metrics
3. Group calibration parity analysis
4. Calibration gap computation

ACE is preferred over ECE for healthcare data because:
- Quantile binning ensures high-risk tail contributes equally
- Uniform binning (ECE) is dominated by low-risk majority in imbalanced data

Methodology: Van Calster et al. (2025), CHAI RAIC AC1.CR102 (calibration).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# ==============================================================================
# Result Dataclasses
# ==============================================================================


@dataclass(frozen=True)
class CalibrationResult:
    """Result of calibration analysis for a single group.

    Attributes:
        group: Group identifier (or 'all' for overall).
        ace: Adaptive Calibration Error.
        ace_ci_lower: Lower bound of ACE CI.
        ace_ci_upper: Upper bound of ACE CI.
        n_bins: Number of bins requested.
        n_samples: Total sample size.
        n_effective: Effective sample size (accounting for clustering).
        bins_used: Number of bins actually used.
    """

    group: str
    ace: float
    ace_ci_lower: float | None
    ace_ci_upper: float | None
    n_bins: int
    n_samples: int
    n_effective: int
    bins_used: int


@dataclass
class GroupCalibrationResult:
    """Result of group-level calibration analysis.

    Attributes:
        group_results: Dict mapping group name to CalibrationResult.
        calibration_gap: Max ACE - Min ACE across groups.
        calibration_gap_ci: CI for calibration gap (if bootstrap).
        worst_calibrated_group: Group with highest ACE.
    """

    group_results: dict[str, CalibrationResult]
    calibration_gap: float
    calibration_gap_ci: tuple[float, float] | None
    worst_calibrated_group: str


# ==============================================================================
# ACE Computation
# ==============================================================================


def compute_ace(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    min_per_bin: int = 5,
) -> tuple[float, int]:
    """
    Compute Adaptive Calibration Error using quantile binning.

    Unlike ECE which uses uniform bins, ACE uses quantile bins to ensure
    each bin has approximately equal representation. This is critical for
    imbalanced healthcare data where most patients are low-risk.

    Args:
        y_true: Binary outcomes (0/1).
        y_prob: Predicted probabilities.
        n_bins: Number of quantile bins.
        min_per_bin: Minimum samples per bin (bins with fewer are excluded).

    Returns:
        Tuple of (ace_value, bins_actually_used).
    """
    # Handle empty data
    if len(y_true) == 0 or len(y_prob) == 0:
        return (np.nan, 0)

    # Filter NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(y_true) == 0:
        return (np.nan, 0)

    # Handle constant predictions (all same value)
    if np.all(y_prob == y_prob[0]):
        # Single effective bin
        observed = np.mean(y_true)
        predicted = y_prob[0]
        return (abs(observed - predicted), 1)

    # Compute quantile bin edges
    try:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles)
        # Remove duplicate edges (happens when many predictions are identical)
        bin_edges = np.unique(bin_edges)
    except Exception:
        return (np.nan, 0)

    # Assign samples to bins
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    # Compute ACE
    total_error = 0.0
    total_weight = 0
    bins_used = 0

    for bin_idx in range(len(bin_edges) - 1):
        bin_mask = bin_indices == bin_idx
        n_in_bin = np.sum(bin_mask)

        if n_in_bin < min_per_bin:
            continue

        bins_used += 1
        observed_rate = np.mean(y_true[bin_mask])
        predicted_rate = np.mean(y_prob[bin_mask])
        total_error += n_in_bin * abs(observed_rate - predicted_rate)
        total_weight += n_in_bin

    if total_weight == 0:
        return (np.nan, 0)

    ace = total_error / total_weight
    return (ace, bins_used)


# ==============================================================================
# ACE with Bootstrap CI
# ==============================================================================


def compute_ace_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cluster_ids: np.ndarray | None = None,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> tuple[float, tuple[float, float], int]:
    """
    Compute ACE with bootstrap confidence interval.

    Supports cluster-aware bootstrap (resampling at patient level)
    to account for within-patient correlation.

    Args:
        y_true: Binary outcomes (0/1).
        y_prob: Predicted probabilities.
        cluster_ids: Optional cluster IDs for cluster bootstrap.
        n_bins: Number of quantile bins.
        n_bootstrap: Number of bootstrap iterations.
        alpha: Significance level.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (ace, (ci_lower, ci_upper), bins_used).
    """
    rng = np.random.default_rng(random_state)

    # Point estimate
    ace, bins_used = compute_ace(y_true, y_prob, n_bins)

    if n_bootstrap == 0:
        return (ace, (np.nan, np.nan), bins_used)

    n = len(y_true)
    if n == 0:
        return (np.nan, (np.nan, np.nan), 0)

    bootstrap_aces = np.zeros(n_bootstrap)

    if cluster_ids is not None:
        # Cluster-aware bootstrap
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        for i in range(n_bootstrap):
            sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)

            # Build resampled indices
            indices_list: list[int] = []
            for cluster in sampled_clusters:
                cluster_mask = cluster_ids == cluster
                indices_list.extend(np.where(cluster_mask)[0].tolist())

            indices = np.array(indices_list)
            boot_ace, _ = compute_ace(y_true[indices], y_prob[indices], n_bins)
            bootstrap_aces[i] = boot_ace
    else:
        # Standard bootstrap
        for i in range(n_bootstrap):
            indices = rng.integers(0, n, size=n)
            boot_ace, _ = compute_ace(y_true[indices], y_prob[indices], n_bins)
            bootstrap_aces[i] = boot_ace

    # Filter valid values
    valid_aces = bootstrap_aces[~np.isnan(bootstrap_aces)]
    if len(valid_aces) == 0:
        return (ace, (np.nan, np.nan), bins_used)

    ci_lower = np.percentile(valid_aces, 100 * alpha / 2)
    ci_upper = np.percentile(valid_aces, 100 * (1 - alpha / 2))

    return (ace, (ci_lower, ci_upper), bins_used)


# ==============================================================================
# Group Calibration
# ==============================================================================


def compute_group_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    cluster_ids: np.ndarray | None = None,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> GroupCalibrationResult:
    """
    Compute calibration metrics separately for each group.

    Args:
        y_true: Binary outcomes (0/1).
        y_prob: Predicted probabilities.
        groups: Group assignments.
        cluster_ids: Optional cluster IDs for cluster bootstrap.
        n_bins: Number of quantile bins.
        n_bootstrap: Number of bootstrap iterations.
        alpha: Significance level.
        random_state: Random seed.

    Returns:
        GroupCalibrationResult with per-group results and gap.
    """
    rng = np.random.default_rng(random_state)
    unique_groups = np.unique(groups)

    group_results = {}

    for group in unique_groups:
        group_mask = groups == group
        group_y_true = y_true[group_mask]
        group_y_prob = y_prob[group_mask]
        group_clusters = cluster_ids[group_mask] if cluster_ids is not None else None

        # Compute effective sample size
        if group_clusters is not None:
            n_effective = len(np.unique(group_clusters))
        else:
            n_effective = len(group_y_true)

        ace, (ci_lower, ci_upper), bins_used = compute_ace_with_ci(
            group_y_true,
            group_y_prob,
            cluster_ids=group_clusters,
            n_bins=n_bins,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            random_state=rng.integers(0, 2**31) if random_state is not None else None,
        )

        group_results[str(group)] = CalibrationResult(
            group=str(group),
            ace=ace,
            ace_ci_lower=ci_lower,
            ace_ci_upper=ci_upper,
            n_bins=n_bins,
            n_samples=len(group_y_true),
            n_effective=n_effective,
            bins_used=bins_used,
        )

    # Compute calibration gap
    aces = [r.ace for r in group_results.values() if not np.isnan(r.ace)]
    if len(aces) >= 2:
        calibration_gap = max(aces) - min(aces)
        worst_group = max(
            group_results.items(), key=lambda x: x[1].ace if not np.isnan(x[1].ace) else -np.inf
        )[0]
    else:
        calibration_gap = 0.0
        worst_group = list(group_results.keys())[0] if group_results else ""

    # Bootstrap CI for calibration gap if requested
    gap_ci = None
    if n_bootstrap > 0 and len(unique_groups) >= 2:
        gap_bootstraps = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            group_aces = []
            for group in unique_groups:
                group_mask = groups == group
                group_y_true = y_true[group_mask]
                group_y_prob = y_prob[group_mask]

                n = len(group_y_true)
                if n == 0:
                    continue

                indices = rng.integers(0, n, size=n)
                boot_ace, _ = compute_ace(group_y_true[indices], group_y_prob[indices], n_bins)
                if not np.isnan(boot_ace):
                    group_aces.append(boot_ace)

            if len(group_aces) >= 2:
                gap_bootstraps[i] = max(group_aces) - min(group_aces)
            else:
                gap_bootstraps[i] = np.nan

        valid_gaps = gap_bootstraps[~np.isnan(gap_bootstraps)]
        if len(valid_gaps) > 0:
            gap_ci = (
                np.percentile(valid_gaps, 100 * alpha / 2),
                np.percentile(valid_gaps, 100 * (1 - alpha / 2)),
            )

    return GroupCalibrationResult(
        group_results=group_results,
        calibration_gap=calibration_gap,
        calibration_gap_ci=gap_ci,
        worst_calibrated_group=worst_group,
    )


# ==============================================================================
# DataFrame Interface
# ==============================================================================


def compute_calibration_from_df(
    df: pl.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    group_col: str | None = None,
    cluster_col: str | None = None,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> CalibrationResult | GroupCalibrationResult:
    """
    Compute calibration metrics from a Polars DataFrame.

    Args:
        df: DataFrame with predictions and outcomes.
        y_true_col: Column for true labels.
        y_prob_col: Column for predicted probabilities.
        group_col: Optional column for group analysis.
        cluster_col: Optional column for clustering.
        n_bins: Number of quantile bins.
        n_bootstrap: Number of bootstrap iterations.
        alpha: Significance level.
        random_state: Random seed.

    Returns:
        CalibrationResult if no group_col, else GroupCalibrationResult.
    """
    y_true = df[y_true_col].to_numpy()
    y_prob = df[y_prob_col].to_numpy()
    cluster_ids = df[cluster_col].to_numpy() if cluster_col else None

    if group_col is None:
        # Overall calibration
        if cluster_col:
            n_effective = df[cluster_col].n_unique()
        else:
            n_effective = len(df)

        ace, (ci_lower, ci_upper), bins_used = compute_ace_with_ci(
            y_true, y_prob, cluster_ids, n_bins, n_bootstrap, alpha, random_state
        )

        return CalibrationResult(
            group="all",
            ace=ace,
            ace_ci_lower=ci_lower,
            ace_ci_upper=ci_upper,
            n_bins=n_bins,
            n_samples=len(df),
            n_effective=n_effective,
            bins_used=bins_used,
        )
    else:
        # Group-level calibration
        groups = df[group_col].to_numpy()
        return compute_group_calibration(
            y_true, y_prob, groups, cluster_ids, n_bins, n_bootstrap, alpha, random_state
        )
