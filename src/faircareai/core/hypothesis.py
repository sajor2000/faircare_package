"""
FairCareAI Hypothesis Testing Module

Implements rigorous hypothesis tests for fairness analysis:
1. Stratified cluster permutation tests
2. Correct stratum selection for TPR (y=1) vs FPR (y=0) metrics
3. Patient-level (cluster) permutation preserving correlation
4. Confounder stratification for Simpson's Paradox control
5. +1 p-value correction to avoid p=0

CRITICAL: For conditional metrics (TPR, FPR), permutation must occur
only within the relevant outcome stratum AND at the patient level.

Methodology: Van Calster et al. (2025).
Note: Statistical significance differs from clinical significance.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
import polars as pl

# ==============================================================================
# Metric Type Mapping
# ==============================================================================


def compute_metric_by_type(metric_name: str) -> str:
    """
    Map metric name to stratum type for permutation testing.

    TPR/Sensitivity/Recall: Permute within y_true==1 stratum
    FPR: Permute within y_true==0 stratum
    TNR/Specificity: Permute within y_true==0 stratum
    Other: Permute globally (Independence)

    Args:
        metric_name: Name of the metric.

    Returns:
        Stratum type: 'TPR', 'Recall', 'FPR', 'TNR', 'Specificity', or 'Independence'.
    """
    name_lower = metric_name.lower()

    if name_lower in ("tpr", "sensitivity", "recall"):
        return "TPR" if name_lower == "tpr" else "Recall"
    elif name_lower == "fpr":
        return "FPR"
    elif name_lower in ("tnr", "specificity"):
        return "TNR" if name_lower == "tnr" else "Specificity"
    else:
        return "Independence"


# ==============================================================================
# Stratified Cluster Permutation Test
# ==============================================================================


def stratified_cluster_permutation_test(
    df: pl.DataFrame,
    statistic_fn: Callable[[pl.DataFrame], float],
    metric_type: Literal["TPR", "FPR", "TNR", "Independence"],
    group_col: str,
    y_true_col: str,
    cluster_col: str | None = None,
    confound_cols: list[str] | None = None,
    n_perms: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> dict:
    """
    Perform stratified cluster permutation test.

    CRITICAL: For conditional metrics (TPR, FPR), permutation occurs
    only within the relevant outcome stratum (y_true==1 for TPR,
    y_true==0 for FPR). This is essential when base rates differ
    between groups.

    Args:
        df: DataFrame with data.
        statistic_fn: Function computing the test statistic from df.
        metric_type: Type of metric determining stratum.
        group_col: Column for group membership.
        y_true_col: Column for true labels.
        cluster_col: Column for clustering (patient ID).
        confound_cols: Columns to stratify by (site, etc.).
        n_perms: Number of permutations.
        alpha: Significance level.
        random_state: Random seed.

    Returns:
        Dict with 'observed_stat', 'p_value', 'null_distribution', 'ci_lower', 'ci_upper'.
    """
    rng = np.random.default_rng(random_state)

    # Determine stratum filter
    if metric_type == "TPR":
        stratum_filter = pl.col(y_true_col) == 1
    elif metric_type in ("FPR", "TNR"):
        stratum_filter = pl.col(y_true_col) == 0
    else:
        stratum_filter = None  # No filtering for Independence

    # Get stratum data
    if stratum_filter is not None:
        stratum_df = df.filter(stratum_filter)
    else:
        stratum_df = df

    # Check if stratum is empty
    if len(stratum_df) == 0:
        return {
            "observed_stat": np.nan,
            "p_value": np.nan,
            "null_distribution": np.array([]),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }

    # Compute observed statistic
    observed_stat = statistic_fn(stratum_df)

    # Build permutation structure
    if cluster_col is not None and cluster_col in stratum_df.columns:
        # Get unique cluster-group mappings - sort for reproducibility
        cluster_groups = stratum_df.select([cluster_col, group_col]).unique().sort(cluster_col)
        clusters = cluster_groups[cluster_col].to_numpy()
        groups = cluster_groups[group_col].to_numpy()
        use_clusters = True
    else:
        # Row-level permutation
        clusters = None
        groups = stratum_df[group_col].to_numpy()
        use_clusters = False

    # Handle confounder stratification
    if confound_cols:
        # Build strata from confounders using Polars unique
        confound_df = stratum_df.select(confound_cols)
        unique_strata_df = confound_df.unique()
        unique_strata_rows = unique_strata_df.to_dicts()
    else:
        confound_df = None
        unique_strata_rows = None

    # Permutation loop
    null_distribution = np.zeros(n_perms)

    for i in range(n_perms):
        if confound_df is not None and unique_strata_rows is not None:
            # Stratified permutation
            permuted_groups = groups.copy()
            for stratum_dict in unique_strata_rows:
                # Build filter for this stratum using the main stratum_df
                stratum_filter = pl.lit(True)
                for col, val in stratum_dict.items():
                    stratum_filter = stratum_filter & (pl.col(col) == val)

                if use_clusters and cluster_col is not None and clusters is not None:
                    # Get clusters in this stratum from the main stratum_df
                    stratum_clusters = (
                        stratum_df.filter(stratum_filter)[cluster_col].unique().to_numpy()
                    )
                    stratum_cluster_mask = np.isin(clusters, stratum_clusters)
                    stratum_indices = np.where(stratum_cluster_mask)[0]
                else:
                    # Use index matching for row-level permutation
                    confound_dicts = confound_df.to_dicts()
                    stratum_indices = np.array(
                        [j for j, row in enumerate(confound_dicts) if row == stratum_dict]
                    )

                # Permute within stratum
                if len(stratum_indices) > 0:
                    rng.shuffle(permuted_groups[stratum_indices])
        else:
            # Global permutation
            permuted_groups = groups.copy()
            rng.shuffle(permuted_groups)

        # Apply permuted groups
        if use_clusters and cluster_col is not None and clusters is not None:
            # Build cluster-to-group mapping
            # Create a mapping dict for deterministic replacement
            cluster_to_group_map = dict(zip(clusters, permuted_groups))

            # Map clusters to permuted groups using a list comprehension
            cluster_values = stratum_df[cluster_col].to_list()
            new_groups = [cluster_to_group_map[c] for c in cluster_values]
            permuted_df = stratum_df.with_columns(pl.Series(group_col, new_groups))
        else:
            permuted_df = stratum_df.with_columns(pl.Series(group_col, permuted_groups))

        # Compute statistic on permuted data
        null_distribution[i] = statistic_fn(permuted_df)

    # Compute p-value with +1 correction (ensures p > 0)
    # p = (1 + count(|null| >= |observed|)) / (1 + n_perms)
    if np.isnan(observed_stat):
        p_value = np.nan
    else:
        count_extreme = np.sum(np.abs(null_distribution) >= np.abs(observed_stat))
        p_value = (1 + count_extreme) / (1 + n_perms)

    # CI from null distribution
    if len(null_distribution) > 0:
        ci_lower = float(np.percentile(null_distribution, 100 * alpha / 2))
        ci_upper = float(np.percentile(null_distribution, 100 * (1 - alpha / 2)))
    else:
        ci_lower = float(np.nan)
        ci_upper = float(np.nan)

    return {
        "observed_stat": observed_stat,
        "p_value": p_value,
        "null_distribution": null_distribution,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# ==============================================================================
# Legacy Function (Backward Compatibility)
# ==============================================================================


def stratified_permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    metric_type: Literal["TPR", "FPR", "TNR", "Independence"] = "Independence",
    n_perms: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> dict:
    """
    Legacy function for stratified permutation test.

    DEPRECATED: Use stratified_cluster_permutation_test with DataFrame input.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        groups: Group assignments.
        metric_fn: Function(y_true, y_pred, groups) -> statistic.
        metric_type: Type of metric.
        n_perms: Number of permutations.
        alpha: Significance level.
        random_state: Random seed.

    Returns:
        Dict with test results.
    """
    warnings.warn(
        "stratified_permutation_test is deprecated. "
        "Use stratified_cluster_permutation_test with DataFrame input.",
        DeprecationWarning,
        stacklevel=2,
    )

    rng = np.random.default_rng(random_state)

    # Determine stratum filter
    if metric_type == "TPR":
        stratum_mask = y_true == 1
    elif metric_type in ("FPR", "TNR"):
        stratum_mask = y_true == 0
    else:
        stratum_mask = np.ones(len(y_true), dtype=bool)

    stratum_y_true = y_true[stratum_mask]
    stratum_y_pred = y_pred[stratum_mask]
    stratum_groups = groups[stratum_mask]

    if len(stratum_y_true) == 0:
        return {
            "observed_stat": np.nan,
            "p_value": np.nan,
            "null_distribution": np.array([]),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }

    # Observed statistic
    observed_stat = metric_fn(stratum_y_true, stratum_y_pred, stratum_groups)

    # Permutation loop
    null_distribution = np.zeros(n_perms)
    for i in range(n_perms):
        permuted_groups = stratum_groups.copy()
        rng.shuffle(permuted_groups)
        null_distribution[i] = metric_fn(stratum_y_true, stratum_y_pred, permuted_groups)

    # P-value with +1 correction
    count_extreme = np.sum(np.abs(null_distribution) >= np.abs(observed_stat))
    p_value = (1 + count_extreme) / (1 + n_perms)

    # CI from null distribution
    ci_lower = np.percentile(null_distribution, 100 * alpha / 2)
    ci_upper = np.percentile(null_distribution, 100 * (1 - alpha / 2))

    return {
        "observed_stat": observed_stat,
        "p_value": p_value,
        "null_distribution": null_distribution,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
