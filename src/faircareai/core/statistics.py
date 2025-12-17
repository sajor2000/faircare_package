"""
FairCareAI Statistical Methods Module

Implements rigorous statistical methods for fairness analysis:
1. Wilson Score CI for proportions (Brown et al. 2001)
2. Newcombe-Wilson CI for differences (Newcombe 1998)
3. Katz Log-Method CI for ratios - 80% rule (Katz et al. 1978)
4. Cluster bootstrap CI (preserves hierarchical structure)
5. Sample size adequacy (stratum-specific Rule of 5)
6. Multiplicity control (Holm-Bonferroni, BH-FDR)
7. Disparate impact decision logic

Methodology: Van Calster et al. (2025).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy import stats

# ==============================================================================
# Configuration Dataclass
# ==============================================================================


@dataclass
class AnalysisContext:
    """Configuration for statistical analysis context.

    Attributes:
        cluster_col: Column for patient-level clustering (e.g., 'patient_id').
        site_col: Column for site-level stratification (e.g., 'hospital_id').
        threshold: Decision threshold for binary classification.
        n_bootstrap: Number of bootstrap iterations for CIs.
        n_permutations: Number of permutations for hypothesis tests.
        alpha: Significance level (default 0.05).
        multiplicity_method: Method for multiple testing correction.
    """

    cluster_col: str | None = None
    site_col: str | None = None
    threshold: float = 0.5
    n_bootstrap: int = 2000
    n_permutations: int = 2000
    alpha: float = 0.05
    multiplicity_method: Literal["holm", "fdr_bh", "none"] = "fdr_bh"


# ==============================================================================
# Wilson Score CI
# ==============================================================================


def ci_wilson(
    successes: int,
    trials: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    The Wilson score interval has better coverage properties than the
    Wald interval, especially near 0 or 1.

    Args:
        successes: Number of successes (positive outcomes).
        trials: Total number of trials.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound) for the proportion.

    Reference:
        Wilson, E.B. (1927). Probable inference.
        Brown, Cai, DasGupta (2001). Interval Estimation for a Binomial Proportion.
    """
    if trials == 0:
        return (np.nan, np.nan)

    if successes == 0:
        # Lower bound is 0 for 0 successes
        z = stats.norm.ppf(1 - alpha / 2)
        upper = (z**2) / (trials + z**2)
        return (0.0, upper)

    if successes == trials:
        # Upper bound is 1 for all successes
        z = stats.norm.ppf(1 - alpha / 2)
        lower = trials / (trials + z**2)
        return (lower, 1.0)

    p_hat = successes / trials
    z = stats.norm.ppf(1 - alpha / 2)
    z2 = z * z

    denominator = 1 + z2 / trials
    center = (p_hat + z2 / (2 * trials)) / denominator
    margin = (z / denominator) * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * trials)) / trials)

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


# ==============================================================================
# Newcombe-Wilson CI for Difference
# ==============================================================================


def ci_newcombe_wilson(
    successes1: int,
    trials1: int,
    successes2: int,
    trials2: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Compute Newcombe-Wilson CI for difference of two proportions.

    Uses the Newcombe hybrid score method (Method 10 in Newcombe 1998).
    Returns CI for p1 - p2 (first group minus second group).

    Args:
        successes1: Successes in group 1.
        trials1: Trials in group 1.
        successes2: Successes in group 2.
        trials2: Trials in group 2.
        alpha: Significance level.

    Returns:
        Tuple of (lower, upper) for the difference p1 - p2.

    Reference:
        Newcombe, R.G. (1998). Interval estimation for the difference
        between independent proportions.
    """
    if trials1 == 0 or trials2 == 0:
        return (np.nan, np.nan)

    p1 = successes1 / trials1
    p2 = successes2 / trials2
    diff = p1 - p2

    # Get Wilson CIs for each proportion
    l1, u1 = ci_wilson(successes1, trials1, alpha)
    l2, u2 = ci_wilson(successes2, trials2, alpha)

    # Handle NaN cases
    if np.isnan(l1) or np.isnan(l2):
        return (np.nan, np.nan)

    # Newcombe hybrid method
    lower = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)

    return (max(-1.0, lower), min(1.0, upper))


# ==============================================================================
# Katz Log-Method CI for Ratio (80% Rule)
# ==============================================================================


def ci_ratio_katz(
    successes1: int,
    trials1: int,
    successes2: int,
    trials2: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Compute Katz log-method CI for risk ratio (p1/p2).

    Critical for validating the 80% Rule in disparate impact analysis.
    Uses Haldane-Anscombe correction (+0.5) for zero counts.

    Args:
        successes1: Successes in group 1 (numerator group).
        trials1: Trials in group 1.
        successes2: Successes in group 2 (reference/denominator group).
        trials2: Trials in group 2.
        alpha: Significance level.

    Returns:
        Tuple of (lower, upper) for the risk ratio p1/p2.

    Reference:
        Katz, Baptista, Azen, Pike (1978). Obtaining confidence intervals
        for the risk ratio in cohort studies.
    """
    if trials1 == 0 or trials2 == 0:
        return (np.nan, np.nan)

    # Haldane-Anscombe correction for zero counts
    s1 = successes1 + 0.5 if successes1 == 0 or successes1 == trials1 else successes1
    s2 = successes2 + 0.5 if successes2 == 0 or successes2 == trials2 else successes2
    t1 = trials1 + 1 if successes1 == 0 or successes1 == trials1 else trials1
    t2 = trials2 + 1 if successes2 == 0 or successes2 == trials2 else trials2

    p1 = s1 / t1
    p2 = s2 / t2

    if p2 == 0:
        return (np.nan, np.nan)

    ratio = p1 / p2
    log_ratio = math.log(ratio)

    # Standard error in log scale
    se_log = math.sqrt((1 - p1) / s1 + (1 - p2) / s2)

    z = stats.norm.ppf(1 - alpha / 2)

    lower = math.exp(log_ratio - z * se_log)
    upper = math.exp(log_ratio + z * se_log)

    return (lower, upper)


# ==============================================================================
# Disparate Impact Decision
# ==============================================================================


@dataclass(frozen=True)
class DisparateImpactDecision:
    """Result of disparate impact assessment.

    Attributes:
        decision: One of 'violation_supported', 'compliant', 'inconclusive', 'insufficient_data'.
        ratio: Point estimate of the selection ratio.
        ci_lower: Lower bound of CI.
        ci_upper: Upper bound of CI.
        threshold: Threshold used for decision.
    """

    decision: Literal["violation_supported", "compliant", "inconclusive", "insufficient_data"]
    ratio: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    threshold: float = 0.80


def disparate_impact_decision(
    ci_lower: float,
    ci_upper: float,
    threshold: float = 0.80,
) -> DisparateImpactDecision:
    """
    Make disparate impact decision based on CI relative to threshold.

    Uses CI-based decision logic:
    - violation_supported: Entire CI below threshold
    - compliant: Entire CI at or above threshold
    - inconclusive: CI spans threshold
    - insufficient_data: CI cannot be computed (NaN)

    Args:
        ci_lower: Lower bound of ratio CI.
        ci_upper: Upper bound of ratio CI.
        threshold: Threshold for 80% rule (default 0.80).

    Returns:
        DisparateImpactDecision with decision and details.
    """
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        return DisparateImpactDecision(
            decision="insufficient_data",
            threshold=threshold,
        )

    if ci_upper < threshold:
        return DisparateImpactDecision(
            decision="violation_supported",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            threshold=threshold,
        )
    elif ci_lower >= threshold:
        return DisparateImpactDecision(
            decision="compliant",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            threshold=threshold,
        )
    else:
        return DisparateImpactDecision(
            decision="inconclusive",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            threshold=threshold,
        )


def compute_disparate_impact_with_ci(
    successes1: int,
    trials1: int,
    successes2: int,
    trials2: int,
    threshold: float = 0.80,
    alpha: float = 0.05,
) -> DisparateImpactDecision:
    """
    Compute disparate impact ratio with CI and make decision.

    Args:
        successes1: Successes in comparison group.
        trials1: Trials in comparison group.
        successes2: Successes in reference group.
        trials2: Trials in reference group.
        threshold: Threshold for 80% rule.
        alpha: Significance level.

    Returns:
        DisparateImpactDecision with full results.
    """
    ci_lower, ci_upper = ci_ratio_katz(successes1, trials1, successes2, trials2, alpha)

    if trials1 > 0 and trials2 > 0:
        p1 = successes1 / trials1
        p2 = successes2 / trials2
        ratio = p1 / p2 if p2 > 0 else np.nan
    else:
        ratio = np.nan

    decision = disparate_impact_decision(ci_lower, ci_upper, threshold)

    return DisparateImpactDecision(
        decision=decision.decision,
        ratio=ratio,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        threshold=threshold,
    )


# ==============================================================================
# Bootstrap CI
# ==============================================================================


def bootstrap_ci_simple(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> tuple[float, tuple[float, float]]:
    """
    Compute bootstrap CI using percentile method.

    Args:
        data: Array of data values.
        statistic_fn: Function to compute statistic from data.
        n_bootstrap: Number of bootstrap iterations.
        alpha: Significance level.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, (ci_lower, ci_upper)).
    """
    rng = np.random.default_rng(random_state)
    n = len(data)

    if n == 0:
        return (np.nan, (np.nan, np.nan))

    point_estimate = statistic_fn(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_sample = data[indices]
        bootstrap_stats[i] = statistic_fn(bootstrap_sample)

    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return (point_estimate, (float(ci_lower), float(ci_upper)))


def cluster_bootstrap_ci(
    df: pl.DataFrame,
    cluster_col: str,
    statistic_fn: Callable[[pl.DataFrame], float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> tuple[float, tuple[float, float]]:
    """
    Compute cluster-aware bootstrap CI.

    Resamples at the cluster level (e.g., patient) rather than
    observation level to account for within-cluster correlation.

    Args:
        df: Polars DataFrame with data.
        cluster_col: Column identifying clusters (e.g., 'patient_id').
        statistic_fn: Function to compute statistic from DataFrame.
        n_bootstrap: Number of bootstrap iterations.
        alpha: Significance level.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, (ci_lower, ci_upper)).
    """
    rng = np.random.default_rng(random_state)

    # Get unique clusters
    clusters = df[cluster_col].unique().to_numpy()
    n_clusters = len(clusters)

    if n_clusters == 0:
        return (np.nan, (np.nan, np.nan))

    point_estimate = statistic_fn(df)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample clusters with replacement
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)

        # Build resampled DataFrame
        resampled_dfs = []
        for cluster_id in sampled_clusters:
            cluster_data = df.filter(pl.col(cluster_col) == cluster_id)
            resampled_dfs.append(cluster_data)

        if resampled_dfs:
            resampled_df = pl.concat(resampled_dfs)
            bootstrap_stats[i] = statistic_fn(resampled_df)
        else:
            bootstrap_stats[i] = np.nan

    # Remove NaN values for percentile calculation
    valid_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]
    if len(valid_stats) == 0:
        return (point_estimate, (np.nan, np.nan))

    ci_lower = np.percentile(valid_stats, 100 * alpha / 2)
    ci_upper = np.percentile(valid_stats, 100 * (1 - alpha / 2))

    return (point_estimate, (ci_lower, ci_upper))


# ==============================================================================
# Sample Size Adequacy
# ==============================================================================


@dataclass
class SampleAdequacy:
    """Result of sample adequacy assessment.

    Attributes:
        status: One of 'ADEQUATE', 'MODERATE', 'LOW', 'VERY_LOW'.
        n_total: Total sample size.
        n_positive: Number of positive cases.
        n_negative: Number of negative cases.
        warning: Warning message if applicable.
    """

    status: Literal["ADEQUATE", "MODERATE", "LOW", "VERY_LOW"]
    n_total: int
    n_positive: int | None = None
    n_negative: int | None = None
    warning: str | None = None


def assess_sample_adequacy(
    n_total: int,
    n_positive: int | None = None,
    n_negative: int | None = None,
) -> SampleAdequacy:
    """
    Assess sample size adequacy following Rule of 5.

    Args:
        n_total: Total sample size.
        n_positive: Number of positive cases (optional).
        n_negative: Number of negative cases (optional).

    Returns:
        SampleAdequacy with status and details.
    """
    # Basic sample size checks
    if n_total < 5:
        return SampleAdequacy(
            status="VERY_LOW",
            n_total=n_total,
            n_positive=n_positive,
            n_negative=n_negative,
            warning=(
                f"CAUTION: Sample size n={n_total} < 5. "
                "Results are highly unreliable. Consider suppressing."
            ),
        )

    if n_total < 10:
        return SampleAdequacy(
            status="VERY_LOW",
            n_total=n_total,
            n_positive=n_positive,
            n_negative=n_negative,
            warning=(f"CAUTION: Sample size n={n_total} < 10. Results are highly uncertain."),
        )

    # Check positive/negative counts if provided
    if n_positive is not None and n_negative is not None:
        if n_positive < 5 or n_negative < 5:
            warning = ""
            if n_positive < 5:
                warning = f"Only {n_positive} positives (need 5+ for stable TPR estimates). "
            if n_negative < 5:
                warning += f"Only {n_negative} negatives (need 5+ for stable FPR estimates)."

            return SampleAdequacy(
                status="VERY_LOW",
                n_total=n_total,
                n_positive=n_positive,
                n_negative=n_negative,
                warning=warning.strip(),
            )

        if n_positive < 10 or n_negative < 10:
            return SampleAdequacy(
                status="LOW",
                n_total=n_total,
                n_positive=n_positive,
                n_negative=n_negative,
                warning=(
                    f"Limited events: {n_positive} positives, {n_negative} negatives. "
                    "Confidence intervals may be wide."
                ),
            )

    if n_total < 30:
        return SampleAdequacy(
            status="LOW",
            n_total=n_total,
            n_positive=n_positive,
            n_negative=n_negative,
            warning=f"Sample size n={n_total} < 30. Interpret with caution.",
        )

    if n_total < 50:
        return SampleAdequacy(
            status="MODERATE",
            n_total=n_total,
            n_positive=n_positive,
            n_negative=n_negative,
            warning=None,
        )

    return SampleAdequacy(
        status="ADEQUATE",
        n_total=n_total,
        n_positive=n_positive,
        n_negative=n_negative,
        warning=None,
    )


# ==============================================================================
# Stratum-Specific Adequacy
# ==============================================================================


@dataclass
class StratumAdequacy:
    """Result of stratum-specific adequacy assessment.

    Attributes:
        status: One of 'REPORT', 'FLAG', 'SUPPRESS'.
        unique_patients: Number of unique patients in stratum.
        warning: Warning or action message.
    """

    status: Literal["REPORT", "FLAG", "SUPPRESS"]
    unique_patients: int
    warning: str | None = None


def assess_stratum_adequacy(
    df: pl.DataFrame,
    group_col: str,
    group_value: str,
    metric_type: Literal["TPR", "FPR", "TNR"],
    y_true_col: str,
    cluster_col: str | None = None,
) -> StratumAdequacy:
    """
    Assess sample adequacy for a specific metric stratum.

    TPR requires y_true==1, FPR/TNR require y_true==0.

    Args:
        df: DataFrame with data.
        group_col: Column for group membership.
        group_value: Value to filter for.
        metric_type: Type of metric (determines stratum).
        y_true_col: Column for true labels.
        cluster_col: Column for clustering (e.g., patient_id).

    Returns:
        StratumAdequacy with status and details.
    """
    # Filter to group
    group_df = df.filter(pl.col(group_col) == group_value)

    # Filter to relevant stratum
    if metric_type == "TPR":
        stratum_df = group_df.filter(pl.col(y_true_col) == 1)
    else:  # FPR or TNR
        stratum_df = group_df.filter(pl.col(y_true_col) == 0)

    # Count unique patients (or rows if no cluster col)
    if cluster_col and cluster_col in stratum_df.columns:
        n_unique = stratum_df[cluster_col].n_unique()
    else:
        n_unique = len(stratum_df)

    # Decision thresholds
    if n_unique < 5:
        return StratumAdequacy(
            status="SUPPRESS",
            unique_patients=n_unique,
            warning=f"SUPPRESS: Only {n_unique} unique patients in {metric_type} stratum for {group_value}.",
        )

    if n_unique < 30:
        return StratumAdequacy(
            status="FLAG",
            unique_patients=n_unique,
            warning=f"FLAG: {n_unique} patients in {metric_type} stratum for {group_value}. Use caution.",
        )

    return StratumAdequacy(
        status="REPORT",
        unique_patients=n_unique,
        warning=None,
    )


# ==============================================================================
# Effective Sample Size
# ==============================================================================


def get_effective_sample_size(
    df: pl.DataFrame,
    cluster_col: str | None = None,
) -> int:
    """
    Get effective sample size accounting for clustering.

    Args:
        df: DataFrame with data.
        cluster_col: Column for clustering. If None, returns row count.

    Returns:
        Effective sample size (unique clusters or rows).
    """
    if cluster_col is None or cluster_col not in df.columns:
        return len(df)

    return df[cluster_col].n_unique()


# ==============================================================================
# Multiplicity Control
# ==============================================================================


def adjust_pvalues_holm(pvalues: np.ndarray) -> np.ndarray:
    """
    Apply Holm-Bonferroni correction for FWER control.

    Args:
        pvalues: Array of p-values.

    Returns:
        Adjusted p-values (same length as input).
    """
    n = len(pvalues)
    if n == 0:
        return np.array([])

    # Get sorted indices
    order = np.argsort(pvalues)
    sorted_pvals = pvalues[order]

    # Holm adjustment
    adjusted = np.zeros(n)
    cummax = 0.0
    for i in range(n):
        multiplier = n - i
        adj_p = min(1.0, sorted_pvals[i] * multiplier)
        cummax = max(cummax, adj_p)
        adjusted[i] = cummax

    # Restore original order
    result = np.zeros(n)
    result[order] = adjusted

    return result


def adjust_pvalues_fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    """
    Apply Benjamini-Hochberg correction for FDR control.

    Args:
        pvalues: Array of p-values.

    Returns:
        Adjusted p-values (same length as input).
    """
    n = len(pvalues)
    if n == 0:
        return np.array([])

    # Get sorted indices
    order = np.argsort(pvalues)
    sorted_pvals = pvalues[order]

    # BH adjustment
    adjusted = np.zeros(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        multiplier = n / (i + 1)
        adj_p = min(1.0, sorted_pvals[i] * multiplier)
        cummin = min(cummin, adj_p)
        adjusted[i] = cummin

    # Restore original order
    result = np.zeros(n)
    result[order] = adjusted

    return result


def adjust_pvalues(
    pvalues: np.ndarray,
    method: Literal["holm", "fdr_bh", "none"] = "fdr_bh",
) -> np.ndarray:
    """
    Adjust p-values for multiple testing.

    Args:
        pvalues: Array of p-values.
        method: Correction method ('holm', 'fdr_bh', or 'none').

    Returns:
        Adjusted p-values.
    """
    if method == "holm":
        return adjust_pvalues_holm(pvalues)
    elif method == "fdr_bh":
        return adjust_pvalues_fdr_bh(pvalues)
    else:
        return pvalues.copy()
