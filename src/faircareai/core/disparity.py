"""
Disparity Analysis Module

Compares fairness metrics between demographic groups with
statistical significance testing and confidence intervals.

Status values ("pass", "warn", "fail") indicate position relative to
configured thresholds. Interpretation of these statuses rests with
your organization's governance process.
"""

from dataclasses import dataclass
from typing import Literal

import polars as pl

from faircareai.core.statistical import (
    newcombe_wilson_ci,
    z_test_two_proportions,
)


@dataclass
class DisparityResult:
    """Container for disparity analysis between two groups."""

    reference_group: str
    comparison_group: str
    metric: str
    reference_value: float
    comparison_value: float
    difference: float  # comparison - reference
    diff_ci_lower: float
    diff_ci_upper: float
    ratio: float  # comparison / reference (disparate impact)
    status: Literal["pass", "warn", "fail"]
    p_value: float | None
    statistically_significant: bool


def compute_disparity(
    reference_group: str,
    reference_value: float,
    reference_successes: int,
    reference_trials: int,
    comparison_group: str,
    comparison_value: float,
    comparison_successes: int,
    comparison_trials: int,
    metric: str,
    warn_threshold: float = 0.05,
    fail_threshold: float = 0.10,
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> DisparityResult:
    """
    Compute disparity between reference and comparison group.

    Args:
        reference_group: Name of reference group.
        reference_value: Metric value for reference group.
        reference_successes: Numerator for reference group.
        reference_trials: Denominator for reference group.
        comparison_group: Name of comparison group.
        comparison_value: Metric value for comparison group.
        comparison_successes: Numerator for comparison group.
        comparison_trials: Denominator for comparison group.
        metric: Name of the metric being compared.
        warn_threshold: Absolute difference to trigger warning (default 0.05).
        fail_threshold: Absolute difference to trigger failure (default 0.10).
        alpha: Significance level for hypothesis test (default 0.05).
        confidence: Confidence level for CI (default 0.95).

    Returns:
        DisparityResult with difference, CI, ratio, and status.
    """
    difference = comparison_value - reference_value
    abs_diff = abs(difference)

    # Compute CI for difference
    ci_lower, ci_upper = newcombe_wilson_ci(
        reference_successes,
        reference_trials,
        comparison_successes,
        comparison_trials,
        confidence,
    )

    # Compute ratio (disparate impact)
    if reference_value > 0:
        ratio = comparison_value / reference_value
    else:
        ratio = float("inf") if comparison_value > 0 else 1.0

    # Hypothesis test
    z_stat, p_value = z_test_two_proportions(
        reference_successes,
        reference_trials,
        comparison_successes,
        comparison_trials,
    )
    statistically_significant = p_value < alpha

    # Determine status based on magnitude relative to configured thresholds
    status: Literal["pass", "warn", "fail"]
    if abs_diff >= fail_threshold:
        status = "fail"  # Outside threshold
    elif abs_diff >= warn_threshold:
        status = "warn"  # Near threshold
    else:
        status = "pass"  # Within threshold

    return DisparityResult(
        reference_group=reference_group,
        comparison_group=comparison_group,
        metric=metric,
        reference_value=reference_value,
        comparison_value=comparison_value,
        difference=difference,
        diff_ci_lower=ci_lower,
        diff_ci_upper=ci_upper,
        ratio=ratio,
        status=status,
        p_value=p_value,
        statistically_significant=statistically_significant,
    )


def compute_disparities(
    metrics_df: pl.DataFrame,
    metric: str = "tpr",
    reference_group: str | None = None,
    reference_strategy: Literal["largest", "best", "specified"] = "largest",
    warn_threshold: float = 0.05,
    fail_threshold: float = 0.10,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """
    Compute disparities for all groups against a reference.

    Args:
        metrics_df: DataFrame from compute_group_metrics().
        metric: Which metric to compare (default: tpr).
        reference_group: Specific group to use as reference (if strategy="specified").
        reference_strategy: How to select reference:
            - "largest": Group with most samples (default)
            - "best": Group with best metric value
            - "specified": Use reference_group parameter
        warn_threshold: Difference to trigger warning.
        fail_threshold: Difference to trigger failure.
        alpha: Significance level.

    Returns:
        Polars DataFrame with columns:
        - reference_group, comparison_group
        - reference_value, comparison_value
        - difference, diff_ci_lower, diff_ci_upper
        - ratio, status, p_value, statistically_significant
    """
    # Filter out overall row
    df = metrics_df.filter(pl.col("group") != "_overall")

    if len(df) < 2:
        return pl.DataFrame()

    # Determine reference group
    if reference_strategy == "largest":
        ref_row = df.sort("n", descending=True).head(1)
        ref_group = ref_row["group"][0]
    elif reference_strategy == "best":
        ref_row = df.sort(metric, descending=True).head(1)
        ref_group = ref_row["group"][0]
    elif reference_strategy == "specified":
        if reference_group is None:
            raise ValueError("reference_group required when strategy='specified'")
        ref_group = reference_group
        ref_row = df.filter(pl.col("group") == ref_group)
        if len(ref_row) == 0:
            raise ValueError(f"Reference group '{ref_group}' not found in data")
    else:
        raise ValueError(f"Unknown reference_strategy: {reference_strategy}")

    ref_value = ref_row[metric][0]
    ref_n_pos = ref_row["n_positive"][0]

    # For TPR: successes = TP, trials = TP + FN = n_positive
    # We need to back-calculate from the metric value
    ref_successes = int(round(ref_value * ref_n_pos))
    ref_trials = ref_n_pos

    # Compute disparities for each non-reference group
    results = []

    for row in df.iter_rows(named=True):
        if row["group"] == ref_group:
            continue

        comp_value = row[metric]
        comp_n_pos = row["n_positive"]
        comp_successes = int(round(comp_value * comp_n_pos))
        comp_trials = comp_n_pos

        disparity = compute_disparity(
            reference_group=ref_group,
            reference_value=ref_value,
            reference_successes=ref_successes,
            reference_trials=ref_trials,
            comparison_group=row["group"],
            comparison_value=comp_value,
            comparison_successes=comp_successes,
            comparison_trials=comp_trials,
            metric=metric,
            warn_threshold=warn_threshold,
            fail_threshold=fail_threshold,
            alpha=alpha,
        )

        results.append(
            {
                "reference_group": disparity.reference_group,
                "comparison_group": disparity.comparison_group,
                "metric": disparity.metric,
                "reference_value": disparity.reference_value,
                "comparison_value": disparity.comparison_value,
                "difference": disparity.difference,
                "diff_ci_lower": disparity.diff_ci_lower,
                "diff_ci_upper": disparity.diff_ci_upper,
                "ratio": disparity.ratio,
                "status": disparity.status,
                "p_value": disparity.p_value,
                "statistically_significant": int(disparity.statistically_significant),
            }
        )

    # Define explicit schema to ensure consistent types
    schema = {
        "reference_group": pl.Utf8,
        "comparison_group": pl.Utf8,
        "metric": pl.Utf8,
        "reference_value": pl.Float64,
        "comparison_value": pl.Float64,
        "difference": pl.Float64,
        "diff_ci_lower": pl.Float64,
        "diff_ci_upper": pl.Float64,
        "ratio": pl.Float64,
        "status": pl.Utf8,
        "p_value": pl.Float64,
        "statistically_significant": pl.Int64,
    }

    return pl.DataFrame(results, schema=schema)


def get_worst_disparity(
    disparities_df: pl.DataFrame,
) -> tuple[str, str, float] | None:
    """
    Find the worst disparity in the results.

    Returns:
        Tuple of (group, metric, value) or None if no disparities.
    """
    if len(disparities_df) == 0:
        return None

    # Sort by absolute difference descending
    worst = (
        disparities_df.with_columns(pl.col("difference").abs().alias("abs_diff"))
        .sort("abs_diff", descending=True)
        .head(1)
    )

    return (
        worst["comparison_group"][0],
        worst["metric"][0],
        worst["difference"][0],
    )


def count_by_status(disparities_df: pl.DataFrame) -> dict[str, int]:
    """
    Count disparities by status.

    Returns:
        Dictionary with pass/warn/fail counts.
    """
    if len(disparities_df) == 0:
        return {"pass": 0, "warn": 0, "fail": 0}

    counts = disparities_df.group_by("status").len()

    result = {"pass": 0, "warn": 0, "fail": 0}
    for row in counts.iter_rows(named=True):
        result[row["status"]] = row["len"]

    return result
