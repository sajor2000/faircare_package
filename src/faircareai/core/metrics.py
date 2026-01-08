"""
Fairness Metrics Computation

Computes group-level performance metrics with confidence intervals.
Follows statistical standards from CLAUDE.md.
"""

from dataclasses import dataclass

import polars as pl

from faircareai.core.statistics import ci_wilson
from faircareai.core.statistical import (
    clopper_pearson_ci,
    get_sample_status,
    get_sample_warning,
    wilson_score_ci,
)


@dataclass
class GroupMetrics:
    """Container for group-level fairness metrics."""

    group: str
    n: int
    n_positive: int
    n_negative: int
    tp: int
    fp: int
    tn: int
    fn: int
    tpr: float  # True Positive Rate (Sensitivity/Recall)
    fpr: float  # False Positive Rate
    tnr: float  # True Negative Rate (Specificity)
    fnr: float  # False Negative Rate
    ppv: float  # Positive Predictive Value (Precision)
    npv: float  # Negative Predictive Value
    accuracy: float
    ci_method: str
    sample_status: str
    warning: str | None


def _compute_confusion_matrix(
    y_true: pl.Series,
    y_pred: pl.Series,
) -> tuple[int, int, int, int]:
    """Compute confusion matrix values."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def _safe_divide(numerator: int, denominator: int, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    return numerator / denominator if denominator > 0 else default


def compute_metrics_for_group(
    df: pl.DataFrame,
    group_name: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    confidence: float = 0.95,
) -> GroupMetrics:
    """
    Compute all fairness metrics for a single group.

    Args:
        df: DataFrame containing predictions for this group.
        group_name: Name of the group.
        y_true_col: Column name for ground truth.
        y_pred_col: Column name for predictions.
        confidence: Confidence level for CIs.

    Returns:
        GroupMetrics dataclass with all metrics and CIs.
    """
    y_true = df[y_true_col]
    y_pred = df[y_pred_col]

    n = len(df)
    n_positive = int(y_true.sum())
    n_negative = n - n_positive

    tp, fp, tn, fn = _compute_confusion_matrix(y_true, y_pred)

    # Compute rates
    tpr = _safe_divide(tp, tp + fn)  # Sensitivity
    fpr = _safe_divide(fp, fp + tn)
    tnr = _safe_divide(tn, tn + fp)  # Specificity
    fnr = _safe_divide(fn, fn + tp)
    ppv = _safe_divide(tp, tp + fp)  # Precision
    npv = _safe_divide(tn, tn + fn)
    accuracy = _safe_divide(tp + tn, n)

    # Determine CI method based on proportions
    ci_method = "wilson"
    if n_positive < n * 0.01 or n_positive > n * 0.99:
        ci_method = "clopper_pearson"

    # Get sample status
    sample_status = get_sample_status(n, n_positive)
    warning = get_sample_warning(group_name, n, n_positive, sample_status)

    return GroupMetrics(
        group=group_name,
        n=n,
        n_positive=n_positive,
        n_negative=n_negative,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        tpr=tpr,
        fpr=fpr,
        tnr=tnr,
        fnr=fnr,
        ppv=ppv,
        npv=npv,
        accuracy=accuracy,
        ci_method=ci_method,
        sample_status=sample_status,
        warning=warning,
    )


def compute_metric_ci(
    metric: str,
    group_metrics: GroupMetrics,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute confidence interval for a specific metric.

    Args:
        metric: Metric name (tpr, fpr, ppv, etc.)
        group_metrics: GroupMetrics object with confusion matrix.
        confidence: Confidence level.

    Returns:
        Tuple of (lower, upper) CI bounds.
    """
    ci_func = (
        clopper_pearson_ci if group_metrics.ci_method == "clopper_pearson" else wilson_score_ci
    )

    # Map metric to numerator/denominator
    metric_map = {
        "tpr": (group_metrics.tp, group_metrics.tp + group_metrics.fn),
        "fpr": (group_metrics.fp, group_metrics.fp + group_metrics.tn),
        "tnr": (group_metrics.tn, group_metrics.tn + group_metrics.fp),
        "fnr": (group_metrics.fn, group_metrics.fn + group_metrics.tp),
        "ppv": (group_metrics.tp, group_metrics.tp + group_metrics.fp),
        "npv": (group_metrics.tn, group_metrics.tn + group_metrics.fn),
        "accuracy": (
            group_metrics.tp + group_metrics.tn,
            group_metrics.n,
        ),
    }

    if metric not in metric_map:
        raise ValueError(f"Unknown metric: {metric}")

    successes, trials = metric_map[metric]
    return ci_func(successes, trials, confidence)


def compute_group_metrics(
    df: pl.DataFrame,
    group_col: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    metrics: list[str] | None = None,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """
    Compute fairness metrics for all groups in a column.

    Args:
        df: DataFrame with predictions and demographics.
        group_col: Column name for demographic grouping.
        y_true_col: Column name for ground truth.
        y_pred_col: Column name for predictions.
        metrics: List of metrics to include (default: tpr, fpr, ppv).
        confidence: Confidence level for CIs.

    Returns:
        Polars DataFrame with metrics per group, including:
        - group: Group name
        - n: Sample size
        - n_positive: Number of positive cases
        - {metric}: Metric value for each requested metric
        - {metric}_ci_lower: Lower CI bound
        - {metric}_ci_upper: Upper CI bound
        - sample_status: ADEQUATE/MODERATE/LOW/VERY_LOW
        - warning: Sample size warning message or None
    """
    if metrics is None:
        metrics = ["tpr", "fpr", "ppv"]

    # Get unique groups
    groups = df[group_col].unique().sort().to_list()

    # Compute metrics for each group
    rows = []

    for group in groups:
        group_df = df.filter(pl.col(group_col) == group)
        gm = compute_metrics_for_group(
            group_df,
            str(group),
            y_true_col,
            y_pred_col,
            confidence,
        )

        row = {
            "group": gm.group,
            "n": gm.n,
            "n_positive": gm.n_positive,
            "sample_status": gm.sample_status,
            "warning": gm.warning or "",  # Use empty string instead of None
        }

        for metric in metrics:
            value = getattr(gm, metric)
            row[metric] = value

            ci_lower, ci_upper = compute_metric_ci(metric, gm, confidence)
            row[f"{metric}_ci_lower"] = ci_lower
            row[f"{metric}_ci_upper"] = ci_upper

        rows.append(row)

    # Add overall row
    overall_gm = compute_metrics_for_group(df, "_overall", y_true_col, y_pred_col, confidence)

    overall_row = {
        "group": "_overall",
        "n": overall_gm.n,
        "n_positive": overall_gm.n_positive,
        "sample_status": overall_gm.sample_status,
        "warning": "",  # Empty string for overall
    }

    for metric in metrics:
        value = getattr(overall_gm, metric)
        overall_row[metric] = value

        ci_lower, ci_upper = compute_metric_ci(metric, overall_gm, confidence)
        overall_row[f"{metric}_ci_lower"] = ci_lower
        overall_row[f"{metric}_ci_upper"] = ci_upper

    rows.append(overall_row)

    return pl.DataFrame(rows)
