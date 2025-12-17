"""
FairCareAI Van Calster Performance Metrics Module

Implements the four key performance measures from Van Calster et al. (2025)
for evaluating predictive AI models, computed both overall and by subgroup:

1. AUROC by subgroup: Discrimination measure
2. Calibration by subgroup: Detecting differential miscalibration
3. Net Benefit by subgroup: Clinical utility across groups
4. Risk Distribution by subgroup: Probability distributions by outcome

Methodology Reference:
    Van Calster B, Collins GS, Vickers AJ, et al. Evaluation of performance
    measures in predictive artificial intelligence models to support medical
    decisions: overview and guidance. Lancet Digit Health 2025.
    https://doi.org/10.1016/j.landig.2025.100916

Healthcare organizations interpret these metrics based on their clinical
context, organizational values, and governance frameworks.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from faircareai.core.constants import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP,
    MIN_BOOTSTRAP_SAMPLES,
    MIN_SAMPLE_SIZE_CALIBRATION,
    MIN_SAMPLE_SIZE_FLAG,
    PROB_CLIP_MAX,
    PROB_CLIP_MIN,
)
from faircareai.core.logging import get_logger

logger = get_logger(__name__)

# Van Calster recommended constants
CALIBRATION_BINS_DEFAULT = 10
NET_BENEFIT_THRESHOLDS_DEFAULT = np.linspace(0.01, 0.99, 99)
AUROC_DIFF_CLINICALLY_MEANINGFUL = 0.05  # Per Van Calster guidance


def compute_vancalster_metrics(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_col: str | None = None,
    threshold: float = 0.5,
    reference: str | None = None,
    bootstrap_ci: bool = True,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    calibration_bins: int = CALIBRATION_BINS_DEFAULT,
    net_benefit_thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute all Van Calster recommended metrics overall and by subgroup.

    This is the primary entry point implementing the four key performance
    measures recommended by Van Calster et al. (2025):
    1. AUROC (discrimination)
    2. Calibration (smoothed calibration curves)
    3. Net Benefit (clinical utility via decision curve analysis)
    4. Risk Distribution (probability distributions by outcome)

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels (0/1).
        group_col: Column name for subgroup variable (optional).
        threshold: Primary decision threshold for net benefit.
        reference: Reference group for disparity calculations.
        bootstrap_ci: Whether to compute bootstrap 95% CIs.
        n_bootstrap: Number of bootstrap iterations.
        calibration_bins: Number of bins for calibration curve.
        net_benefit_thresholds: Thresholds for decision curve analysis.

    Returns:
        Dict containing:
        - overall: Metrics for entire dataset
        - by_subgroup: Metrics per subgroup (if group_col provided)
        - disparities: Subgroup differences from reference
        - interpretation: Clinical interpretation guidance
        - citation: Van Calster et al. citation

    Example:
        >>> results = compute_vancalster_metrics(
        ...     df, "risk_score", "outcome", group_col="race"
        ... )
        >>> print(results["overall"]["auroc"])
        0.847
    """
    results: dict[str, Any] = {
        "citation": (
            "Van Calster B, et al. Evaluation of performance measures in "
            "predictive AI models to support medical decisions. "
            "Lancet Digit Health 2025. doi:10.1016/j.landig.2025.100916"
        ),
        "methodology": "Van Calster et al. (2025) recommended core metrics",
        "threshold": threshold,
    }

    if net_benefit_thresholds is None:
        net_benefit_thresholds = NET_BENEFIT_THRESHOLDS_DEFAULT

    # Get overall arrays
    y_true = df[y_true_col].to_numpy()
    y_prob = df[y_prob_col].to_numpy()

    # === OVERALL METRICS ===
    results["overall"] = _compute_vancalster_single(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        bootstrap_ci=bootstrap_ci,
        n_bootstrap=n_bootstrap,
        calibration_bins=calibration_bins,
        net_benefit_thresholds=net_benefit_thresholds,
        label="Overall",
    )

    # === BY SUBGROUP METRICS ===
    if group_col is not None:
        results["by_subgroup"] = {}
        results["disparities"] = {}

        groups = df[group_col].drop_nulls().unique().sort().to_list()

        # Determine reference group (largest by default)
        if reference is None:
            group_counts = df.group_by(group_col).len().sort("len", descending=True)
            reference = group_counts[group_col][0]

        results["reference_group"] = reference

        # Compute metrics for each subgroup
        for group in groups:
            group_df = df.filter(pl.col(group_col) == group)
            y_true_g = group_df[y_true_col].to_numpy()
            y_prob_g = group_df[y_prob_col].to_numpy()

            group_metrics = _compute_vancalster_single(
                y_true=y_true_g,
                y_prob=y_prob_g,
                threshold=threshold,
                bootstrap_ci=bootstrap_ci,
                n_bootstrap=n_bootstrap,
                calibration_bins=calibration_bins,
                net_benefit_thresholds=net_benefit_thresholds,
                label=str(group),
            )
            group_metrics["is_reference"] = str(group) == str(reference)
            results["by_subgroup"][str(group)] = group_metrics

        # Compute disparities vs reference
        results["disparities"] = _compute_vancalster_disparities(
            results["by_subgroup"], str(reference), threshold
        )

        # Clinical interpretation
        results["interpretation"] = _interpret_vancalster_results(results)

    return results


def _compute_vancalster_single(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    bootstrap_ci: bool,
    n_bootstrap: int,
    calibration_bins: int,
    net_benefit_thresholds: np.ndarray,
    label: str,
) -> dict[str, Any]:
    """Compute Van Calster metrics for a single group/overall.

    Args:
        y_true: True binary outcomes.
        y_prob: Predicted probabilities.
        threshold: Decision threshold.
        bootstrap_ci: Whether to compute bootstrap CIs.
        n_bootstrap: Number of bootstrap iterations.
        calibration_bins: Number of calibration bins.
        net_benefit_thresholds: Thresholds for DCA.
        label: Label for this computation.

    Returns:
        Dict with all four Van Calster recommended metrics.
    """
    result: dict[str, Any] = {
        "label": label,
        "n": len(y_true),
        "n_events": int(np.sum(y_true)),
        "prevalence": float(np.mean(y_true)),
        "small_sample_warning": len(y_true) < MIN_SAMPLE_SIZE_FLAG,
    }

    if len(y_true) < MIN_SAMPLE_SIZE_FLAG:
        result["error"] = f"Insufficient sample size (n < {MIN_SAMPLE_SIZE_FLAG})"
        return result

    # === 1. AUROC (Discrimination) ===
    result["discrimination"] = _compute_auroc_metrics(y_true, y_prob, bootstrap_ci, n_bootstrap)

    # === 2. Calibration ===
    result["calibration"] = _compute_calibration_metrics(y_true, y_prob, calibration_bins)

    # === 3. Net Benefit (Clinical Utility) ===
    result["clinical_utility"] = _compute_net_benefit_metrics(
        y_true, y_prob, threshold, net_benefit_thresholds
    )

    # === 4. Risk Distribution ===
    result["risk_distribution"] = _compute_risk_distribution(y_true, y_prob)

    return result


def _compute_auroc_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bootstrap_ci: bool,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Compute AUROC with bootstrap confidence interval.

    Per Van Calster et al.: "AUROC is the key measure for discrimination."

    Args:
        y_true: True binary outcomes.
        y_prob: Predicted probabilities.
        bootstrap_ci: Whether to compute bootstrap CI.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with AUROC, CI, and interpretation.
    """
    result: dict[str, Any] = {}

    # Check if both classes present
    if len(np.unique(y_true)) < 2:
        result["error"] = "Single class in outcome - AUROC undefined"
        return result

    # Point estimate
    try:
        auroc = roc_auc_score(y_true, y_prob)
        result["auroc"] = float(auroc)
    except ValueError as e:
        result["error"] = f"AUROC computation failed: {e}"
        return result

    # Bootstrap CI
    if bootstrap_ci and len(y_true) >= 20:
        auroc_samples = _bootstrap_metric(y_true, y_prob, roc_auc_score, n_bootstrap)
        if len(auroc_samples) > MIN_BOOTSTRAP_SAMPLES:
            ci = np.percentile(auroc_samples, [2.5, 97.5])
            result["auroc_ci_95"] = [float(ci[0]), float(ci[1])]
            result["auroc_se"] = float(np.std(auroc_samples))
            result["auroc_ci_fmt"] = f"(95% CI: {ci[0]:.3f}-{ci[1]:.3f})"

    # Interpretation per Van Calster
    result["interpretation"] = _interpret_auroc(result.get("auroc", 0))

    return result


def _compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
) -> dict[str, Any]:
    """Compute calibration metrics with smoothed calibration curve.

    Per Van Calster et al.: "Calibration plot is the most insightful approach
    to assess calibration, particularly when smoothing is used."

    Args:
        y_true: True binary outcomes.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration curve.

    Returns:
        Dict with calibration slope, intercept, Brier score, and curve data.
    """
    result: dict[str, Any] = {}

    if len(y_true) < MIN_SAMPLE_SIZE_CALIBRATION:
        result["error"] = (
            f"Insufficient samples for calibration (n < {MIN_SAMPLE_SIZE_CALIBRATION})"
        )
        return result

    # Brier score (strictly proper per Van Calster)
    result["brier_score"] = float(brier_score_loss(y_true, y_prob))

    # O:E ratio (Observed/Expected)
    expected = np.sum(y_prob)
    observed = np.sum(y_true)
    result["oe_ratio"] = float(observed / expected) if expected > 0 else None

    # Calibration slope and intercept via logistic regression on logit(p)
    try:
        y_prob_clipped = np.clip(y_prob, PROB_CLIP_MIN, PROB_CLIP_MAX)
        log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)

        # Fit logistic regression: outcome ~ logit(predicted)
        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(log_odds, y_true)

        result["calibration_intercept"] = float(lr.intercept_[0])
        result["calibration_slope"] = float(lr.coef_[0, 0])
    except Exception as e:
        logger.warning("Calibration slope computation failed: %s", str(e))
        result["calibration_intercept"] = None
        result["calibration_slope"] = None

    # Calibration curve data (for plotting)
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        result["calibration_curve"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "n_bins": n_bins,
        }

        # ICI (Integrated Calibration Index)
        result["ici"] = float(np.mean(np.abs(prob_pred - prob_true)))
        # E_max (Maximum Calibration Error)
        result["e_max"] = float(np.max(np.abs(prob_pred - prob_true)))
    except ValueError as e:
        logger.warning("Calibration curve failed: %s", str(e))
        result["calibration_curve"] = None
        result["ici"] = None
        result["e_max"] = None

    # Interpretation
    result["interpretation"] = _interpret_calibration(result)

    return result


def _compute_net_benefit_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    thresholds: np.ndarray,
) -> dict[str, Any]:
    """Compute net benefit and decision curve analysis.

    Per Van Calster et al.: "Net benefit with decision curve analysis
    is essential to report for clinical utility assessment."

    The net benefit formula:
        NB = TP/n - FP/n * (t / (1-t))

    where t is the decision threshold.

    Args:
        y_true: True binary outcomes.
        y_prob: Predicted probabilities.
        threshold: Primary decision threshold.
        thresholds: Array of thresholds for decision curve.

    Returns:
        Dict with net benefit at threshold and decision curve data.
    """
    result: dict[str, Any] = {}
    n = len(y_true)
    prevalence = np.mean(y_true)

    # Net benefit at primary threshold
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    if threshold < 1:
        nb = tp / n - fp / n * (threshold / (1 - threshold))
    else:
        nb = 0.0

    result["net_benefit"] = float(nb)
    result["net_benefit_max"] = float(prevalence)  # Max possible NB

    # Standardized net benefit (NB / prevalence)
    if prevalence > 0:
        result["standardized_net_benefit"] = float(nb / prevalence)
    else:
        result["standardized_net_benefit"] = None

    # Decision curve analysis across thresholds
    nb_model = []
    nb_all = []
    nb_none = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        tp_t = np.sum((y_pred_t == 1) & (y_true == 1))
        fp_t = np.sum((y_pred_t == 1) & (y_true == 0))

        if t < 1:
            nb_model.append(float(tp_t / n - fp_t / n * (t / (1 - t))))
            nb_all.append(float(prevalence - (1 - prevalence) * (t / (1 - t))))
        else:
            nb_model.append(0.0)
            nb_all.append(0.0)

        nb_none.append(0.0)

    result["decision_curve"] = {
        "thresholds": thresholds.tolist(),
        "net_benefit_model": nb_model,
        "net_benefit_all": nb_all,
        "net_benefit_none": nb_none,
    }

    # Find useful range where model > treat all and > treat none
    useful_range = [float(t) for i, t in enumerate(thresholds) if nb_model[i] > max(nb_all[i], 0)]

    result["useful_range"] = {
        "min": float(min(useful_range)) if useful_range else None,
        "max": float(max(useful_range)) if useful_range else None,
        "thresholds": useful_range,
    }

    # Interpretation
    result["interpretation"] = _interpret_net_benefit(result, threshold)

    return result


def _compute_risk_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute risk distribution statistics by outcome category.

    Per Van Calster et al.: "A plot showing probability distributions
    for each outcome category provides valuable insights."

    Args:
        y_true: True binary outcomes.
        y_prob: Predicted probabilities.

    Returns:
        Dict with distribution statistics for events and non-events.
    """
    result: dict[str, Any] = {}

    # Separate by outcome
    prob_events = y_prob[y_true == 1]
    prob_nonevents = y_prob[y_true == 0]

    # Distribution for events (outcome = 1)
    if len(prob_events) > 0:
        result["events"] = {
            "n": int(len(prob_events)),
            "mean": float(np.mean(prob_events)),
            "median": float(np.median(prob_events)),
            "std": float(np.std(prob_events)),
            "min": float(np.min(prob_events)),
            "max": float(np.max(prob_events)),
            "q25": float(np.percentile(prob_events, 25)),
            "q75": float(np.percentile(prob_events, 75)),
            "iqr": float(np.percentile(prob_events, 75) - np.percentile(prob_events, 25)),
            # Histogram data for plotting
            "histogram": _compute_histogram(prob_events),
        }
    else:
        result["events"] = {"n": 0, "error": "No events in sample"}

    # Distribution for non-events (outcome = 0)
    if len(prob_nonevents) > 0:
        result["non_events"] = {
            "n": int(len(prob_nonevents)),
            "mean": float(np.mean(prob_nonevents)),
            "median": float(np.median(prob_nonevents)),
            "std": float(np.std(prob_nonevents)),
            "min": float(np.min(prob_nonevents)),
            "max": float(np.max(prob_nonevents)),
            "q25": float(np.percentile(prob_nonevents, 25)),
            "q75": float(np.percentile(prob_nonevents, 75)),
            "iqr": float(np.percentile(prob_nonevents, 75) - np.percentile(prob_nonevents, 25)),
            # Histogram data for plotting
            "histogram": _compute_histogram(prob_nonevents),
        }
    else:
        result["non_events"] = {"n": 0, "error": "No non-events in sample"}

    # Discrimination slope (coefficient of discrimination)
    # Per Van Calster: difference between mean predicted for events vs non-events
    if len(prob_events) > 0 and len(prob_nonevents) > 0:
        result["discrimination_slope"] = float(np.mean(prob_events) - np.mean(prob_nonevents))

        # Kolmogorov-Smirnov test for distribution difference
        try:
            ks_stat, ks_pval = stats.ks_2samp(prob_events, prob_nonevents)
            result["ks_test"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_pval),
            }
        except Exception:
            pass

    return result


def _compute_histogram(
    values: np.ndarray,
    bins: int = 20,
) -> dict[str, Any]:
    """Compute histogram data for plotting."""
    counts, edges = np.histogram(values, bins=bins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2

    return {
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
    }


def _bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
) -> list[float]:
    """Bootstrap a metric function.

    Note: This is a thin wrapper around faircareai.core.bootstrap.bootstrap_metric
    for backward compatibility.
    """
    from faircareai.core.bootstrap import bootstrap_metric

    samples, _ = bootstrap_metric(
        y_true,
        y_prob,
        metric_fn,
        n_bootstrap=n_bootstrap,
        seed=DEFAULT_BOOTSTRAP_SEED,
        min_classes=2,
    )
    return samples


def _compute_vancalster_disparities(
    subgroup_results: dict[str, Any],
    reference: str,
    threshold: float,
) -> dict[str, Any]:
    """Compute disparities between subgroups and reference.

    Args:
        subgroup_results: Results by subgroup.
        reference: Reference group name.
        threshold: Decision threshold used.

    Returns:
        Dict with disparities for each non-reference group.
    """
    disparities: dict[str, Any] = {"reference": reference, "comparisons": {}}

    ref_data = subgroup_results.get(reference, {})
    if "error" in ref_data:
        return {"error": f"Reference group '{reference}' has insufficient data"}

    # Reference metrics
    ref_auroc = ref_data.get("discrimination", {}).get("auroc")
    ref_brier = ref_data.get("calibration", {}).get("brier_score")
    ref_nb = ref_data.get("clinical_utility", {}).get("net_benefit")
    ref_oe = ref_data.get("calibration", {}).get("oe_ratio")
    ref_disc_slope = ref_data.get("risk_distribution", {}).get("discrimination_slope")

    for group_name, group_data in subgroup_results.items():
        if str(group_name) == str(reference):
            continue
        if "error" in group_data:
            continue

        group_disp: dict[str, Any] = {"group": group_name, "n": group_data.get("n")}

        # AUROC difference
        group_auroc = group_data.get("discrimination", {}).get("auroc")
        if group_auroc is not None and ref_auroc is not None:
            auroc_diff = group_auroc - ref_auroc
            group_disp["auroc_diff"] = float(auroc_diff)
            group_disp["auroc_group"] = float(group_auroc)
            group_disp["auroc_reference"] = float(ref_auroc)
            group_disp["auroc_clinically_meaningful"] = (
                abs(auroc_diff) >= AUROC_DIFF_CLINICALLY_MEANINGFUL
            )

        # Brier score difference (lower is better, so negative diff = group is better)
        group_brier = group_data.get("calibration", {}).get("brier_score")
        if group_brier is not None and ref_brier is not None:
            group_disp["brier_diff"] = float(group_brier - ref_brier)

        # O:E ratio difference
        group_oe = group_data.get("calibration", {}).get("oe_ratio")
        if group_oe is not None and ref_oe is not None:
            group_disp["oe_ratio_diff"] = float(group_oe - ref_oe)
            group_disp["oe_ratio_group"] = float(group_oe)
            group_disp["oe_ratio_reference"] = float(ref_oe)

        # Net benefit difference
        group_nb = group_data.get("clinical_utility", {}).get("net_benefit")
        if group_nb is not None and ref_nb is not None:
            group_disp["net_benefit_diff"] = float(group_nb - ref_nb)

        # Discrimination slope difference
        group_disc = group_data.get("risk_distribution", {}).get("discrimination_slope")
        if group_disc is not None and ref_disc_slope is not None:
            group_disp["discrimination_slope_diff"] = float(group_disc - ref_disc_slope)

        disparities["comparisons"][str(group_name)] = group_disp

    return disparities


def _interpret_auroc(auroc: float) -> str:
    """Interpret AUROC value."""
    if auroc >= 0.9:
        return "Excellent discrimination"
    elif auroc >= 0.8:
        return "Good discrimination"
    elif auroc >= 0.7:
        return "Acceptable discrimination"
    elif auroc >= 0.6:
        return "Poor discrimination"
    else:
        return "Failed discrimination (no better than chance)"


def _interpret_calibration(cal_metrics: dict) -> str:
    """Interpret calibration metrics."""
    issues = []

    slope = cal_metrics.get("calibration_slope")
    intercept = cal_metrics.get("calibration_intercept")
    brier = cal_metrics.get("brier_score")
    oe = cal_metrics.get("oe_ratio")

    if slope is not None:
        if slope < 0.8:
            issues.append(f"overfitting (slope={slope:.2f})")
        elif slope > 1.2:
            issues.append(f"underfitting (slope={slope:.2f})")

    if intercept is not None and abs(intercept) > 0.5:
        direction = "underestimation" if intercept > 0 else "overestimation"
        issues.append(f"systematic {direction} (intercept={intercept:.2f})")

    if oe is not None:
        if oe < 0.8:
            issues.append(f"model overpredicts events (O:E={oe:.2f})")
        elif oe > 1.2:
            issues.append(f"model underpredicts events (O:E={oe:.2f})")

    if brier is not None and brier > 0.25:
        issues.append(f"poor overall calibration (Brier={brier:.3f})")

    if not issues:
        return "Good calibration"
    return "Calibration issues: " + "; ".join(issues)


def _interpret_net_benefit(nb_result: dict, threshold: float) -> str:
    """Interpret net benefit results."""
    snb = nb_result.get("standardized_net_benefit")
    useful_range = nb_result.get("useful_range", {})

    interpretations = []

    if snb is not None:
        if snb >= 0.9:
            interpretations.append("Excellent clinical utility")
        elif snb >= 0.7:
            interpretations.append("Good clinical utility")
        elif snb >= 0.5:
            interpretations.append("Moderate clinical utility")
        else:
            interpretations.append("Limited clinical utility")

    if useful_range.get("min") and useful_range.get("max"):
        min_t = useful_range["min"]
        max_t = useful_range["max"]
        if threshold < min_t or threshold > max_t:
            interpretations.append(
                f"Selected threshold ({threshold:.0%}) outside useful range "
                f"({min_t:.0%}-{max_t:.0%})"
            )

    return "; ".join(interpretations) if interpretations else "Clinical utility assessment complete"


def _interpret_vancalster_results(results: dict) -> dict[str, Any]:
    """Compute summary of Van Calster metric disparities.

    Presents findings objectively per Van Calster et al. (2025) methodology.
    Interpretation rests with your organization's governance process.

    Args:
        results: Full results dictionary.

    Returns:
        Dict with computed findings and disparity summary.
    """
    summary: dict[str, Any] = {
        "methodology": "Van Calster et al. (2025) Lancet Digit Health",
        "summary": [],
        "findings": [],
    }

    subgroups = results.get("by_subgroup", {})
    disparities = results.get("disparities", {}).get("comparisons", {})

    # Identify AUROC disparities
    auroc_diffs = [
        (g, d.get("auroc_diff", 0))
        for g, d in disparities.items()
        if d.get("auroc_diff") is not None
    ]
    if auroc_diffs:
        max_auroc_diff = max(auroc_diffs, key=lambda x: abs(x[1]))
        if abs(max_auroc_diff[1]) >= AUROC_DIFF_CLINICALLY_MEANINGFUL:
            summary["findings"].append(
                {
                    "metric": "AUROC",
                    "group": max_auroc_diff[0],
                    "difference": max_auroc_diff[1],
                    "threshold": AUROC_DIFF_CLINICALLY_MEANINGFUL,
                    "description": f"AUROC difference of {max_auroc_diff[1]:+.3f} in '{max_auroc_diff[0]}'",
                }
            )

    # Identify calibration differences
    oe_diffs = [
        (g, d.get("oe_ratio_group", 1.0))
        for g, d in disparities.items()
        if d.get("oe_ratio_group") is not None
    ]
    miscalibrated_groups = [(g, oe) for g, oe in oe_diffs if abs(oe - 1.0) > 0.2]
    for g, oe in miscalibrated_groups:
        summary["findings"].append(
            {
                "metric": "O/E Ratio",
                "group": g,
                "value": oe,
                "threshold": 0.2,
                "description": f"O/E ratio of {oe:.2f} in '{g}' (>0.2 from 1.0)",
            }
        )

    # Identify net benefit differences
    nb_diffs = [
        (g, d.get("net_benefit_diff", 0))
        for g, d in disparities.items()
        if d.get("net_benefit_diff") is not None
    ]
    if nb_diffs:
        max_nb_diff = max(nb_diffs, key=lambda x: abs(x[1]))
        if abs(max_nb_diff[1]) > 0.05:
            summary["findings"].append(
                {
                    "metric": "Net Benefit",
                    "group": max_nb_diff[0],
                    "difference": max_nb_diff[1],
                    "threshold": 0.05,
                    "description": f"Net benefit difference of {max_nb_diff[1]:+.3f} in '{max_nb_diff[0]}'",
                }
            )

    # Generate summary text
    n_subgroups = len(subgroups)
    n_findings = len(summary["findings"])

    summary["summary"].append(
        f"Analyzed {n_subgroups} subgroups. {n_findings} metric(s) outside configured thresholds."
    )

    # Count-based summary for display
    summary["subgroup_count"] = n_subgroups
    summary["finding_count"] = n_findings

    # Backward compatibility aliases
    summary["concerns"] = [f["description"] for f in summary["findings"]]
    summary["recommendations"] = []  # No longer providing recommendations

    return summary


# =============================================================================
# CONVENIENCE FUNCTIONS FOR INDIVIDUAL METRICS
# =============================================================================


def compute_auroc_by_subgroup(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_col: str,
    reference: str | None = None,
    bootstrap_ci: bool = True,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> dict[str, Any]:
    """Compute AUROC for each subgroup with bootstrap CIs.

    Per Van Calster et al. (2025): "AUROC is the key measure for
    comparing discrimination across demographic groups."

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        group_col: Column name for subgroup variable.
        reference: Reference group for comparisons.
        bootstrap_ci: Whether to compute 95% CIs.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with per-subgroup AUROC and comparisons.
    """
    results: dict[str, Any] = {
        "metric": "auroc",
        "citation": "Van Calster et al. (2025) Lancet Digit Health",
        "groups": {},
    }

    groups = df[group_col].drop_nulls().unique().sort().to_list()

    # Determine reference
    if reference is None:
        group_counts = df.group_by(group_col).len().sort("len", descending=True)
        reference = group_counts[group_col][0]

    results["reference"] = reference

    for group in groups:
        group_df = df.filter(pl.col(group_col) == group)
        y_true = group_df[y_true_col].to_numpy()
        y_prob = group_df[y_prob_col].to_numpy()

        metrics = _compute_auroc_metrics(y_true, y_prob, bootstrap_ci, n_bootstrap)
        metrics["n"] = len(y_true)
        metrics["is_reference"] = str(group) == str(reference)

        results["groups"][str(group)] = metrics

    # Compute disparities
    results["disparities"] = {}
    ref_auroc = results["groups"].get(str(reference), {}).get("auroc")

    if ref_auroc is not None:
        for group_name, group_data in results["groups"].items():
            if str(group_name) == str(reference):
                continue

            group_auroc = group_data.get("auroc")
            if group_auroc is not None:
                diff = group_auroc - ref_auroc
                results["disparities"][str(group_name)] = {
                    "auroc_diff": float(diff),
                    "clinically_meaningful": abs(diff) >= AUROC_DIFF_CLINICALLY_MEANINGFUL,
                }

    return results


def compute_calibration_by_subgroup(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_col: str,
    n_bins: int = CALIBRATION_BINS_DEFAULT,
) -> dict[str, Any]:
    """Compute calibration curves for each subgroup.

    Per Van Calster et al. (2025): "Calibration plot is the most insightful
    approach to assess calibration, particularly when smoothing is used."

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        group_col: Column name for subgroup variable.
        n_bins: Number of bins for calibration curves.

    Returns:
        Dict with per-subgroup calibration metrics and curve data.
    """
    results: dict[str, Any] = {
        "metric": "calibration",
        "citation": "Van Calster et al. (2025) Lancet Digit Health",
        "groups": {},
    }

    groups = df[group_col].drop_nulls().unique().sort().to_list()

    for group in groups:
        group_df = df.filter(pl.col(group_col) == group)
        y_true = group_df[y_true_col].to_numpy()
        y_prob = group_df[y_prob_col].to_numpy()

        metrics = _compute_calibration_metrics(y_true, y_prob, n_bins)
        metrics["n"] = len(y_true)

        results["groups"][str(group)] = metrics

    return results


def compute_net_benefit_by_subgroup(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_col: str,
    threshold: float = 0.5,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute net benefit (decision curve analysis) for each subgroup.

    Per Van Calster et al. (2025): "Net benefit with decision curve analysis
    is essential to report. It evaluates whether a model leads to improved
    clinical decisions on average."

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        group_col: Column name for subgroup variable.
        threshold: Primary decision threshold.
        thresholds: Array of thresholds for decision curves.

    Returns:
        Dict with per-subgroup net benefit and decision curves.
    """
    if thresholds is None:
        thresholds = NET_BENEFIT_THRESHOLDS_DEFAULT

    results: dict[str, Any] = {
        "metric": "net_benefit",
        "citation": "Van Calster et al. (2025) Lancet Digit Health",
        "primary_threshold": threshold,
        "groups": {},
    }

    groups = df[group_col].drop_nulls().unique().sort().to_list()

    for group in groups:
        group_df = df.filter(pl.col(group_col) == group)
        y_true = group_df[y_true_col].to_numpy()
        y_prob = group_df[y_prob_col].to_numpy()

        metrics = _compute_net_benefit_metrics(y_true, y_prob, threshold, thresholds)
        metrics["n"] = len(y_true)
        metrics["prevalence"] = float(np.mean(y_true))

        results["groups"][str(group)] = metrics

    return results


def compute_risk_distribution_by_subgroup(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_col: str,
) -> dict[str, Any]:
    """Compute risk distribution statistics for each subgroup.

    Per Van Calster et al. (2025): "A plot showing probability distributions
    for each outcome category provides valuable insights into a model's behavior."

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        group_col: Column name for subgroup variable.

    Returns:
        Dict with per-subgroup risk distribution statistics.
    """
    results: dict[str, Any] = {
        "metric": "risk_distribution",
        "citation": "Van Calster et al. (2025) Lancet Digit Health",
        "groups": {},
    }

    groups = df[group_col].drop_nulls().unique().sort().to_list()

    for group in groups:
        group_df = df.filter(pl.col(group_col) == group)
        y_true = group_df[y_true_col].to_numpy()
        y_prob = group_df[y_prob_col].to_numpy()

        metrics = _compute_risk_distribution(y_true, y_prob)
        metrics["n"] = len(y_true)

        results["groups"][str(group)] = metrics

    return results
