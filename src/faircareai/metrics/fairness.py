"""
FairCareAI Fairness Metrics Module

Compute fairness metrics for binary classifiers across sensitive groups.
Includes demographic parity, equalized odds, and calibration-based metrics.

Methodology: CHAI RAIC AC1.CR92 (bias testing), Van Calster et al. (2025).
Note: Impossibility theorem applies - cannot satisfy all metrics simultaneously.
"""

from typing import Any

import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score

from faircareai.core.constants import (
    AUROC_DIFF_MODERATE,
    AUROC_DIFF_NEGLIGIBLE,
    AUROC_DIFF_SMALL,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP_SUBGROUP,
    DEMOGRAPHIC_PARITY_LOWER,
    DEMOGRAPHIC_PARITY_UPPER,
    DISPARITY_INDEX_HIGH,
    DISPARITY_INDEX_LOW,
    DISPARITY_INDEX_MODERATE,
    DISPARITY_WEIGHT_DEMOGRAPHIC_PARITY,
    DISPARITY_WEIGHT_EQUAL_OPPORTUNITY,
    DISPARITY_WEIGHT_EQUALIZED_ODDS,
    DISPARITY_WEIGHT_PREDICTIVE_PARITY,
    EQUALIZED_ODDS_THRESHOLD,
    MIN_BOOTSTRAP_SAMPLES,
    MIN_SAMPLE_SIZE_CALIBRATION,
    MIN_SAMPLE_SIZE_FLAG,
)
from faircareai.core.logging import get_logger

logger = get_logger(__name__)


def compute_fairness_metrics(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_col: str,
    threshold: float = 0.5,
    reference: str | None = None,
) -> dict[str, Any]:
    """Compute comprehensive fairness metrics for a sensitive attribute.

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        group_col: Column name for sensitive attribute.
        threshold: Decision threshold for classification.
        reference: Reference group for ratio calculations.

    Returns:
        Dict containing:
        - demographic_parity_ratio: Selection rate ratios
        - tpr_diff: True positive rate differences (equal opportunity)
        - fpr_diff: False positive rate differences
        - equalized_odds_diff: Max of TPR/FPR differences
        - ppv_ratio: Predictive parity ratios
        - calibration_diff: Calibration differences
        - group_metrics: Per-group raw metrics
    """
    results: dict[str, Any] = {
        "group_col": group_col,
        "threshold": threshold,
        "group_metrics": {},
    }

    groups = df[group_col].drop_nulls().unique().sort().to_list()

    # Determine reference group
    if reference is None:
        # Use largest group
        group_counts = df.group_by(group_col).len().sort("len", descending=True)
        reference = group_counts[group_col][0]

    results["reference"] = str(reference)

    # Compute per-group metrics
    for group in groups:
        group_df = df.filter(pl.col(group_col) == group)
        y_true = group_df[y_true_col].to_numpy()
        y_prob = group_df[y_prob_col].to_numpy()
        y_pred = (y_prob >= threshold).astype(int)

        n = len(y_true)
        if n < MIN_SAMPLE_SIZE_FLAG:
            results["group_metrics"][str(group)] = {
                "n": n,
                "error": f"Insufficient sample size (n < {MIN_SAMPLE_SIZE_FLAG})",
            }
            continue

        # Confusion matrix components
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except ValueError as e:
            logger.debug("Confusion matrix computation failed for group %s: %s", group, str(e))
            results["group_metrics"][str(group)] = {
                "n": n,
                "error": "Could not compute confusion matrix",
            }
            continue

        # Basic rates
        selection_rate = (tp + fp) / n if n > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        prevalence = (tp + fn) / n if n > 0 else 0.0

        # Mean predicted probability
        mean_prob = float(np.mean(y_prob))

        # Calibration (difference between mean predicted and observed rate)
        observed_rate = (tp + fn) / n if n > 0 else 0.0
        predicted_rate = mean_prob
        mean_calibration_error = predicted_rate - observed_rate

        results["group_metrics"][str(group)] = {
            "n": int(n),
            "prevalence": float(prevalence),
            "selection_rate": float(selection_rate),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "ppv": float(ppv),
            "npv": float(npv),
            "mean_predicted_prob": mean_prob,
            "mean_calibration_error": float(mean_calibration_error),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "is_reference": str(group) == str(reference),
        }

    # Get reference group metrics
    ref_metrics = results["group_metrics"].get(str(reference), {})
    if "error" in ref_metrics:
        results["error"] = f"Reference group '{reference}' has insufficient data"
        return results

    # Compute disparity metrics
    results["demographic_parity_ratio"] = {}
    results["demographic_parity_diff"] = {}
    results["tpr_diff"] = {}
    results["fpr_diff"] = {}
    results["equalized_odds_diff"] = {}
    results["ppv_ratio"] = {}
    results["calibration_diff"] = {}

    ref_selection = ref_metrics.get("selection_rate", 0)
    ref_tpr = ref_metrics.get("tpr", 0)
    ref_fpr = ref_metrics.get("fpr", 0)
    ref_ppv = ref_metrics.get("ppv", 0)
    ref_cal = ref_metrics.get("mean_calibration_error", 0)

    for group in groups:
        if str(group) == str(reference):
            continue

        group_data = results["group_metrics"].get(str(group), {})
        if "error" in group_data:
            continue

        # Demographic parity
        selection = group_data.get("selection_rate", 0)
        if ref_selection > 0:
            results["demographic_parity_ratio"][str(group)] = float(selection / ref_selection)
        else:
            results["demographic_parity_ratio"][str(group)] = None
        results["demographic_parity_diff"][str(group)] = float(selection - ref_selection)

        # Equal opportunity (TPR difference)
        tpr = group_data.get("tpr", 0)
        results["tpr_diff"][str(group)] = float(tpr - ref_tpr)

        # FPR difference
        fpr = group_data.get("fpr", 0)
        results["fpr_diff"][str(group)] = float(fpr - ref_fpr)

        # Equalized odds (max of TPR and FPR differences)
        results["equalized_odds_diff"][str(group)] = float(
            max(abs(tpr - ref_tpr), abs(fpr - ref_fpr))
        )

        # Predictive parity (PPV ratio)
        ppv = group_data.get("ppv", 0)
        if ref_ppv > 0:
            results["ppv_ratio"][str(group)] = float(ppv / ref_ppv)
        else:
            results["ppv_ratio"][str(group)] = None

        # Calibration difference
        cal = group_data.get("mean_calibration_error", 0)
        results["calibration_diff"][str(group)] = float(cal - ref_cal)

    # Summary statistics
    results["summary"] = _compute_fairness_summary(results)

    return results


def _compute_fairness_summary(metrics: dict) -> dict[str, Any]:
    """Compute summary statistics for fairness metrics.

    Args:
        metrics: Dict from compute_fairness_metrics.

    Returns:
        Summary dict with worst disparities and pass/fail flags.
    """
    summary: dict[str, Any] = {}

    # Demographic parity
    dp_diffs = list(metrics.get("demographic_parity_diff", {}).values())
    if dp_diffs:
        worst_dp = max(dp_diffs, key=abs)
        summary["demographic_parity"] = {
            "worst_diff": float(worst_dp),
            "within_threshold": abs(worst_dp) <= EQUALIZED_ODDS_THRESHOLD,
        }

    # Equal opportunity
    tpr_diffs = list(metrics.get("tpr_diff", {}).values())
    if tpr_diffs:
        worst_tpr = max(tpr_diffs, key=abs)
        summary["equal_opportunity"] = {
            "worst_diff": float(worst_tpr),
            "within_threshold": abs(worst_tpr) <= EQUALIZED_ODDS_THRESHOLD,
        }

    # Equalized odds
    eo_diffs = list(metrics.get("equalized_odds_diff", {}).values())
    if eo_diffs:
        worst_eo = max(eo_diffs)
        summary["equalized_odds"] = {
            "worst_diff": float(worst_eo),
            "within_threshold": worst_eo <= EQUALIZED_ODDS_THRESHOLD,
        }

    # Predictive parity - filter None values (occur when reference PPV is 0)
    ppv_ratios_raw = list(metrics.get("ppv_ratio", {}).values())
    ppv_ratios = [r for r in ppv_ratios_raw if r is not None]
    if ppv_ratios:
        min_ppv = min(ppv_ratios)
        worst_ppv = min_ppv if min_ppv < 1 else max(ppv_ratios)
        summary["predictive_parity"] = {
            "worst_ratio": float(worst_ppv),
            "within_threshold": DEMOGRAPHIC_PARITY_LOWER <= worst_ppv <= DEMOGRAPHIC_PARITY_UPPER,
        }

    return summary


def compute_disparity_index(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_col: str,
    threshold: float = 0.5,
    reference: str | None = None,
) -> dict[str, Any]:
    """Compute aggregate disparity index across metrics.

    The disparity index combines multiple fairness metrics into
    a single score for governance reporting.

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        group_col: Column name for sensitive attribute.
        threshold: Decision threshold.
        reference: Reference group.

    Returns:
        Dict with disparity index and component scores.
    """
    metrics = compute_fairness_metrics(df, y_prob_col, y_true_col, group_col, threshold, reference)

    if "error" in metrics:
        return {"error": metrics["error"]}

    # Component scores (0-1 scale, higher = more disparity)
    components = {}

    # Demographic parity component
    dp_diffs = list(metrics.get("demographic_parity_diff", {}).values())
    if dp_diffs:
        components["demographic_parity"] = min(1.0, max(abs(d) for d in dp_diffs) / 0.2)

    # Equal opportunity component
    tpr_diffs = list(metrics.get("tpr_diff", {}).values())
    if tpr_diffs:
        components["equal_opportunity"] = min(1.0, max(abs(d) for d in tpr_diffs) / 0.2)

    # Equalized odds component
    eo_diffs = list(metrics.get("equalized_odds_diff", {}).values())
    if eo_diffs:
        components["equalized_odds"] = min(1.0, max(eo_diffs) / 0.2)

    # Predictive parity component - filter None values (occur when reference PPV is 0)
    ppv_ratios_raw = list(metrics.get("ppv_ratio", {}).values())
    ppv_ratios = [r for r in ppv_ratios_raw if r is not None]
    if ppv_ratios:
        # Convert ratio to disparity score
        min_ppv_ratio = min(ppv_ratios)
        worst_ratio = min_ppv_ratio if min_ppv_ratio < 1 else max(ppv_ratios)
        ratio_deviation = abs(worst_ratio - 1.0)
        components["predictive_parity"] = min(1.0, ratio_deviation / 0.4)

    # Aggregate index (weighted average)
    if components:
        weights = {
            "demographic_parity": DISPARITY_WEIGHT_DEMOGRAPHIC_PARITY,
            "equal_opportunity": DISPARITY_WEIGHT_EQUAL_OPPORTUNITY,
            "equalized_odds": DISPARITY_WEIGHT_EQUALIZED_ODDS,
            "predictive_parity": DISPARITY_WEIGHT_PREDICTIVE_PARITY,
        }
        total_weight = sum(weights.get(k, 0) for k in components)
        disparity_index = (
            sum(components[k] * weights.get(k, 0.25) for k in components) / total_weight
            if total_weight > 0
            else 0.0
        )
    else:
        disparity_index = 0.0

    return {
        "disparity_index": float(disparity_index),
        "components": components,
        "interpretation": _interpret_disparity_index(disparity_index),
        "raw_metrics": metrics,
    }


def _interpret_disparity_index(index: float) -> dict[str, Any]:
    """Interpret aggregate disparity index."""
    if index < DISPARITY_INDEX_LOW:
        return {
            "level": "LOW",
            "color": "green",
            "description": "Minimal disparities detected across groups.",
        }
    elif index < DISPARITY_INDEX_MODERATE:
        return {
            "level": "MODERATE",
            "color": "yellow",
            "description": "Some disparities detected. Review specific metrics.",
        }
    elif index < DISPARITY_INDEX_HIGH:
        return {
            "level": "HIGH",
            "color": "orange",
            "description": "Significant disparities detected. Mitigation recommended.",
        }
    else:
        return {
            "level": "SEVERE",
            "color": "red",
            "description": "Severe disparities detected. Address before deployment.",
        }


def compute_calibration_by_group(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_col: str,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration curves for each group.

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        group_col: Column name for sensitive attribute.
        n_bins: Number of bins for calibration curve.

    Returns:
        Dict with per-group calibration data.
    """
    results: dict[str, Any] = {"groups": {}}

    groups = df[group_col].drop_nulls().unique().sort().to_list()

    for group in groups:
        group_df = df.filter(pl.col(group_col) == group)
        y_true = group_df[y_true_col].to_numpy()
        y_prob = group_df[y_prob_col].to_numpy()

        if len(y_true) < MIN_SAMPLE_SIZE_CALIBRATION:
            results["groups"][str(group)] = {
                "n": len(y_true),
                "error": f"Insufficient sample size for calibration (n < {MIN_SAMPLE_SIZE_CALIBRATION})",
            }
            continue

        try:
            prob_true, prob_pred = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy="uniform"
            )

            # Calibration error
            ece = float(np.mean(np.abs(prob_pred - prob_true)))
            mce = float(np.max(np.abs(prob_pred - prob_true)))

            results["groups"][str(group)] = {
                "n": len(y_true),
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
                "ece": ece,  # Expected Calibration Error
                "mce": mce,  # Maximum Calibration Error
            }
        except ValueError as e:
            logger.debug("Calibration curve failed for group %s: %s", group, str(e))
            results["groups"][str(group)] = {
                "n": len(y_true),
                "error": str(e),
            }

    return results


def compute_threshold_fairness(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_col: str,
    thresholds: list[float] | None = None,
    reference: str | None = None,
) -> dict[str, Any]:
    """Analyze how fairness metrics change across thresholds.

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        group_col: Column name for sensitive attribute.
        thresholds: List of thresholds to evaluate.
        reference: Reference group.

    Returns:
        Dict with fairness metrics at each threshold.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results: dict[str, Any] = {
        "thresholds": thresholds,
        "metrics_by_threshold": [],
    }

    for thresh in thresholds:
        metrics = compute_fairness_metrics(df, y_prob_col, y_true_col, group_col, thresh, reference)
        summary = metrics.get("summary", {})

        results["metrics_by_threshold"].append(
            {
                "threshold": thresh,
                "demographic_parity_worst": summary.get("demographic_parity", {}).get("worst_diff"),
                "equal_opportunity_worst": summary.get("equal_opportunity", {}).get("worst_diff"),
                "equalized_odds_worst": summary.get("equalized_odds", {}).get("worst_diff"),
                "predictive_parity_worst": summary.get("predictive_parity", {}).get("worst_ratio"),
            }
        )

    # Find threshold with best fairness
    best_threshold = None
    best_eo = float("inf")

    for item in results["metrics_by_threshold"]:
        eo = item.get("equalized_odds_worst")
        if eo is not None and eo < best_eo:
            best_eo = eo
            best_threshold = item["threshold"]

    results["recommended_threshold"] = {
        "threshold": best_threshold,
        "equalized_odds_diff": best_eo if best_eo != float("inf") else None,
        "note": "Threshold that minimizes equalized odds disparity",
    }

    return results


def compute_group_auroc_comparison(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_col: str,
    reference: str | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP_SUBGROUP,
) -> dict[str, Any]:
    """Compare AUROC across groups with statistical testing.

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        group_col: Column name for sensitive attribute.
        reference: Reference group.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with per-group AUROC and pairwise comparisons.
    """
    results: dict[str, Any] = {"groups": {}, "comparisons": {}}

    groups = df[group_col].drop_nulls().unique().sort().to_list()

    # Compute per-group AUROC
    for group in groups:
        group_df = df.filter(pl.col(group_col) == group)
        y_true = group_df[y_true_col].to_numpy()
        y_prob = group_df[y_prob_col].to_numpy()

        if len(y_true) < MIN_SAMPLE_SIZE_CALIBRATION or len(np.unique(y_true)) < 2:
            results["groups"][str(group)] = {
                "n": len(y_true),
                "error": f"Insufficient data for AUROC (n < {MIN_SAMPLE_SIZE_CALIBRATION} or single class)",
            }
            continue

        auroc = roc_auc_score(y_true, y_prob)

        # Bootstrap CI
        auroc_samples = []
        rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
        n = len(y_true)

        for i in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            y_true_boot = y_true[idx]
            y_prob_boot = y_prob[idx]

            if len(np.unique(y_true_boot)) < 2:
                continue

            try:
                auroc_samples.append(roc_auc_score(y_true_boot, y_prob_boot))
            except ValueError as e:
                logger.debug(
                    "Bootstrap AUROC iteration %d failed for group %s: %s", i, group, str(e)
                )
                continue

        if len(auroc_samples) > MIN_BOOTSTRAP_SAMPLES:
            auroc_ci = np.percentile(auroc_samples, [2.5, 97.5])
        else:
            logger.debug(
                "Insufficient bootstrap samples for group %s: %d < %d",
                group,
                len(auroc_samples),
                MIN_BOOTSTRAP_SAMPLES,
            )
            auroc_ci = [None, None]

        results["groups"][str(group)] = {
            "n": len(y_true),
            "auroc": float(auroc),
            "auroc_ci_95": [
                float(auroc_ci[0]) if auroc_ci[0] is not None else None,
                float(auroc_ci[1]) if auroc_ci[1] is not None else None,
            ],
        }

    # Determine reference
    if reference is None:
        group_counts = df.group_by(group_col).len().sort("len", descending=True)
        reference = group_counts[group_col][0]

    results["reference"] = str(reference)

    # Pairwise comparisons vs reference
    ref_data = results["groups"].get(str(reference), {})
    if "error" not in ref_data:
        ref_auroc = ref_data.get("auroc", 0)

        for group in groups:
            if str(group) == str(reference):
                continue

            group_data = results["groups"].get(str(group), {})
            if "error" in group_data:
                continue

            group_auroc = group_data.get("auroc", 0)
            diff = group_auroc - ref_auroc

            results["comparisons"][str(group)] = {
                "auroc_diff": float(diff),
                "group_auroc": float(group_auroc),
                "reference_auroc": float(ref_auroc),
                "interpretation": _interpret_auroc_diff(diff),
            }

    return results


def _interpret_auroc_diff(diff: float) -> str:
    """Interpret AUROC difference."""
    abs_diff = abs(diff)
    if abs_diff < AUROC_DIFF_NEGLIGIBLE:
        return "negligible"
    elif abs_diff < AUROC_DIFF_SMALL:
        return "small"
    elif abs_diff < AUROC_DIFF_MODERATE:
        return "moderate"
    else:
        return "large"
