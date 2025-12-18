"""
FairCareAI Performance Metrics Module

Compute overall model performance metrics per TRIPOD+AI standards.
Includes discrimination, calibration, and clinical utility metrics.

Methodology: Van Calster et al. (2025), TRIPOD+AI (Collins et al. 2024).

Van Calster et al. (2025) Metric Classification:
-------------------------------------------------
RECOMMENDED (essential for governance reports):
  - AUROC: Key discrimination measure
  - Calibration plot: Smoothed calibration curve
  - Net Benefit: Clinical utility via decision curve analysis
  - Risk Distribution: Probability distributions by outcome

OPTIONAL (acceptable for data science teams):
  - Brier score, Scaled Brier Score (BSS)
  - O:E ratio, Calibration slope, Calibration intercept
  - ICI, ECI (Integrated/E-statistic Calibration Index)
  - Sensitivity + Specificity (together), PPV + NPV (together)

USE_WITH_CAUTION (improper measures - use with explicit caveats only):
  - F1 score: ONLY metric violating BOTH properness AND clear focus
  - Accuracy, Balanced Accuracy, Youden Index
  - MCC (Matthews Correlation Coefficient)
  - DOR (Diagnostic Odds Ratio), Kappa
  - AUPRC, partial AUROC (mix statistical with decision-analytical)

See constants.py for VANCALSTER_* classification constants.
"""

from typing import Any

import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from statsmodels.api import Logit

from faircareai.core.constants import (
    BRIER_POOR_THRESHOLD,
    CALIBRATION_SLOPE_OVERFITTING,
    CALIBRATION_SLOPE_UNDERFITTING,
    PROB_CLIP_MAX,
    PROB_CLIP_MIN,
)
from faircareai.core.logging import get_logger

logger = get_logger(__name__)


def compute_overall_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    thresholds_to_evaluate: list[float] | None = None,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Compute comprehensive model performance metrics.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        threshold: Primary decision threshold.
        thresholds_to_evaluate: List of thresholds for sensitivity analysis.
        bootstrap_ci: Whether to compute bootstrap confidence intervals.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict containing:
        - discrimination: AUROC, AUPRC with 95% CI, curve data
        - calibration: Brier, slope, intercept, E/O ratio, ICI
        - classification_at_threshold: Sens, Spec, PPV, NPV, F1, NNE
        - threshold_analysis: metrics across multiple cutoffs
        - decision_curve: DCA net benefit data
        - confusion_matrix: TP, FP, TN, FN
    """
    results: dict[str, Any] = {
        "primary_threshold": threshold,
    }

    # Ensure numpy arrays
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # === Discrimination Metrics ===
    results["discrimination"] = compute_discrimination_metrics(
        y_true, y_prob, bootstrap_ci, n_bootstrap
    )

    # === Calibration Metrics ===
    results["calibration"] = compute_calibration_metrics(y_true, y_prob)

    # === Classification at Threshold ===
    results["classification_at_threshold"] = compute_classification_at_threshold(
        y_true, y_prob, threshold, bootstrap_ci, n_bootstrap
    )

    # === Threshold Analysis ===
    if thresholds_to_evaluate is None:
        thresholds_to_evaluate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results["threshold_analysis"] = compute_threshold_analysis(
        y_true, y_prob, thresholds_to_evaluate
    )

    # === Decision Curve Analysis ===
    results["decision_curve"] = compute_decision_curve_analysis(y_true, y_prob)

    # === Confusion Matrix ===
    results["confusion_matrix"] = compute_confusion_matrix(y_true, y_prob, threshold)

    return results


def compute_discrimination_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Compute discrimination metrics with confidence intervals.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        bootstrap_ci: Whether to compute bootstrap CI.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with AUROC, AUPRC, and curve data.
    """
    # Point estimates
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    # PR curve data
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

    result = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier_score": float(brier),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist() if len(pr_thresholds) > 0 else [],
        },
        "prevalence": float(np.mean(y_true)),
    }

    # Bootstrap confidence intervals
    if bootstrap_ci and len(y_true) > 10:
        auroc_samples = []
        auprc_samples = []

        rng = np.random.default_rng(42)
        n = len(y_true)

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            y_true_boot = y_true[idx]
            y_prob_boot = y_prob[idx]

            # Need at least one positive and one negative
            if len(np.unique(y_true_boot)) < 2:
                continue

            try:
                auroc_samples.append(roc_auc_score(y_true_boot, y_prob_boot))
                auprc_samples.append(average_precision_score(y_true_boot, y_prob_boot))
            except ValueError:
                continue

        if len(auroc_samples) > 10:
            auroc_ci = np.percentile(auroc_samples, [2.5, 97.5])
            auprc_ci = np.percentile(auprc_samples, [2.5, 97.5])

            result["auroc_ci_95"] = [float(auroc_ci[0]), float(auroc_ci[1])]
            result["auprc_ci_95"] = [float(auprc_ci[0]), float(auprc_ci[1])]
            result["auroc_ci_fmt"] = f"(95% CI: {auroc_ci[0]:.3f}-{auroc_ci[1]:.3f})"
            result["auprc_ci_fmt"] = f"(95% CI: {auprc_ci[0]:.3f}-{auprc_ci[1]:.3f})"

    return result


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration metrics.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration curve.

    Returns:
        Dict with Brier score, slope, intercept, E/O ratio, ICI.
    """
    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    # Scaled Brier Score (BSS) per Van Calster et al.
    # BSS = 1 - (Brier / Brier_null), where Brier_null = prevalence * (1 - prevalence)
    prevalence = np.mean(y_true)
    brier_null = prevalence * (1 - prevalence)
    brier_scaled = 1 - (brier / brier_null) if brier_null > 0 else 0.0

    # Calibration curve
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    except ValueError:
        prob_true, prob_pred = np.array([]), np.array([])

    # Calibration intercept and slope per Van Calster et al. methodology
    # Intercept: fit logistic model with logit(p) as OFFSET (coefficient fixed at 1)
    # Slope: fit logistic regression with logit(p) as predictor
    # Slope = 1 means perfect calibration, <1 = overfitting, >1 = underfitting
    slope = 1.0
    intercept = 0.0
    try:
        # Clip probabilities to avoid log(0)
        y_prob_clipped = np.clip(y_prob, PROB_CLIP_MIN, PROB_CLIP_MAX)
        logit_p = np.log(y_prob_clipped / (1 - y_prob_clipped))

        # Calibration INTERCEPT using statsmodels Logit with offset
        # This is the Van Calster method: logit(Y) = α + offset(logit_p)
        try:
            int_model = Logit(y_true, np.ones_like(y_true), offset=logit_p).fit(disp=0)
            intercept = float(int_model.params.iloc[0])
        except Exception:
            # Fallback to sklearn method if statsmodels fails
            lr_int = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr_int.fit(logit_p.reshape(-1, 1), y_true)
            intercept = float(lr_int.intercept_[0])

        # Calibration SLOPE using sklearn LogisticRegression
        lr_slope = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr_slope.fit(logit_p.reshape(-1, 1), y_true)
        slope = float(lr_slope.coef_[0, 0])
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.warning(
            "Calibration slope computation failed: %s. Using default slope=1.0",
            str(e),
        )

    # E/O ratio (Expected/Observed)
    expected = np.sum(y_prob)
    observed = np.sum(y_true)
    eo_ratio = expected / observed if observed > 0 else float("inf")

    # ICI (Integrated Calibration Index) - average absolute calibration error
    # Per Van Calster: mean(|smoothed_observed - predicted|)
    ici = 0.0
    if len(prob_true) > 0:
        ici = float(np.mean(np.abs(prob_pred - prob_true)))

    # ECI (E-statistic Calibration Index) - squared calibration error normalized
    # Per Van Calster: mean((smoothed_observed - predicted)^2) / mean((prevalence - predicted)^2)
    eci = 0.0
    if len(prob_true) > 0:
        eci_numer = np.mean((prob_true - prob_pred) ** 2)
        eci_denom = np.mean((prevalence - prob_pred) ** 2)
        eci = float(eci_numer / eci_denom) if eci_denom > 0 else 0.0

    # E_max (maximum calibration error)
    e_max = 0.0
    if len(prob_true) > 0:
        e_max = float(np.max(np.abs(prob_pred - prob_true)))

    return {
        "brier_score": float(brier),
        "brier_scaled": float(brier_scaled),  # Van Calster BSS
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "eo_ratio": float(eo_ratio),
        "ici": ici,
        "eci": eci,  # Van Calster E-statistic
        "e_max": e_max,
        "calibration_curve": {
            "prob_true": prob_true.tolist() if len(prob_true) > 0 else [],
            "prob_pred": prob_pred.tolist() if len(prob_pred) > 0 else [],
            "n_bins": n_bins,
        },
        "interpretation": _interpret_calibration(slope, brier),
    }


def _interpret_calibration(slope: float, brier: float) -> str:
    """Interpret calibration quality."""
    issues = []

    if slope < CALIBRATION_SLOPE_OVERFITTING:
        issues.append(f"overfitting (slope < {CALIBRATION_SLOPE_OVERFITTING})")
    elif slope > CALIBRATION_SLOPE_UNDERFITTING:
        issues.append(f"underfitting (slope > {CALIBRATION_SLOPE_UNDERFITTING})")

    if brier > BRIER_POOR_THRESHOLD:
        issues.append(f"poor overall calibration (Brier > {BRIER_POOR_THRESHOLD})")

    if not issues:
        return "Good calibration"
    return "Issues: " + ", ".join(issues)


def compute_classification_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Compute classification metrics at a specific threshold.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        threshold: Decision threshold.
        bootstrap_ci: Whether to compute bootstrap CI.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with sensitivity, specificity, PPV, NPV, F1, accuracy,
        balanced_accuracy, youden_index, mcc (Matthews), NNE.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Core metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Additional Van Calster classification metrics
    n_total = tp + tn + fp + fn
    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2
    youden_index = sensitivity + specificity - 1

    # Matthews Correlation Coefficient (MCC)
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    mcc_denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

    # DOR (Diagnostic Odds Ratio) per Van Calster
    # DOR = (Sens / (1 - Spec)) / ((1 - Sens) / Spec)
    if specificity > 0 and specificity < 1 and sensitivity > 0 and sensitivity < 1:
        dor = (sensitivity / (1 - specificity)) / ((1 - sensitivity) / specificity)
    else:
        dor = float("inf") if sensitivity == 1 and specificity == 1 else 0.0

    # Kappa (Cohen's Kappa) per Van Calster
    # Kappa = (Accuracy - Accuracy_expected) / (1 - Accuracy_expected)
    prevalence = np.mean(y_true)
    acc_expected = (
        prevalence * ((tp + fp) / n_total) + (1 - prevalence) * ((fn + tn) / n_total)
        if n_total > 0
        else 0
    )
    kappa = (accuracy - acc_expected) / (1 - acc_expected) if acc_expected < 1 else 0.0

    # Percentage flagged
    pct_flagged = (tp + fp) / len(y_true) * 100 if len(y_true) > 0 else 0.0

    # Number Needed to Evaluate (NNE)
    nne = 1 / ppv if ppv > 0 else float("inf")

    result = {
        "threshold": threshold,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1_score": float(f1),
        # Van Calster additional classification metrics
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "youden_index": float(youden_index),
        "mcc": float(mcc),
        "dor": float(dor) if dor != float("inf") else None,  # Van Calster DOR
        "kappa": float(kappa),  # Van Calster Kappa
        "pct_flagged": float(pct_flagged),
        "nne": float(nne) if nne != float("inf") else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    # Bootstrap CI for key metrics
    if bootstrap_ci and len(y_true) > 10:
        sens_samples = []
        spec_samples = []
        ppv_samples = []

        rng = np.random.default_rng(42)
        n = len(y_true)

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            y_true_boot = y_true[idx]
            y_pred_boot = (y_prob[idx] >= threshold).astype(int)

            try:
                tn_b, fp_b, fn_b, tp_b = confusion_matrix(
                    y_true_boot, y_pred_boot, labels=[0, 1]
                ).ravel()

                sens_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
                spec_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0
                ppv_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0

                sens_samples.append(sens_b)
                spec_samples.append(spec_b)
                ppv_samples.append(ppv_b)
            except ValueError:
                continue

        if len(sens_samples) > 10:
            result["sensitivity_ci_95"] = [
                float(np.percentile(sens_samples, 2.5)),
                float(np.percentile(sens_samples, 97.5)),
            ]
            result["specificity_ci_95"] = [
                float(np.percentile(spec_samples, 2.5)),
                float(np.percentile(spec_samples, 97.5)),
            ]
            result["ppv_ci_95"] = [
                float(np.percentile(ppv_samples, 2.5)),
                float(np.percentile(ppv_samples, 97.5)),
            ]

    return result


def compute_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    """Compute metrics across multiple thresholds.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        thresholds: List of thresholds to evaluate.

    Returns:
        Dict with metrics at each threshold.
    """
    results: dict[str, Any] = {"thresholds": [], "metrics": []}

    for thresh in thresholds:
        metrics = compute_classification_at_threshold(y_true, y_prob, thresh, bootstrap_ci=False)
        results["thresholds"].append(thresh)
        results["metrics"].append(metrics)

    # Also add data for plotting
    results["plot_data"] = {
        "thresholds": thresholds,
        "sensitivity": [m["sensitivity"] for m in results["metrics"]],
        "specificity": [m["specificity"] for m in results["metrics"]],
        "ppv": [m["ppv"] for m in results["metrics"]],
        "npv": [m["npv"] for m in results["metrics"]],
        "f1": [m["f1_score"] for m in results["metrics"]],
        "accuracy": [m["accuracy"] for m in results["metrics"]],
        "balanced_accuracy": [m["balanced_accuracy"] for m in results["metrics"]],
        "youden_index": [m["youden_index"] for m in results["metrics"]],
        "mcc": [m["mcc"] for m in results["metrics"]],
        "pct_flagged": [m["pct_flagged"] for m in results["metrics"]],
    }

    return results


def compute_decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute Decision Curve Analysis for clinical utility.

    DCA compares the net benefit of using the model vs treating all
    or treating none, across a range of threshold probabilities.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        thresholds: Array of threshold probabilities to evaluate.

    Returns:
        Dict with DCA net benefit curves.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    n = len(y_true)
    prevalence = np.mean(y_true)

    # Net benefit for model
    net_benefit_model = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        # Net benefit = TP/n - FP/n * (threshold / (1 - threshold))
        nb = tp / n - fp / n * (t / (1 - t)) if t < 1 else 0
        net_benefit_model.append(nb)

    # Net benefit for treat all
    net_benefit_all = []
    for t in thresholds:
        nb = prevalence - (1 - prevalence) * (t / (1 - t)) if t < 1 else 0
        net_benefit_all.append(nb)

    # Net benefit for treat none is always 0
    net_benefit_none = [0.0] * len(thresholds)

    # Find useful range (where model > treat all and > treat none)
    useful_range = []
    for i, t in enumerate(thresholds):
        if net_benefit_model[i] > max(net_benefit_all[i], 0):
            useful_range.append(t)

    return {
        "thresholds": thresholds.tolist(),
        "net_benefit_model": net_benefit_model,
        "net_benefit_all": net_benefit_all,
        "net_benefit_none": net_benefit_none,
        "useful_range": useful_range,
        "useful_range_summary": {
            "min": float(min(useful_range)) if useful_range else None,
            "max": float(max(useful_range)) if useful_range else None,
        },
        "prevalence": float(prevalence),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute confusion matrix at threshold.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        threshold: Decision threshold.

    Returns:
        Dict with confusion matrix components.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "labels": ["Negative", "Positive"],
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_subgroup_performance(
    df: pl.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    group_col: str,
    threshold: float = 0.5,
    reference: str | None = None,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    """Compute performance metrics for each subgroup.

    Args:
        df: Polars DataFrame with patient data.
        y_true_col: Column name for true labels.
        y_prob_col: Column name for predicted probabilities.
        group_col: Column name for group variable.
        threshold: Decision threshold.
        reference: Reference group for comparisons.
        bootstrap_ci: Whether to compute bootstrap CI.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with per-group performance metrics.
    """
    results: dict[str, Any] = {"groups": {}, "reference": reference}

    groups = df[group_col].drop_nulls().unique().sort().to_list()

    # Determine reference if not specified
    if reference is None:
        # Use largest group
        group_counts = df.group_by(group_col).len().sort("len", descending=True)
        reference = group_counts[group_col][0]
        results["reference"] = reference

    for group in groups:
        group_df = df.filter(pl.col(group_col) == group)
        y_true = group_df[y_true_col].to_numpy()
        y_prob = group_df[y_prob_col].to_numpy()

        if len(y_true) < 10:
            results["groups"][str(group)] = {
                "n": len(y_true),
                "error": "Insufficient sample size (n < 10)",
            }
            continue

        # Discrimination
        try:
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
        except ValueError:
            auroc = None
            auprc = None

        # Classification at threshold
        y_pred = (y_prob >= threshold).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        except ValueError:
            tpr = fpr = ppv = npv = None

        # Calibration
        brier = brier_score_loss(y_true, y_prob) if len(y_true) > 0 else None

        results["groups"][str(group)] = {
            "n": len(y_true),
            "prevalence": float(np.mean(y_true)),
            "auroc": float(auroc) if auroc is not None else None,
            "auprc": float(auprc) if auprc is not None else None,
            "brier_score": float(brier) if brier is not None else None,
            "tpr": float(tpr) if tpr is not None else None,
            "fpr": float(fpr) if fpr is not None else None,
            "ppv": float(ppv) if ppv is not None else None,
            "npv": float(npv) if npv is not None else None,
            "is_reference": str(group) == str(reference),
        }

        # Bootstrap CI for AUROC if requested
        if bootstrap_ci and auroc is not None and len(y_true) >= 20:
            auroc_samples = []
            rng = np.random.default_rng(42)
            n = len(y_true)

            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                y_true_boot = y_true[idx]
                y_prob_boot = y_prob[idx]

                if len(np.unique(y_true_boot)) < 2:
                    continue

                try:
                    auroc_samples.append(roc_auc_score(y_true_boot, y_prob_boot))
                except ValueError:
                    continue

            if len(auroc_samples) > 10:
                auroc_ci = np.percentile(auroc_samples, [2.5, 97.5])
                results["groups"][str(group)]["auroc_ci_95"] = [
                    float(auroc_ci[0]),
                    float(auroc_ci[1]),
                ]

    return results
