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
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from statsmodels.api import Logit

from faircareai.core.bootstrap import (
    bootstrap_confusion_metrics,
    bootstrap_metric,
    compute_percentile_ci,
)
from faircareai.core.constants import (
    BRIER_POOR_THRESHOLD,
    CALIBRATION_SLOPE_OVERFITTING,
    CALIBRATION_SLOPE_UNDERFITTING,
    DEFAULT_BOOTSTRAP_SEED,
    PROB_CLIP_MAX,
    PROB_CLIP_MIN,
)
from faircareai.core.logging import get_logger
from faircareai.core.types import (
    CalibrationMetrics,
    ClassificationMetrics,
    DiscriminationMetrics,
    OverallPerformance,
)
from faircareai.core.validation import safe_divide

logger = get_logger(__name__)


def compute_overall_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    thresholds_to_evaluate: list[float] | None = None,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
    random_seed: int | None = DEFAULT_BOOTSTRAP_SEED,
) -> OverallPerformance:
    """Compute comprehensive model performance metrics.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        threshold: Primary decision threshold.
        thresholds_to_evaluate: List of thresholds for sensitivity analysis.
        bootstrap_ci: Whether to compute bootstrap confidence intervals.
        n_bootstrap: Number of bootstrap iterations.
        random_seed: Random seed for bootstrap resampling.

    Returns:
        Dict containing:
        - discrimination: AUROC, AUPRC with 95% CI, curve data
        - calibration: Brier, slope, intercept, O:E ratio, ICI
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
        y_true, y_prob, bootstrap_ci, n_bootstrap, random_seed
    )

    # === Calibration Metrics ===
    results["calibration"] = compute_calibration_metrics(y_true, y_prob)

    # === Classification at Threshold ===
    results["classification_at_threshold"] = compute_classification_at_threshold(
        y_true, y_prob, threshold, bootstrap_ci, n_bootstrap, random_seed
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
    random_seed: int | None = DEFAULT_BOOTSTRAP_SEED,
) -> DiscriminationMetrics:
    """Compute discrimination metrics with confidence intervals.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        bootstrap_ci: Whether to compute bootstrap CI.
        n_bootstrap: Number of bootstrap iterations.
        random_seed: Random seed for bootstrap resampling.

    Returns:
        Dict with AUROC, AUPRC, AP, and curve data.
    """
    # Point estimates
    auroc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    # PR curve data
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)

    result = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "average_precision": float(ap),
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

    # Bootstrap confidence intervals using centralized bootstrap module
    if bootstrap_ci and len(y_true) > 10:
        seed = DEFAULT_BOOTSTRAP_SEED if random_seed is None else random_seed
        # AUROC bootstrap (stratified=False for backward compatibility)
        auroc_samples, _ = bootstrap_metric(
            y_true,
            y_prob,
            lambda yt, yp: roc_auc_score(yt, yp),
            n_bootstrap=n_bootstrap,
            seed=seed,
            stratified=False,
        )

        # AUPRC bootstrap
        def _auprc_metric(yt: np.ndarray, yp: np.ndarray) -> float:
            prec, rec, _ = precision_recall_curve(yt, yp)
            return float(auc(rec, prec))

        auprc_samples, _ = bootstrap_metric(
            y_true,
            y_prob,
            _auprc_metric,
            n_bootstrap=n_bootstrap,
            seed=seed,
            stratified=False,
        )

        if len(auroc_samples) > 10:
            auroc_ci_lower, auroc_ci_upper = compute_percentile_ci(auroc_samples)
            auprc_ci_lower, auprc_ci_upper = compute_percentile_ci(auprc_samples)

            if auroc_ci_lower is not None:
                result["auroc_ci_95"] = [auroc_ci_lower, auroc_ci_upper]
                result["auroc_ci_fmt"] = f"(95% CI: {auroc_ci_lower:.3f}-{auroc_ci_upper:.3f})"

            if auprc_ci_lower is not None:
                result["auprc_ci_95"] = [auprc_ci_lower, auprc_ci_upper]
                result["auprc_ci_fmt"] = f"(95% CI: {auprc_ci_lower:.3f}-{auprc_ci_upper:.3f})"

    return result


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute calibration metrics.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration curve.

    Returns:
        Dict with Brier score, slope, intercept, O:E ratio, and ICI.
    """
    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    # Scaled Brier Score (BSS) per Van Calster et al.
    # BSS = 1 - (Brier / Brier_null), where Brier_null = prevalence * (1 - prevalence)
    prevalence = np.mean(y_true)
    brier_null = prevalence * (1 - prevalence)
    brier_scaled = 1 - (brier / brier_null) if brier_null > 0 else 0.0

    # Calibration curve (binned)
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    except ValueError:
        prob_true, prob_pred = np.array([]), np.array([])

    # Smoothed calibration curve using LOWESS (Van Calster reference)
    smoothed_pred = np.array([])
    smoothed_true = np.array([])
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        order = np.argsort(y_prob)
        prob_sorted = y_prob[order]
        y_sorted = y_true[order]
        lowess_result = lowess(y_sorted, prob_sorted, frac=0.75, it=0, return_sorted=True)
        smoothed_pred = lowess_result[:, 0]
        smoothed_true = np.clip(lowess_result[:, 1], 0.0, 1.0)
    except Exception as e:
        logger.warning("LOWESS calibration smoothing failed (%s): %s", type(e).__name__, str(e))

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
            intercept = float(int_model.params[0])
        except (ValueError, np.linalg.LinAlgError) as e:
            # Fallback to sklearn method if statsmodels fails
            logger.warning(
                "Statsmodels calibration intercept failed (%s): %s. Falling back to sklearn.",
                type(e).__name__,
                str(e),
            )
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

    # Calibration-in-the-large ratio
    # O:E ratio (Observed / Expected) per Van Calster et al.
    expected = np.sum(y_prob)
    observed = np.sum(y_true)
    oe_ratio = float(observed / expected) if expected > 0 else None
    # Deprecated legacy E/O ratio (Expected / Observed). Use oe_ratio; remove in next major version.
    eo_ratio = expected / observed if observed > 0 else float("inf")

    # ICI/ECI/E_max using smoothed curve when available (Van Calster)
    error_pred = smoothed_pred if len(smoothed_pred) > 0 else prob_pred
    error_true = smoothed_true if len(smoothed_true) > 0 else prob_true

    ici = 0.0
    eci = 0.0
    e_max = 0.0
    if len(error_true) > 0:
        ici = float(np.mean(np.abs(error_true - error_pred)))
        eci_numer = np.mean((error_true - error_pred) ** 2)
        eci_denom = np.mean((prevalence - error_pred) ** 2)
        eci = float(eci_numer / eci_denom) if eci_denom > 0 else 0.0
        e_max = float(np.max(np.abs(error_true - error_pred)))

    return {
        "brier_score": float(brier),
        "brier_scaled": float(brier_scaled),  # Van Calster BSS
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "oe_ratio": oe_ratio,
        "eo_ratio": float(eo_ratio),
        "ici": ici,
        "eci": eci,  # Van Calster E-statistic
        "e_max": e_max,
        "calibration_curve": {
            "prob_true": prob_true.tolist() if len(prob_true) > 0 else [],
            "prob_pred": prob_pred.tolist() if len(prob_pred) > 0 else [],
            "n_bins": n_bins,
        },
        "calibration_curve_smoothed": {
            "prob_true": smoothed_true.tolist() if len(smoothed_true) > 0 else [],
            "prob_pred": smoothed_pred.tolist() if len(smoothed_pred) > 0 else [],
            "method": "lowess",
            "frac": 0.75,
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
    random_seed: int | None = DEFAULT_BOOTSTRAP_SEED,
) -> ClassificationMetrics:
    """Compute classification metrics at a specific threshold.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        threshold: Decision threshold.
        bootstrap_ci: Whether to compute bootstrap CI.
        n_bootstrap: Number of bootstrap iterations.
        random_seed: Random seed for bootstrap resampling.

    Returns:
        Dict with sensitivity, specificity, PPV, NPV, F1, accuracy,
        balanced_accuracy, youden_index, mcc (Matthews), NNE.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Core metrics using safe_divide for consistency
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    ppv = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Additional Van Calster classification metrics
    n_total = tp + tn + fp + fn
    accuracy = safe_divide(tp + tn, n_total)
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

    result: dict[str, Any] = {
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

    # Bootstrap CI for key metrics using centralized bootstrap module
    if bootstrap_ci and len(y_true) > 10:
        seed = DEFAULT_BOOTSTRAP_SEED if random_seed is None else random_seed
        # Use bootstrap_confusion_metrics for sens/spec/ppv CIs
        ci_results = bootstrap_confusion_metrics(
            y_true,
            y_prob,
            threshold=threshold,
            n_bootstrap=n_bootstrap,
            seed=seed,
            stratified=False,  # Backward compatibility
        )

        if len(ci_results["sensitivity"]) > 10:
            sens_ci_lower, sens_ci_upper = compute_percentile_ci(ci_results["sensitivity"])
            spec_ci_lower, spec_ci_upper = compute_percentile_ci(ci_results["specificity"])
            ppv_ci_lower, ppv_ci_upper = compute_percentile_ci(ci_results["ppv"])

            if sens_ci_lower is not None:
                result["sensitivity_ci_95"] = [sens_ci_lower, sens_ci_upper]
            if spec_ci_lower is not None:
                result["specificity_ci_95"] = [spec_ci_lower, spec_ci_upper]
            if ppv_ci_lower is not None:
                result["ppv_ci_95"] = [ppv_ci_lower, ppv_ci_upper]

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
    random_seed: int | None = DEFAULT_BOOTSTRAP_SEED,
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
        random_seed: Random seed for bootstrap resampling.

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
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            auprc = auc(recall, precision)
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
            seed = DEFAULT_BOOTSTRAP_SEED if random_seed is None else random_seed
            rng = np.random.default_rng(seed)
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
