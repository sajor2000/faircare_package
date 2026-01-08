"""FairCareAI Bootstrap Confidence Interval Utilities.

Provides standardized bootstrap CI computation for all metrics.
This module consolidates the bootstrap logic that was previously duplicated
across performance.py, fairness.py, and subgroup.py.

Usage:
    from faircareai.core.bootstrap import bootstrap_metric, compute_percentile_ci
    from sklearn.metrics import roc_auc_score

    samples, n_failed = bootstrap_metric(
        y_true, y_prob,
        lambda yt, yp: roc_auc_score(yt, yp),
        n_bootstrap=1000,
    )
    ci_lower, ci_upper = compute_percentile_ci(samples)
"""

from collections.abc import Callable
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from faircareai.core.constants import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP,
    MIN_BOOTSTRAP_SAMPLES,
)
from faircareai.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", float, NDArray[np.floating])


def bootstrap_metric(
    y_true: NDArray[np.integer] | NDArray[np.floating],
    y_prob: NDArray[np.floating],
    metric_fn: Callable[[NDArray, NDArray], T],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    min_classes: int = 2,
    stratified: bool = True,
) -> tuple[list[T], int]:
    """Compute bootstrap samples for a metric.

    Performs bootstrap resampling (stratified by default) and computes the specified
    metric for each bootstrap sample. Handles edge cases where bootstrap
    samples don't have sufficient class variation.

    Args:
        y_true: True labels array (binary or continuous).
        y_prob: Predicted probabilities array.
        metric_fn: Function that computes metric from (y_true, y_prob).
            Should accept two numpy arrays and return a float or array.
        n_bootstrap: Number of bootstrap iterations (default: 1000).
        seed: Random seed for reproducibility (default: 42).
        min_classes: Minimum unique classes required in bootstrap sample
            (default: 2 for binary classification).
        stratified: If True, preserve class proportions in each bootstrap sample.
            Recommended for imbalanced datasets (default: True).

    Returns:
        Tuple of:
            - list of metric samples (may be shorter than n_bootstrap if some failed)
            - count of failed iterations

    Example:
        >>> from sklearn.metrics import roc_auc_score
        >>> samples, n_failed = bootstrap_metric(
        ...     y_true, y_prob,
        ...     lambda yt, yp: roc_auc_score(yt, yp),
        ...     n_bootstrap=1000,
        ...     stratified=True,
        ... )
        >>> print(f"Computed {len(samples)} samples, {n_failed} failed")

    Note:
        Stratified bootstrap preserves class proportions by sampling within each
        class separately, then combining. This provides tighter confidence intervals
        for imbalanced datasets (prevalence < 10% or > 90%) while maintaining
        correct coverage properties.
    """
    samples: list[T] = []
    n_failed = 0

    rng = np.random.default_rng(seed)
    n = len(y_true)

    if n == 0:
        logger.warning("Bootstrap called with empty arrays")
        return [], 0

    # Pre-compute class stratification for stratified bootstrap
    if stratified:
        unique_classes = np.unique(y_true)
        # Fall back to simple bootstrap if only one class (stratification not possible)
        if len(unique_classes) < 2:
            stratified = False
        else:
            # Build index map for each class
            class_indices_map = {
                class_value: np.where(y_true == class_value)[0]
                for class_value in unique_classes
            }

    for i in range(n_bootstrap):
        if stratified:
            # Stratified bootstrap: sample within each class separately
            idx_list = []
            for class_value in unique_classes:
                class_indices = class_indices_map[class_value]
                n_class = len(class_indices)
                # Sample with replacement within this class
                sampled = rng.choice(class_indices, size=n_class, replace=True)
                idx_list.extend(sampled)

            # Shuffle to mix classes
            idx = np.array(idx_list)
            rng.shuffle(idx)
        else:
            # Simple bootstrap: sample from all indices
            idx = rng.choice(n, size=n, replace=True)

        y_true_boot = y_true[idx]
        y_prob_boot = y_prob[idx]

        # Check for sufficient class variation
        if len(np.unique(y_true_boot)) < min_classes:
            n_failed += 1
            continue

        try:
            result = metric_fn(y_true_boot, y_prob_boot)
            samples.append(result)
        except (ValueError, RuntimeWarning) as e:
            logger.debug(
                "Bootstrap iteration %d failed: %s",
                i,
                str(e),
            )
            n_failed += 1
            continue

    if n_failed > 0:
        logger.debug(
            "Bootstrap: %d/%d iterations failed due to sampling or computation errors",
            n_failed,
            n_bootstrap,
        )

    return samples, n_failed


def bootstrap_confusion_metrics(
    y_true: NDArray[np.integer],
    y_prob: NDArray[np.floating],
    threshold: float,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    stratified: bool = True,
) -> dict[str, list[float]]:
    """Compute bootstrap samples for confusion matrix-derived metrics.

    Specialized bootstrap for metrics derived from the confusion matrix
    (sensitivity, specificity, PPV, NPV) at a specific threshold.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        threshold: Decision threshold for classification.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.
        stratified: If True, preserve class proportions in each bootstrap sample
            (default: True).

    Returns:
        Dict with lists of bootstrap samples for each metric:
            - sensitivity: True positive rate samples
            - specificity: True negative rate samples
            - ppv: Positive predictive value samples
            - npv: Negative predictive value samples
    """
    from sklearn.metrics import confusion_matrix

    results: dict[str, list[float]] = {
        "sensitivity": [],
        "specificity": [],
        "ppv": [],
        "npv": [],
    }

    rng = np.random.default_rng(seed)
    n = len(y_true)
    n_failed = 0

    # Pre-compute class stratification for stratified bootstrap
    if stratified:
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            stratified = False
        else:
            class_indices_map = {
                class_value: np.where(y_true == class_value)[0]
                for class_value in unique_classes
            }

    for i in range(n_bootstrap):
        if stratified:
            # Stratified bootstrap
            idx_list = []
            for class_value in unique_classes:
                class_indices = class_indices_map[class_value]
                n_class = len(class_indices)
                sampled = rng.choice(class_indices, size=n_class, replace=True)
                idx_list.extend(sampled)

            idx = np.array(idx_list)
            rng.shuffle(idx)
        else:
            # Simple bootstrap
            idx = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = (y_prob[idx] >= threshold).astype(int)

        try:
            tn, fp, fn, tp = confusion_matrix(y_true_boot, y_pred_boot, labels=[0, 1]).ravel()

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

            results["sensitivity"].append(sens)
            results["specificity"].append(spec)
            results["ppv"].append(ppv)
            results["npv"].append(npv)

        except ValueError as e:
            logger.debug("Bootstrap confusion matrix iteration %d failed: %s", i, str(e))
            n_failed += 1
            continue

    if n_failed > 0:
        logger.debug(
            "Bootstrap confusion metrics: %d/%d iterations failed",
            n_failed,
            n_bootstrap,
        )

    return results


def compute_percentile_ci(
    samples: list[float],
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float | None, float | None]:
    """Compute percentile confidence interval from bootstrap samples.

    Uses the percentile method to compute confidence intervals.
    Returns None values if insufficient samples are available.

    Args:
        samples: Bootstrap sample values.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound), or (None, None) if insufficient samples.

    Example:
        >>> samples = [0.75, 0.78, 0.82, 0.79, 0.81]  # AUROC bootstrap samples
        >>> ci_lower, ci_upper = compute_percentile_ci(samples, alpha=0.05)
        >>> print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
    """
    if len(samples) < MIN_BOOTSTRAP_SAMPLES:
        logger.warning(
            "Insufficient bootstrap samples (%d < %d) for CI computation",
            len(samples),
            MIN_BOOTSTRAP_SAMPLES,
        )
        return None, None

    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100

    ci = np.percentile(samples, [lower_pct, upper_pct])
    return float(ci[0]), float(ci[1])


def compute_ci_from_samples(
    samples: list[float],
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float | str | None]:
    """Compute confidence interval with formatted output.

    Convenience function that returns a dictionary with CI bounds
    and a formatted string representation.

    Args:
        samples: Bootstrap sample values.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Dict containing:
            - lower: Lower CI bound (or None)
            - upper: Upper CI bound (or None)
            - ci_fmt: Formatted string "(95% CI: X.XXX-X.XXX)" or None

    Example:
        >>> ci_info = compute_ci_from_samples(auroc_samples)
        >>> print(f"AUROC = {auroc:.3f} {ci_info['ci_fmt']}")
    """
    lower, upper = compute_percentile_ci(samples, alpha)

    confidence_pct = int((1 - alpha) * 100)

    if lower is not None and upper is not None:
        ci_fmt = f"({confidence_pct}% CI: {lower:.3f}-{upper:.3f})"
    else:
        ci_fmt = None

    return {
        "lower": lower,
        "upper": upper,
        "ci_fmt": ci_fmt,
    }


def bootstrap_auroc(
    y_true: NDArray[np.integer],
    y_prob: NDArray[np.floating],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    stratified: bool = True,
) -> tuple[list[float], float | None, float | None]:
    """Bootstrap AUROC with confidence interval.

    Convenience function specifically for AUROC bootstrapping,
    the most common use case.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.
        stratified: If True, preserve class proportions in each bootstrap sample
            (default: True).

    Returns:
        Tuple of:
            - list of AUROC samples
            - lower CI bound (or None)
            - upper CI bound (or None)

    Example:
        >>> samples, ci_lower, ci_upper = bootstrap_auroc(y_true, y_prob)
        >>> if ci_lower is not None:
        ...     print(f"AUROC 95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
    """
    from sklearn.metrics import roc_auc_score

    samples, _ = bootstrap_metric(
        y_true,
        y_prob,
        lambda yt, yp: roc_auc_score(yt, yp),
        n_bootstrap=n_bootstrap,
        seed=seed,
        min_classes=2,
        stratified=stratified,
    )

    ci_lower, ci_upper = compute_percentile_ci(samples)

    return samples, ci_lower, ci_upper
