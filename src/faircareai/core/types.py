"""FairCareAI Type Definitions.

Provides TypedDict and Protocol definitions for type-safe metric results.
These types document the expected structure of dictionaries returned by
metric computation functions.

Usage:
    from faircareai.core.types import FairnessResult, GroupMetrics

    def compute_fairness_metrics(...) -> FairnessResult:
        ...
"""

from enum import Enum
from typing import TypedDict

from typing_extensions import NotRequired

# =============================================================================
# STATUS ENUMS
# =============================================================================


class StatusLevel(str, Enum):
    """Status levels for metric threshold evaluations.

    Used throughout the codebase to indicate pass/warn/fail states.
    Inherits from str for JSON serialization compatibility.
    """

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class GovernanceStatus(str, Enum):
    """Governance review status for audit results.

    Indicates overall readiness for deployment from governance perspective.
    """

    READY = "READY"
    CONDITIONAL = "CONDITIONAL"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


class SeverityLevel(str, Enum):
    """Severity levels for audit flags and warnings.

    Used to categorize the importance of identified issues.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# TYPED DICTS
# =============================================================================


class GroupMetrics(TypedDict):
    """Metrics for a single demographic group.

    Contains all computed metrics for one group within a fairness analysis.
    """

    n: int
    """Sample size for this group."""

    prevalence: float
    """Outcome prevalence (proportion with y_true=1)."""

    selection_rate: float
    """Proportion predicted positive at threshold."""

    tpr: float
    """True Positive Rate (sensitivity/recall)."""

    fpr: float
    """False Positive Rate (1 - specificity)."""

    ppv: float
    """Positive Predictive Value (precision)."""

    npv: float
    """Negative Predictive Value."""

    mean_predicted_prob: float
    """Mean predicted probability for this group."""

    mean_calibration_error: float
    """Mean difference between predicted probability and observed rate (calibration-in-the-large)."""

    tp: int
    """True positives count."""

    fp: int
    """False positives count."""

    tn: int
    """True negatives count."""

    fn: int
    """False negatives count."""

    is_reference: bool
    """Whether this is the reference group for comparisons."""


class GroupMetricsWithError(TypedDict, total=False):
    """Group metrics that may contain an error instead of full metrics.

    Used when computation fails for a group (e.g., insufficient sample size).
    """

    n: int
    """Sample size for this group."""

    error: str
    """Error message if computation failed."""

    # Optional fields present when no error
    prevalence: float
    selection_rate: float
    tpr: float
    fpr: float
    ppv: float
    npv: float
    auroc: float
    auroc_ci_95: list[float | None]


class FairnessResult(TypedDict):
    """Result from compute_fairness_metrics.

    Contains all fairness metrics and comparisons for a sensitive attribute.
    """

    group_col: str
    """Name of the grouping column analyzed."""

    threshold: float
    """Decision threshold used for classification."""

    reference: str
    """Reference group name for ratio/difference calculations."""

    group_metrics: dict[str, GroupMetrics | GroupMetricsWithError]
    """Per-group raw metrics, keyed by group name."""

    demographic_parity_ratio: dict[str, float | None]
    """Selection rate ratios vs reference (80% rule)."""

    demographic_parity_diff: dict[str, float]
    """Selection rate differences vs reference."""

    tpr_diff: dict[str, float]
    """True positive rate differences (equal opportunity)."""

    fpr_diff: dict[str, float]
    """False positive rate differences."""

    equalized_odds_diff: dict[str, float]
    """Max of TPR/FPR differences (equalized odds)."""

    ppv_ratio: dict[str, float | None]
    """PPV ratios vs reference (predictive parity)."""

    calibration_diff: dict[str, float]
    """Calibration error differences vs reference."""

    summary: dict[str, dict]
    """Summary statistics with worst disparities."""

    error: NotRequired[str]
    """Error message if reference group computation failed."""


class DiscriminationMetrics(TypedDict):
    """Discrimination metrics with confidence intervals.

    Contains AUROC, AUPRC, and related curve data.
    """

    auroc: float
    """Area Under ROC Curve."""

    auprc: float
    """Area Under Precision-Recall Curve."""

    average_precision: NotRequired[float]
    """Average precision (AP) score."""

    brier_score: float
    """Brier score for calibration."""

    roc_curve: dict[str, list[float]]
    """ROC curve data with 'fpr', 'tpr', 'thresholds' keys."""

    pr_curve: dict[str, list[float]]
    """PR curve data with 'precision', 'recall', 'thresholds' keys."""

    prevalence: float
    """Outcome prevalence in the dataset."""

    auroc_ci_95: NotRequired[list[float]]
    """95% confidence interval for AUROC [lower, upper]."""

    auprc_ci_95: NotRequired[list[float]]
    """95% confidence interval for AUPRC [lower, upper]."""

    auroc_ci_fmt: NotRequired[str]
    """Formatted CI string like '(95% CI: 0.750-0.850)'."""

    auprc_ci_fmt: NotRequired[str]
    """Formatted CI string for AUPRC."""


class CalibrationMetrics(TypedDict):
    """Calibration metrics for model assessment.

    Contains Brier score, slope, intercept, and calibration curve data.
    """

    brier_score: float
    """Brier score (mean squared error of predictions)."""

    calibration_slope: float
    """Calibration slope (1.0 = perfect, <1 = overfitting, >1 = underfitting)."""

    calibration_intercept: float
    """Calibration intercept from linear regression."""

    oe_ratio: float | None
    """Observed/Expected ratio (calibration-in-the-large)."""

    eo_ratio: NotRequired[float]
    """Deprecated: Expected/Observed ratio. Use oe_ratio. Will be removed in next major release."""

    ici: float
    """Integrated Calibration Index (mean absolute calibration error)."""

    e_max: float
    """Maximum Calibration Error."""

    calibration_curve: dict[str, list[float] | int]
    """Calibration curve data with 'prob_true', 'prob_pred', 'n_bins' keys."""

    interpretation: str
    """Human-readable interpretation of calibration quality."""


class ClassificationMetrics(TypedDict):
    """Classification metrics at a specific threshold.

    Contains sensitivity, specificity, PPV, NPV, and confusion matrix.
    """

    threshold: float
    """Decision threshold used."""

    sensitivity: float
    """True Positive Rate (recall)."""

    specificity: float
    """True Negative Rate."""

    ppv: float
    """Positive Predictive Value (precision)."""

    npv: float
    """Negative Predictive Value."""

    f1_score: float
    """F1 score (harmonic mean of precision and recall)."""

    pct_flagged: float
    """Percentage of samples flagged as positive."""

    nne: float | None
    """Number Needed to Evaluate (1/PPV), None if PPV=0."""

    tp: int
    """True positives count."""

    fp: int
    """False positives count."""

    tn: int
    """True negatives count."""

    fn: int
    """False negatives count."""

    sensitivity_ci_95: NotRequired[list[float]]
    """95% CI for sensitivity."""

    specificity_ci_95: NotRequired[list[float]]
    """95% CI for specificity."""

    ppv_ci_95: NotRequired[list[float]]
    """95% CI for PPV."""


class OverallPerformance(TypedDict):
    """Overall model performance metrics container.

    Top-level structure returned by compute_overall_performance.
    """

    primary_threshold: float
    """Primary decision threshold used for classification metrics."""

    discrimination: DiscriminationMetrics
    """AUROC, AUPRC, and curve data."""

    calibration: CalibrationMetrics
    """Calibration metrics and curve."""

    classification_at_threshold: ClassificationMetrics
    """Classification metrics at primary threshold."""

    threshold_analysis: dict
    """Metrics across multiple thresholds."""

    decision_curve: dict
    """Decision Curve Analysis results."""

    confusion_matrix: dict
    """Confusion matrix data."""


class AuditFlag(TypedDict):
    """A single audit flag or warning.

    Generated when a metric violates configured thresholds.
    """

    severity: str
    """Flag severity: 'warning' or 'error'."""

    category: str
    """Flag category: 'sample_size', 'fairness', or 'data_quality'."""

    message: str
    """Human-readable summary of the issue."""

    details: str
    """Detailed explanation of the flag."""

    chai_criteria: str
    """CHAI criteria reference (e.g., 'AC1.CR92')."""

    attribute: NotRequired[str]
    """Sensitive attribute name (if applicable)."""

    group: NotRequired[str]
    """Group name (if applicable)."""

    metric: NotRequired[str]
    """Metric name (if applicable)."""

    value: NotRequired[float]
    """Observed value that triggered the flag."""

    threshold: NotRequired[float | tuple[float, float]]
    """Threshold that was violated."""


class GovernanceRecommendation(TypedDict):
    """Governance recommendation from audit.

    Contains advisory status and summary of findings.
    """

    status: str
    """Status: 'READY' (within threshold), 'CONDITIONAL' (near), or 'REVIEW' (outside)."""

    advisory: str
    """Human-readable advisory message."""

    disclaimer: str
    """CHAI governance disclaimer."""

    n_errors: int
    """Count of critical errors."""

    n_warnings: int
    """Count of warnings."""

    n_pass: int
    """Count of passed checks."""

    total_checks: int
    """Total checks performed."""

    errors: list[AuditFlag]
    """List of error flags."""

    warnings: list[AuditFlag]
    """List of warning flags."""

    primary_fairness_metric: str | None
    """Selected primary fairness metric value."""

    justification_provided: bool
    """Whether fairness justification was provided."""


class SubgroupPerformanceResult(TypedDict):
    """Subgroup performance metrics.

    Contains performance metrics stratified by a sensitive attribute.
    """

    groups: dict[str, GroupMetricsWithError]
    """Per-group performance metrics."""

    reference: str | None
    """Reference group name."""


class DisparityIndexResult(TypedDict):
    """Aggregate disparity index result.

    Contains composite disparity score and interpretation.
    """

    disparity_index: float
    """Aggregate disparity score [0, 1]."""

    components: dict[str, float]
    """Component scores for each fairness metric."""

    interpretation: dict[str, str]
    """Interpretation dict with 'level', 'color', 'description'."""

    raw_metrics: FairnessResult
    """Raw fairness metrics used in computation."""
