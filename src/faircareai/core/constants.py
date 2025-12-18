"""FairCareAI Constants and Default Values.

Centralized location for all magic numbers, default thresholds, and
configuration values used throughout the package.

Methodology: Van Calster et al. (2025), CHAI RAIC Checkpoint 1.
Note: Default thresholds are evidence-based starting points.
"""

from typing import Final

# =============================================================================
# SAMPLE SIZE THRESHOLDS
# =============================================================================
# Based on statistical power analysis and reporting standards

MIN_SAMPLE_SIZE_SUPPRESS: Final[int] = 5
"""Rule of 5: Suppress results below this threshold for privacy/reliability."""

MIN_SAMPLE_SIZE_FLAG: Final[int] = 10
"""Flag as insufficient - results unreliable below this threshold."""

MIN_SAMPLE_SIZE_WARNING: Final[int] = 30
"""Warn about potential instability (CLT assumption threshold)."""

MIN_SAMPLE_SIZE_ADEQUATE: Final[int] = 50
"""Considered adequate for most analyses."""

MIN_SAMPLE_SIZE_SUBGROUP: Final[int] = 100
"""Default minimum for subgroup analysis (configurable in FairnessConfig)."""

MIN_SAMPLE_SIZE_CALIBRATION: Final[int] = 20
"""Minimum samples for calibration curve computation."""


# =============================================================================
# BOOTSTRAP DEFAULTS
# =============================================================================

DEFAULT_N_BOOTSTRAP: Final[int] = 1000
"""Default number of bootstrap iterations for confidence intervals."""

DEFAULT_N_BOOTSTRAP_SUBGROUP: Final[int] = 500
"""Reduced bootstrap iterations for per-subgroup analysis (performance)."""

DEFAULT_BOOTSTRAP_SEED: Final[int] = 42
"""Default random seed for reproducibility."""

MIN_BOOTSTRAP_SAMPLES: Final[int] = 10
"""Minimum valid bootstrap samples required for CI computation."""

DEFAULT_CONFIDENCE_LEVEL: Final[float] = 0.95
"""Default confidence level for intervals (95%)."""

DEFAULT_ALPHA: Final[float] = 0.05
"""Default significance level (1 - confidence_level)."""


# =============================================================================
# FAIRNESS THRESHOLDS
# =============================================================================
# Based on EEOC 80% rule and fairness literature

DEMOGRAPHIC_PARITY_LOWER: Final[float] = 0.8
"""Lower bound for demographic parity ratio (EEOC 80% rule)."""

DEMOGRAPHIC_PARITY_UPPER: Final[float] = 1.25
"""Upper bound for demographic parity ratio (reciprocal of 80%)."""

EQUALIZED_ODDS_THRESHOLD: Final[float] = 0.1
"""Maximum acceptable equalized odds difference (TPR/FPR)."""

EQUAL_OPPORTUNITY_THRESHOLD: Final[float] = 0.1
"""Maximum acceptable TPR difference."""

CALIBRATION_DIFF_THRESHOLD: Final[float] = 0.05
"""Maximum acceptable calibration difference between groups."""

PREDICTIVE_PARITY_LOWER: Final[float] = 0.8
"""Lower bound for PPV ratio (80%)."""

PREDICTIVE_PARITY_UPPER: Final[float] = 1.25
"""Upper bound for PPV ratio (125%)."""


# =============================================================================
# PERFORMANCE THRESHOLDS
# =============================================================================

MIN_AUROC_THRESHOLD: Final[float] = 0.65
"""Minimum acceptable AUROC for clinical utility."""

MAX_MISSING_RATE: Final[float] = 0.10
"""Maximum acceptable missing data rate (10%)."""

DEFAULT_DECISION_THRESHOLD: Final[float] = 0.5
"""Default decision/classification threshold."""


# =============================================================================
# CALIBRATION PARAMETERS
# =============================================================================

DEFAULT_CALIBRATION_BINS: Final[int] = 10
"""Default number of bins for calibration curve."""

CALIBRATION_SLOPE_OVERFITTING: Final[float] = 0.8
"""Slope below this indicates overfitting."""

CALIBRATION_SLOPE_UNDERFITTING: Final[float] = 1.2
"""Slope above this indicates underfitting."""

BRIER_POOR_THRESHOLD: Final[float] = 0.25
"""Brier score above this indicates poor calibration."""


# =============================================================================
# AUROC INTERPRETATION
# =============================================================================

AUROC_DIFF_NEGLIGIBLE: Final[float] = 0.02
"""AUROC difference below this is negligible."""

AUROC_DIFF_SMALL: Final[float] = 0.05
"""AUROC difference below this is small."""

AUROC_DIFF_MODERATE: Final[float] = 0.1
"""AUROC difference below this is moderate, above is large."""


# =============================================================================
# DISPARITY INDEX WEIGHTS
# =============================================================================
# Weights for computing aggregate disparity index

DISPARITY_WEIGHT_DEMOGRAPHIC_PARITY: Final[float] = 0.2
DISPARITY_WEIGHT_EQUAL_OPPORTUNITY: Final[float] = 0.3
DISPARITY_WEIGHT_EQUALIZED_ODDS: Final[float] = 0.3
DISPARITY_WEIGHT_PREDICTIVE_PARITY: Final[float] = 0.2


# =============================================================================
# DISPARITY INDEX INTERPRETATION
# =============================================================================

DISPARITY_INDEX_LOW: Final[float] = 0.25
"""Disparity index below this is LOW."""

DISPARITY_INDEX_MODERATE: Final[float] = 0.5
"""Disparity index below this is MODERATE."""

DISPARITY_INDEX_HIGH: Final[float] = 0.75
"""Disparity index below this is HIGH, above is SEVERE."""


# =============================================================================
# PROBABILITY CLIPPING
# =============================================================================

PROB_CLIP_MIN: Final[float] = 1e-7
"""Minimum probability for log calculations (avoid log(0))."""

PROB_CLIP_MAX: Final[float] = 1 - 1e-7
"""Maximum probability for log calculations (avoid log(1))."""


# =============================================================================
# VAN CALSTER METRIC CLASSIFICATION (2025)
# =============================================================================
# Based on Van Calster et al. (2025) Lancet Digital Health Table 2
# https://doi.org/10.1016/j.landig.2025.100916
#
# Classification:
# - RECOMMENDED: Essential for all reports (Table 2 "Recommended")
# - OPTIONAL: Acceptable but not essential (Table 2 "Not Essential")
# - USE_WITH_CAUTION: Improper measures (Table 2 "Inadvisable")

# Discrimination metrics
VANCALSTER_RECOMMENDED_DISCRIMINATION: Final[tuple[str, ...]] = ("auroc",)
"""Van Calster RECOMMENDED: AUROC is the key discrimination measure."""

VANCALSTER_OPTIONAL_DISCRIMINATION: Final[tuple[str, ...]] = (
    "discrimination_slope",  # Improper but sometimes useful for internal validation
)
"""Van Calster OPTIONAL: Acceptable for data science teams."""

VANCALSTER_CAUTION_DISCRIMINATION: Final[tuple[str, ...]] = (
    "auprc",  # Mixes statistical with decision-analytical
    "partial_auroc",  # Mixes statistical with decision-analytical
)
"""Van Calster INADVISABLE: Use with explicit caveats only."""

# Calibration metrics
VANCALSTER_RECOMMENDED_CALIBRATION: Final[tuple[str, ...]] = (
    "calibration_plot",  # Smoothed (loess) calibration plot
)
"""Van Calster RECOMMENDED: Calibration plot is essential."""

VANCALSTER_OPTIONAL_CALIBRATION: Final[tuple[str, ...]] = (
    "oe_ratio",  # Interpretable but partial
    "calibration_intercept",  # Hard to interpret
    "calibration_slope",  # Hard to interpret
    "ici",  # Summarizes plot but conceals direction
    "eci",  # Summarizes plot but conceals direction
    "ece",  # Summarizes plot but conceals direction
    "brier_score",  # Strictly proper, useful
    "brier_scaled",  # Strictly proper, useful
)
"""Van Calster OPTIONAL: Acceptable for data science teams."""

# Clinical utility metrics
VANCALSTER_RECOMMENDED_CLINICAL_UTILITY: Final[tuple[str, ...]] = (
    "net_benefit",
    "standardized_net_benefit",
    "decision_curve",
)
"""Van Calster RECOMMENDED: Essential for clinical decision support."""

VANCALSTER_OPTIONAL_CLINICAL_UTILITY: Final[tuple[str, ...]] = ("expected_cost",)
"""Van Calster OPTIONAL: Alternative to net benefit."""

# Overall performance metrics
VANCALSTER_RECOMMENDED_OVERALL: Final[tuple[str, ...]] = (
    "risk_distribution_plot",  # Shows probability distributions by outcome
)
"""Van Calster RECOMMENDED: Distribution plots provide valuable insights."""

VANCALSTER_OPTIONAL_OVERALL: Final[tuple[str, ...]] = (
    "loglikelihood",
    "logloss",
    "mcfadden_r2",
    "cox_snell_r2",
    "nagelkerke_r2",
)
"""Van Calster OPTIONAL: Useful for model selection."""

VANCALSTER_CAUTION_OVERALL: Final[tuple[str, ...]] = (
    "mape",  # Improper
)
"""Van Calster INADVISABLE: Mean absolute prediction error is improper."""

# Classification metrics
VANCALSTER_OPTIONAL_CLASSIFICATION: Final[tuple[str, ...]] = (
    "sensitivity",  # Can be descriptive if reported with specificity
    "specificity",  # Can be descriptive if reported with sensitivity
    "ppv",  # Can be descriptive if reported with NPV
    "npv",  # Can be descriptive if reported with PPV
)
"""Van Calster OPTIONAL: Acceptable if reported in pairs (Sens+Spec, PPV+NPV)."""

VANCALSTER_CAUTION_CLASSIFICATION: Final[tuple[str, ...]] = (
    "accuracy",  # Improper at clinically relevant thresholds
    "balanced_accuracy",  # Improper at clinically relevant thresholds
    "youden_index",  # Improper at clinically relevant thresholds
    "f1_score",  # ONLY measure violating BOTH properness AND clear focus
    "mcc",  # Improper, hard to interpret
    "dor",  # Improper
    "kappa",  # Improper
)
"""Van Calster INADVISABLE: Classification summary measures are improper.

These metrics can mislead - use ONLY for data science exploration with explicit
caveats. Never use for clinical decision making or governance reports.

Van Calster et al. (2025): "We warn against the use of measures that are improper
(13 measures) or that do not have a clear focus on either statistical or
decision-analytical performance (three measures). Remarkably, F1 is the only
measure violating both characteristics."
"""

# Combined sets for easy filtering
VANCALSTER_ALL_RECOMMENDED: Final[tuple[str, ...]] = (
    *VANCALSTER_RECOMMENDED_DISCRIMINATION,
    *VANCALSTER_RECOMMENDED_CALIBRATION,
    *VANCALSTER_RECOMMENDED_CLINICAL_UTILITY,
    *VANCALSTER_RECOMMENDED_OVERALL,
)
"""All Van Calster RECOMMENDED metrics - use for governance reports."""

VANCALSTER_ALL_OPTIONAL: Final[tuple[str, ...]] = (
    *VANCALSTER_OPTIONAL_DISCRIMINATION,
    *VANCALSTER_OPTIONAL_CALIBRATION,
    *VANCALSTER_OPTIONAL_CLINICAL_UTILITY,
    *VANCALSTER_OPTIONAL_OVERALL,
    *VANCALSTER_OPTIONAL_CLASSIFICATION,
)
"""All Van Calster OPTIONAL metrics - acceptable for data science."""

VANCALSTER_ALL_CAUTION: Final[tuple[str, ...]] = (
    *VANCALSTER_CAUTION_DISCRIMINATION,
    *VANCALSTER_CAUTION_OVERALL,
    *VANCALSTER_CAUTION_CLASSIFICATION,
)
"""All Van Calster INADVISABLE metrics - use with explicit caveats only."""
