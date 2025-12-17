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
