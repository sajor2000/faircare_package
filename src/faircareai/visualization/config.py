"""
FairCareAI Visualization Configuration Constants.

Centralized location for all visualization thresholds, dimensions, and settings.
These constants are used throughout the visualization module to ensure consistency.

Modifying these values will affect all charts in the package.
"""

from typing import Final

# =============================================================================
# CHART DIMENSIONS (pixels)
# =============================================================================

# Default dimensions for most charts
DEFAULT_CHART_WIDTH: Final[int] = 800
DEFAULT_CHART_HEIGHT: Final[int] = 500
MIN_CHART_HEIGHT: Final[int] = 200
MAX_CHART_HEIGHT: Final[int] = 900

# Forest plot dimensions (height scales with number of items)
FOREST_PLOT_BASE_HEIGHT: Final[int] = 300
FOREST_PLOT_PER_ITEM_HEIGHT: Final[int] = 50
FOREST_PLOT_MAX_HEIGHT: Final[int] = 800

# Heatmap dimensions (height scales with number of rows)
HEATMAP_BASE_HEIGHT: Final[int] = 300
HEATMAP_PER_ITEM_HEIGHT: Final[int] = 60
HEATMAP_MAX_HEIGHT: Final[int] = 900

# Radar/spider chart dimensions
RADAR_CHART_SIZE: Final[int] = 500

# Dashboard layout dimensions
DASHBOARD_HEIGHT: Final[int] = 800
DASHBOARD_SUBPLOT_SPACING: Final[float] = 0.08

# =============================================================================
# GHOSTING THRESHOLDS (Sample Size-Based Opacity)
# =============================================================================
# Per CHAI methodology: visual de-emphasis for low sample size groups

# Sample size thresholds
GHOSTING_ADEQUATE_N: Final[int] = 50  # Full opacity, reliable estimates
GHOSTING_MODERATE_N: Final[int] = 30  # Slightly reduced opacity
GHOSTING_LOW_N: Final[int] = 10  # Reduced opacity, interpret with caution
# Below LOW_N = very low opacity, high uncertainty

# Opacity levels corresponding to sample size thresholds
OPACITY_ADEQUATE: Final[float] = 1.0  # n >= GHOSTING_ADEQUATE_N
OPACITY_MODERATE: Final[float] = 0.7  # GHOSTING_MODERATE_N <= n < GHOSTING_ADEQUATE_N
OPACITY_LOW: Final[float] = 0.3  # GHOSTING_LOW_N <= n < GHOSTING_MODERATE_N
OPACITY_VERY_LOW: Final[float] = 0.15  # n < GHOSTING_LOW_N

# =============================================================================
# EXPORT SETTINGS
# =============================================================================

# Default export dimensions (larger than display for quality)
DEFAULT_EXPORT_WIDTH: Final[int] = 1200
DEFAULT_EXPORT_HEIGHT: Final[int] = 800
DEFAULT_EXPORT_SCALE: Final[float] = 2.0  # Retina/HiDPI quality

# Journal publication settings
JOURNAL_EXPORT_WIDTH: Final[int] = 1200
JOURNAL_EXPORT_HEIGHT: Final[int] = 800
JOURNAL_EXPORT_DPI: Final[int] = 300

# Presentation settings
PRESENTATION_WIDTH: Final[int] = 1920
PRESENTATION_HEIGHT: Final[int] = 1080

# =============================================================================
# CHART MARGINS (pixels)
# =============================================================================

# Standard margins for most charts
MARGIN_LEFT: Final[int] = 80
MARGIN_RIGHT: Final[int] = 40
MARGIN_TOP: Final[int] = 60
MARGIN_BOTTOM: Final[int] = 80

# Forest plot margins (need more left space for labels)
FOREST_MARGIN_LEFT: Final[int] = 180
FOREST_MARGIN_RIGHT: Final[int] = 100
FOREST_MARGIN_TOP: Final[int] = 100
FOREST_MARGIN_BOTTOM: Final[int] = 80

# Heatmap margins
HEATMAP_MARGIN_LEFT: Final[int] = 120
HEATMAP_MARGIN_RIGHT: Final[int] = 100
HEATMAP_MARGIN_TOP: Final[int] = 80
HEATMAP_MARGIN_BOTTOM: Final[int] = 100

# =============================================================================
# PERFORMANCE THRESHOLDS (for status indicators)
# =============================================================================

# AUROC thresholds
AUROC_EXCELLENT: Final[float] = 0.9
AUROC_GOOD: Final[float] = 0.8
AUROC_ACCEPTABLE: Final[float] = 0.7
AUROC_POOR: Final[float] = 0.6

# Brier score thresholds (lower is better)
BRIER_EXCELLENT: Final[float] = 0.1
BRIER_GOOD: Final[float] = 0.15
BRIER_ACCEPTABLE: Final[float] = 0.25

# Calibration slope thresholds
CALIBRATION_SLOPE_IDEAL: Final[float] = 1.0
CALIBRATION_SLOPE_TOLERANCE: Final[float] = 0.2  # Acceptable range: 0.8 - 1.2

# =============================================================================
# DISPARITY THRESHOLDS (for fairness assessment)
# =============================================================================

# Disparity index thresholds (1.0 = parity)
DISPARITY_EXCELLENT: Final[float] = 0.1  # |DI - 1| < 0.1 → excellent
DISPARITY_ACCEPTABLE: Final[float] = 0.2  # |DI - 1| < 0.2 → acceptable
DISPARITY_CONCERNING: Final[float] = 0.3  # |DI - 1| < 0.3 → concerning
# Above CONCERNING = requires attention

# =============================================================================
# SAFE ZONE CONFIGURATION (for forest plots)
# =============================================================================

# Safe zone bounds (where performance is considered acceptable)
SAFE_ZONE_MIN: Final[float] = 0.8  # Lower bound of safe zone
SAFE_ZONE_MAX: Final[float] = 1.2  # Upper bound of safe zone

# Reference line position
REFERENCE_LINE_VALUE: Final[float] = 1.0  # Parity/ideal reference

# =============================================================================
# HISTOGRAM/DISTRIBUTION SETTINGS
# =============================================================================

DEFAULT_N_BINS: Final[int] = 20
CALIBRATION_N_BINS: Final[int] = 10
RISK_DISTRIBUTION_N_BINS: Final[int] = 25

# =============================================================================
# ANIMATION SETTINGS
# =============================================================================

# Transition duration in milliseconds
TRANSITION_DURATION_MS: Final[int] = 500
HOVER_TRANSITION_MS: Final[int] = 100

# =============================================================================
# LINE AND MARKER SETTINGS
# =============================================================================

# Default line widths
LINE_WIDTH_THIN: Final[int] = 1
LINE_WIDTH_NORMAL: Final[int] = 2
LINE_WIDTH_THICK: Final[int] = 3
LINE_WIDTH_EMPHASIS: Final[int] = 4

# Default marker sizes
MARKER_SIZE_SMALL: Final[int] = 8
MARKER_SIZE_NORMAL: Final[int] = 12
MARKER_SIZE_LARGE: Final[int] = 16
MARKER_SIZE_EMPHASIS: Final[int] = 20

# Whisker/error bar settings
WHISKER_WIDTH: Final[int] = 2
WHISKER_CAP_SIZE: Final[int] = 6
