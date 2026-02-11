"""Shared utility functions for visualization modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    pass

# Import Van Calster constants from core.constants
from faircareai.core.constants import (
    VANCALSTER_ALL_CAUTION,
    VANCALSTER_ALL_OPTIONAL,
    VANCALSTER_ALL_RECOMMENDED,
)

# Import theming constants


def get_metric_category(metric: str) -> str:
    """Get Van Calster category for a metric.

    Categorizes performance metrics according to the Van Calster et al. (2025)
    methodology into RECOMMENDED, OPTIONAL, CAUTION, or UNKNOWN categories.

    Args:
        metric: Metric name (case-insensitive).

    Returns:
        One of "RECOMMENDED", "OPTIONAL", "CAUTION", or "UNKNOWN".

    Example:
        >>> get_metric_category("auroc")
        "RECOMMENDED"
        >>> get_metric_category("accuracy")
        "CAUTION"
    """
    metric_lower = metric.lower()
    if metric_lower in VANCALSTER_ALL_RECOMMENDED:
        return "RECOMMENDED"
    if metric_lower in VANCALSTER_ALL_OPTIONAL:
        return "OPTIONAL"
    if metric_lower in VANCALSTER_ALL_CAUTION:
        return "CAUTION"
    return "UNKNOWN"


def add_source_annotation(
    fig: go.Figure,
    source_note: str | None = None,
    citation: str | None = None,
) -> go.Figure:
    """Add FairCareAI source annotation to a figure.

    NOTE: Source annotations have been moved to HTML report containers to avoid
    Plotly clipping issues. This function now returns the figure unchanged.
    Source attributions are rendered in the HTML report footer.

    Args:
        fig: Plotly Figure object.
        source_note: Deprecated - no longer used.
        citation: Deprecated - no longer used.

    Returns:
        Figure unchanged (annotations handled in HTML).
    """
    # Source annotations moved to HTML report footer to prevent clipping
    return fig
