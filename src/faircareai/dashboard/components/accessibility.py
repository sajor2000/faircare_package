"""
WCAG 2.1 AA Accessibility Utilities

Provides accessibility features for FairCareAI dashboard:
- Alt text generation for charts
- ARIA landmarks and live regions
- Skip links and focus management
- Screen reader announcements
"""

from typing import Any

import streamlit as st

# WCAG 2.1 AA compliant CSS additions
ACCESSIBILITY_CSS = """
<style>
/* Focus Indicators - WCAG 2.4.7 Focus Visible */
*:focus-visible {
    outline: 3px solid #0072B2 !important;
    outline-offset: 2px !important;
}

/* Skip Link - WCAG 2.4.1 Bypass Blocks */
.skip-link {
    position: absolute;
    top: -40px;
    left: 0;
    background: #0072B2;
    color: white !important;
    padding: 8px 16px;
    z-index: 9999;
    text-decoration: none;
    font-weight: 600;
    border-radius: 0 0 4px 0;
}

.skip-link:focus {
    top: 0;
}

/* Screen Reader Only - Hidden but accessible */
.sr-only {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    padding: 0 !important;
    margin: -1px !important;
    overflow: hidden !important;
    clip: rect(0, 0, 0, 0) !important;
    white-space: nowrap !important;
    border: 0 !important;
}

/* Reduced Motion - WCAG 2.3.3 Animation from Interactions */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}

/* High Contrast Support */
@media (prefers-contrast: high) {
    .scorecard-item {
        border: 2px solid currentColor !important;
    }

    .status-pass, .status-warn, .status-fail {
        border-width: 3px !important;
    }

    [data-testid="stMetric"] {
        border: 2px solid #121212 !important;
    }
}

/* Live Region for Announcements */
.aria-live-region {
    position: absolute;
    left: -10000px;
    width: 1px;
    height: 1px;
    overflow: hidden;
}

/* Ensure sufficient color contrast on hover states */
.dataframe tr:hover td {
    background: #FFF9C4 !important;
    color: #121212 !important;
}

/* Visible focus for custom interactive elements */
.scorecard-item:focus-within,
.legend-item:focus-within {
    outline: 3px solid #0072B2;
    outline-offset: 2px;
}
</style>
"""


def inject_accessibility_css() -> None:
    """Inject WCAG 2.1 AA compliant CSS into Streamlit."""
    st.markdown(ACCESSIBILITY_CSS, unsafe_allow_html=True)


def create_skip_link(target_id: str = "main-content") -> str:
    """Create a skip link for keyboard navigation.

    WCAG 2.4.1: Bypass Blocks

    Args:
        target_id: ID of the main content element to skip to.

    Returns:
        HTML string for skip link.
    """
    return f'''
    <a href="#{target_id}" class="skip-link">Skip to main content</a>
    <div id="{target_id}" tabindex="-1"></div>
    '''


def render_skip_link(target_id: str = "main-content") -> None:
    """Render skip link in Streamlit."""
    st.markdown(create_skip_link(target_id), unsafe_allow_html=True)


def announce_status_change(message: str, priority: str = "polite") -> None:
    """Announce status change to screen readers.

    WCAG 4.1.3: Status Messages

    Args:
        message: Message to announce.
        priority: "polite" (wait) or "assertive" (interrupt).
    """
    st.markdown(
        f'''<div role="status" aria-live="{priority}" aria-atomic="true" class="aria-live-region">
            {message}
        </div>''',
        unsafe_allow_html=True,
    )


def generate_chart_alt_text(
    chart_type: str,
    data_summary: dict[str, Any],
    key_findings: list[str] | None = None,
) -> str:
    """Generate descriptive alt text for charts.

    WCAG 1.1.1: Non-text Content

    Args:
        chart_type: Type of chart (e.g., "forest_plot", "calibration", "roc").
        data_summary: Dictionary with chart data summary.
        key_findings: Optional list of key findings to highlight.

    Returns:
        Descriptive alt text string.
    """
    chart_descriptions = {
        "forest_plot": _generate_forest_plot_alt,
        "calibration": _generate_calibration_alt,
        "roc": _generate_roc_alt,
        "heatmap": _generate_heatmap_alt,
        "bar": _generate_bar_alt,
        "threshold": _generate_threshold_alt,
        "decision_curve": _generate_decision_curve_alt,
    }

    generator = chart_descriptions.get(chart_type, _generate_generic_alt)
    alt_text = generator(data_summary)

    if key_findings:
        findings_text = " Key findings: " + "; ".join(key_findings)
        alt_text += findings_text

    return alt_text


def _generate_forest_plot_alt(data: dict) -> str:
    """Generate alt text for forest plot."""
    n_groups = data.get("n_groups", 0)
    metric = data.get("metric", "metric")
    attribute = data.get("attribute", "attribute")
    reference = data.get("reference_group", "reference")
    range_min = data.get("range_min", 0)
    range_max = data.get("range_max", 1)
    flagged_count = data.get("flagged_count", 0)

    alt = (
        f"Forest plot comparing {metric.upper()} across {n_groups} groups for {attribute}. "
        f"Reference group: {reference}. "
        f"Values range from {range_min:.1%} to {range_max:.1%}. "
    )

    if flagged_count > 0:
        alt += (
            f"{flagged_count} group(s) flagged for review due to disparities exceeding threshold."
        )
    else:
        alt += "No groups flagged for disparities."

    return alt


def _generate_calibration_alt(data: dict) -> str:
    """Generate alt text for calibration curve."""
    n_groups = data.get("n_groups", 1)
    brier_score = data.get("brier_score")
    slope = data.get("calibration_slope")

    alt = "Calibration curve showing predicted vs observed probabilities"

    if n_groups > 1:
        alt += f" for {n_groups} demographic groups"

    alt += ". Perfect calibration follows the diagonal line."

    if brier_score is not None:
        alt += f" Brier score: {brier_score:.4f}."

    if slope is not None:
        alt += f" Calibration slope: {slope:.2f} (ideal: 1.00)."

    return alt


def _generate_roc_alt(data: dict) -> str:
    """Generate alt text for ROC curve."""
    n_groups = data.get("n_groups", 1)
    auc_values = data.get("auc_values", {})

    alt = "ROC curve showing sensitivity vs false positive rate"

    if n_groups > 1:
        alt += f" for {n_groups} demographic groups"

    alt += ". Higher area under curve (AUC) indicates better discrimination."

    if auc_values:
        auc_text = ", ".join([f"{k}: {v:.3f}" for k, v in auc_values.items()])
        alt += f" AUC values: {auc_text}."

    return alt


def _generate_heatmap_alt(data: dict) -> str:
    """Generate alt text for disparity heatmap."""
    n_rows = data.get("n_rows", 0)
    n_cols = data.get("n_cols", 0)
    metric = data.get("metric", "disparity")
    max_disparity = data.get("max_disparity")

    alt = f"Heatmap showing {metric} comparisons in a {n_rows}x{n_cols} grid."

    if max_disparity is not None:
        alt += f" Maximum disparity: {max_disparity:.1%}."

    return alt


def _generate_bar_alt(data: dict) -> str:
    """Generate alt text for bar chart."""
    n_bars = data.get("n_bars", 0)
    metric = data.get("metric", "value")
    max_val = data.get("max_value")
    min_val = data.get("min_value")

    alt = f"Bar chart comparing {metric} across {n_bars} categories."

    if max_val is not None and min_val is not None:
        alt += f" Values range from {min_val:.1%} to {max_val:.1%}."

    return alt


def _generate_threshold_alt(data: dict) -> str:
    """Generate alt text for threshold analysis chart."""
    current_threshold = data.get("threshold", 0.5)
    sensitivity = data.get("sensitivity")
    specificity = data.get("specificity")

    alt = "Threshold analysis showing metric trade-offs at different decision thresholds."
    alt += f" Current threshold: {current_threshold:.0%}."

    if sensitivity is not None and specificity is not None:
        alt += f" At this threshold: sensitivity {sensitivity:.1%}, specificity {specificity:.1%}."

    return alt


def _generate_decision_curve_alt(data: dict) -> str:
    """Generate alt text for decision curve analysis."""
    threshold_range = data.get("threshold_range", (0, 1))

    alt = (
        f"Decision curve analysis showing net benefit across threshold probabilities "
        f"from {threshold_range[0]:.0%} to {threshold_range[1]:.0%}. "
        "Compares model performance against treat-all and treat-none strategies."
    )

    return alt


def _generate_generic_alt(data: dict) -> str:
    """Generate generic alt text for unknown chart types."""
    title = data.get("title", "Chart")
    return f"{title}. Data visualization showing analysis results."


def accessible_plotly_chart(
    fig: Any,
    chart_id: str,
    alt_text: str,
    use_container_width: bool = True,
    **kwargs: Any,
) -> None:
    """Render Plotly chart with accessibility enhancements.

    Args:
        fig: Plotly Figure object.
        chart_id: Unique identifier for the chart.
        alt_text: Descriptive alt text for screen readers.
        use_container_width: Whether to use container width.
        **kwargs: Additional arguments for st.plotly_chart.
    """
    # Add ARIA description to figure
    fig.update_layout(
        meta={"description": alt_text},
    )

    # Render with accessible wrapper
    st.markdown(
        f'<div role="img" aria-label="{alt_text}" id="{chart_id}-container">',
        unsafe_allow_html=True,
    )

    st.plotly_chart(fig, use_container_width=use_container_width, key=chart_id, **kwargs)

    # Hidden text for screen readers with full description
    st.markdown(
        f'<span class="sr-only">{alt_text}</span></div>',
        unsafe_allow_html=True,
    )


def create_data_table_summary(
    df: Any,
    table_id: str,
    caption: str,
    summary: str | None = None,
) -> str:
    """Create accessible table summary for screen readers.

    WCAG 1.3.1: Info and Relationships

    Args:
        df: DataFrame being displayed.
        table_id: Unique identifier for the table.
        caption: Visible caption for the table.
        summary: Optional detailed summary for screen readers.

    Returns:
        HTML string with table accessibility attributes.
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    columns = ", ".join(df.columns[:5])

    if len(df.columns) > 5:
        columns += f", and {len(df.columns) - 5} more columns"

    if summary is None:
        summary = f"Table with {n_rows} rows and {n_cols} columns. Columns: {columns}."

    return f'''
    <div id="{table_id}-description" class="sr-only">
        {summary}
    </div>
    <div aria-describedby="{table_id}-description">
        <p style="font-weight: 600; margin-bottom: 8px;">{caption}</p>
    </div>
    '''


def render_semantic_heading(
    text: str,
    level: int = 2,
    id: str | None = None,
) -> None:
    """Render semantic heading with proper hierarchy.

    WCAG 1.3.1: Info and Relationships
    WCAG 2.4.6: Headings and Labels

    Args:
        text: Heading text.
        level: Heading level (1-6).
        id: Optional ID for anchor links.
    """
    level = max(1, min(6, level))  # Clamp to valid range
    id_attr = f' id="{id}"' if id else ""
    st.markdown(f"<h{level}{id_attr}>{text}</h{level}>", unsafe_allow_html=True)


def create_aria_landmark(
    content: str,
    role: str,
    label: str,
) -> str:
    """Wrap content in ARIA landmark.

    Args:
        content: HTML content to wrap.
        role: ARIA role (main, navigation, region, etc.).
        label: Accessible label for the region.

    Returns:
        HTML with ARIA landmark.
    """
    return f'<div role="{role}" aria-label="{label}">{content}</div>'


def make_figure_accessible(
    fig: Any,
    chart_type: str,
    data_summary: dict[str, Any],
    key_findings: list[str] | None = None,
) -> Any:
    """Automatically add alt text and ARIA attributes to a Plotly figure.

    This is a convenience function that wraps generate_chart_alt_text and
    applies the result to the figure's meta attribute.

    WCAG 1.1.1: Non-text Content

    Args:
        fig: Plotly Figure object.
        chart_type: Type of chart (forest_plot, calibration, roc, etc.).
        data_summary: Dictionary with chart data summary.
        key_findings: Optional list of key findings to highlight.

    Returns:
        The figure with accessibility metadata applied.
    """
    alt_text = generate_chart_alt_text(chart_type, data_summary, key_findings)
    fig.update_layout(meta={"description": alt_text})
    return fig
