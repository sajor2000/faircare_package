"""
NYT-Style Chart Components

Editorial design patterns inspired by New York Times data visualization:
- Headline-first design
- Direct labeling (values on data points)
- Contextual annotations
- Source attribution
- Clean, minimal aesthetic
"""

import plotly.graph_objects as go
import polars as pl

from faircareai.visualization.themes import (
    FAIRCAREAI_BRAND,
    GHOSTING_CONFIG,
    GROUP_COLORS,
    SEMANTIC_COLORS,
    TYPOGRAPHY,
    get_plotly_template,
)

# Use unified TYPOGRAPHY which now includes NYT editorial extensions
# (headline_size, deck_size, annotation_size, source_size, callout_size)


def create_nyt_layout(
    title: str,
    subtitle: str | None = None,
    source_note: str | None = None,
    height: int = 450,
) -> dict:
    """Create NYT-style layout configuration.

    Args:
        title: Main headline.
        subtitle: Deck/subtitle text.
        source_note: Data source attribution.
        height: Chart height in pixels.

    Returns:
        Layout configuration dictionary.
    """
    template = get_plotly_template()
    layout = template.get("layout", {}).copy()

    # NYT headline styling
    layout["title"] = {
        "text": f"<b>{title}</b>"
        + (
            f"<br><span style='font-size:{TYPOGRAPHY['deck_size']}px;color:#666666;font-weight:400;'>{subtitle}</span>"
            if subtitle
            else ""
        ),
        "font": {
            "family": TYPOGRAPHY["heading_font"],
            "size": TYPOGRAPHY["headline_size"],
            "color": SEMANTIC_COLORS["text"],
        },
        "x": 0,
        "xanchor": "left",
        "y": 0.98,
        "yanchor": "top",
    }

    layout["height"] = height

    # Add source annotation at bottom (use brand default if not provided)
    effective_source = source_note if source_note is not None else FAIRCAREAI_BRAND["source_note"]
    if effective_source:
        layout["annotations"] = layout.get("annotations", []) + [
            {
                "text": effective_source,
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.12,
                "xanchor": "left",
                "yanchor": "top",
                "font": {
                    "size": TYPOGRAPHY["source_size"],
                    "color": "#999999",
                },
            }
        ]

    return layout


def generate_chart_headline(
    chart_type: str,
    data: dict,
) -> str:
    """Auto-generate insight headline from chart data.

    Args:
        chart_type: Type of chart.
        data: Chart data dictionary.

    Returns:
        Headline string.
    """
    if chart_type == "forest_plot":
        flagged = data.get("flagged_groups", [])
        if flagged:
            worst = flagged[0]
            return (
                f"{worst['group']} shows largest disparity in {data.get('metric', 'performance')}"
            )
        return f"No significant disparities found in {data.get('metric', 'performance')}"

    elif chart_type == "calibration":
        slope = data.get("calibration_slope")
        if slope is not None:
            if slope < 0.8:
                return "Model predictions are overconfident"
            elif slope > 1.2:
                return "Model predictions are underconfident"
            return "Model predictions are well-calibrated"

    elif chart_type == "roc":
        auroc = data.get("auroc")
        if auroc is not None:
            if auroc >= 0.9:
                return "Model shows excellent discrimination"
            elif auroc >= 0.8:
                return "Model shows good discrimination"
            elif auroc >= 0.7:
                return "Model shows acceptable discrimination"
            return "Model discrimination may need improvement"

    elif chart_type == "threshold":
        return "How threshold choice affects clinical metrics"

    return "Analysis Results"


def add_nyt_annotations(
    fig: go.Figure,
    annotations: list[dict],
) -> go.Figure:
    """Add NYT-style editorial annotations to chart.

    Args:
        fig: Plotly Figure.
        annotations: List of annotation configs with keys:
            - x, y: Position
            - text: Annotation text
            - style: "callout", "label", "reference"

    Returns:
        Updated figure.
    """
    annotation_styles = {
        "callout": {
            "bgcolor": "#FFFDE7",
            "bordercolor": "#E0E0E0",
            "borderwidth": 1,
            "borderpad": 8,
            "font": {"size": TYPOGRAPHY["callout_size"], "color": "#333333"},
            "showarrow": True,
            "arrowhead": 0,
            "arrowwidth": 1,
            "arrowcolor": "#999999",
        },
        "label": {
            "bgcolor": "rgba(255,255,255,0.9)",
            "borderpad": 4,
            "font": {"size": TYPOGRAPHY["annotation_size"], "color": "#666666"},
            "showarrow": False,
        },
        "reference": {
            "bgcolor": "transparent",
            "font": {"size": TYPOGRAPHY["annotation_size"], "color": "#999999", "style": "italic"},
            "showarrow": False,
        },
        "highlight": {
            "bgcolor": "#FFE0B2",
            "bordercolor": "#E65100",
            "borderwidth": 1,
            "borderpad": 6,
            "font": {"size": TYPOGRAPHY["callout_size"], "color": "#E65100", "weight": 600},
            "showarrow": True,
            "arrowhead": 2,
            "arrowwidth": 2,
            "arrowcolor": "#E65100",
        },
    }

    plotly_annotations = []
    for ann in annotations:
        style = ann.get("style", "label")
        style_config = annotation_styles.get(style, annotation_styles["label"]).copy()

        plotly_ann = {
            "x": ann.get("x"),
            "y": ann.get("y"),
            "text": ann.get("text", ""),
            "xref": ann.get("xref", "x"),
            "yref": ann.get("yref", "y"),
            **style_config,
        }

        # Override with any custom settings
        if ann.get("ax"):
            plotly_ann["ax"] = ann["ax"]
        if ann.get("ay"):
            plotly_ann["ay"] = ann["ay"]

        plotly_annotations.append(plotly_ann)

    fig.update_layout(annotations=list(fig.layout.annotations or []) + plotly_annotations)
    return fig


def create_nyt_forest_plot(
    metrics_df: pl.DataFrame,
    metric: str,
    headline: str | None = None,
    subtitle: str | None = None,
    annotation_text: str | None = None,
    source_note: str | None = None,
    reference_line: float | None = None,
    threshold: float = 0.1,
    show_direct_labels: bool = True,
) -> go.Figure:
    """Create NYT-style forest plot with editorial enhancements.

    Args:
        metrics_df: DataFrame with columns: group, {metric}, ci_lower, ci_upper, n.
        metric: Metric column name.
        headline: Chart headline (auto-generated if None).
        subtitle: Chart subtitle/deck.
        annotation_text: Contextual annotation to add.
        source_note: Source attribution.
        reference_line: Reference value for comparison.
        threshold: Threshold for flagging disparities.
        show_direct_labels: Show values directly on data points.

    Returns:
        Plotly Figure.
    """
    # Auto-generate headline if not provided
    if headline is None:
        groups_data = metrics_df.to_dicts()
        flagged = [
            g for g in groups_data if abs(g.get(metric, 0) - (reference_line or 0)) > threshold
        ]
        headline = generate_chart_headline(
            "forest_plot",
            {
                "metric": metric,
                "flagged_groups": flagged,
            },
        )

    fig = go.Figure()

    groups = metrics_df["group"].to_list()
    values = metrics_df[metric].to_list()
    ci_lower = (
        metrics_df["ci_lower"].to_list()
        if "ci_lower" in metrics_df.columns
        else [None] * len(groups)
    )
    ci_upper = (
        metrics_df["ci_upper"].to_list()
        if "ci_upper" in metrics_df.columns
        else [None] * len(groups)
    )
    n_values = metrics_df["n"].to_list() if "n" in metrics_df.columns else [100] * len(groups)

    # Add reference line if provided
    if reference_line is not None:
        fig.add_vline(
            x=reference_line,
            line_dash="dash",
            line_color=SEMANTIC_COLORS["text_secondary"],
            line_width=1,
            annotation_text="Reference",
            annotation_position="top",
        )

    # Plot each group with ghosting based on sample size
    for i, (group, val, ci_l, ci_u, n) in enumerate(
        zip(groups, values, ci_lower, ci_upper, n_values)
    ):
        opacity = GHOSTING_CONFIG.get_opacity(n)

        # Determine color based on disparity
        if reference_line is not None:
            diff = abs(val - reference_line)
            if diff > threshold:
                color = SEMANTIC_COLORS["fail"]
            elif diff > threshold * 0.5:
                color = SEMANTIC_COLORS["warn_dark"]
            else:
                color = SEMANTIC_COLORS["pass"]
        else:
            color = GROUP_COLORS[i % len(GROUP_COLORS)]

        # Add error bar
        if ci_l is not None and ci_u is not None:
            fig.add_trace(
                go.Scatter(
                    x=[ci_l, ci_u],
                    y=[group, group],
                    mode="lines",
                    line={"color": color, "width": 2},
                    opacity=opacity,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Add point
        fig.add_trace(
            go.Scatter(
                x=[val],
                y=[group],
                mode="markers",
                marker={
                    "size": 16,  # Larger markers for readability
                    "color": color,
                    "line": {"width": 2, "color": "white"},
                },
                opacity=opacity,
                name=group,
                showlegend=False,
                hovertemplate=f"<b>{group}</b><br>{metric.upper()}: %{{x:.1%}}<br>N: {n:,}<extra></extra>",
            )
        )

        # Direct label (NYT style)
        if show_direct_labels:
            fig.add_annotation(
                x=val,
                y=group,
                text=f"{val:.1%}",
                showarrow=False,
                xanchor="left",
                xshift=10,
                font={"size": TYPOGRAPHY["annotation_size"], "color": color},
            )

    # Add contextual annotation
    if annotation_text:
        add_nyt_annotations(
            fig,
            [
                {
                    "x": 0.95,
                    "y": 0.95,
                    "xref": "paper",
                    "yref": "paper",
                    "text": annotation_text,
                    "style": "callout",
                    "ax": -40,
                    "ay": 30,
                }
            ],
        )

    # Apply NYT layout
    layout = create_nyt_layout(headline, subtitle, source_note)
    layout["xaxis"] = {
        "title": metric.upper(),
        "tickformat": ".0%",
        "showgrid": True,
        "gridcolor": SEMANTIC_COLORS["grid"],
    }
    layout["yaxis"] = {
        "title": "",
        "showgrid": False,
        "categoryorder": "array",
        "categoryarray": list(reversed(groups)),
    }
    layout["margin"] = {"l": 180, "r": 100, "t": 120, "b": 100}

    fig.update_layout(**layout)

    return fig


def create_nyt_bar_chart(
    data: pl.DataFrame,
    x_col: str,
    y_col: str,
    headline: str | None = None,
    subtitle: str | None = None,
    source_note: str | None = None,
    highlight_bars: list[str] | None = None,
    show_direct_labels: bool = True,
) -> go.Figure:
    """Create NYT-style bar chart with direct labeling.

    Args:
        data: DataFrame with data.
        x_col: Column for x-axis (categories).
        y_col: Column for y-axis (values).
        headline: Chart headline.
        subtitle: Chart subtitle.
        source_note: Source attribution.
        highlight_bars: List of x values to highlight.
        show_direct_labels: Show values on bars.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    x_values = data[x_col].to_list()
    y_values = data[y_col].to_list()

    colors = []
    for x in x_values:
        if highlight_bars and x in highlight_bars:
            colors.append(SEMANTIC_COLORS["fail"])
        else:
            colors.append(SEMANTIC_COLORS["primary"])

    fig.add_trace(
        go.Bar(
            x=x_values,
            y=y_values,
            marker_color=colors,
            text=[f"{v:.1%}" for v in y_values] if show_direct_labels else None,
            textposition="outside",
            textfont={"size": TYPOGRAPHY["annotation_size"]},
        )
    )

    layout = create_nyt_layout(headline or y_col.title(), subtitle, source_note)
    layout["yaxis"]["tickformat"] = ".0%"
    layout["showlegend"] = False

    fig.update_layout(**layout)

    return fig


def create_nyt_line_chart(
    data: pl.DataFrame,
    x_col: str,
    y_cols: list[str],
    headline: str | None = None,
    subtitle: str | None = None,
    source_note: str | None = None,
    highlight_point: float | None = None,
    y_format: str = ".1%",
) -> go.Figure:
    """Create NYT-style line chart with end labels.

    Args:
        data: DataFrame with data.
        x_col: Column for x-axis.
        y_cols: Columns for y-axis lines.
        headline: Chart headline.
        subtitle: Chart subtitle.
        source_note: Source attribution.
        highlight_point: X value to highlight with vertical line.
        y_format: Format string for y-axis values.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    x_values = data[x_col].to_list()

    for i, y_col in enumerate(y_cols):
        y_values = data[y_col].to_list()
        color = GROUP_COLORS[i % len(GROUP_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=y_col.title(),
                line={"color": color, "width": 2},
            )
        )

        # NYT-style end label
        fig.add_annotation(
            x=x_values[-1],
            y=y_values[-1],
            text=f"<b>{y_col.title()}</b>",
            showarrow=False,
            xanchor="left",
            xshift=5,
            font={"size": TYPOGRAPHY["annotation_size"], "color": color},
        )

    # Highlight point
    if highlight_point is not None:
        fig.add_vline(
            x=highlight_point,
            line_dash="dash",
            line_color=SEMANTIC_COLORS["text_secondary"],
            annotation_text=f"Current: {highlight_point:{y_format}}",
        )

    layout = create_nyt_layout(headline or "Trend Analysis", subtitle, source_note)
    layout["yaxis"]["tickformat"] = y_format
    layout["showlegend"] = False  # Using end labels instead

    fig.update_layout(**layout)

    return fig


def create_scorecard_figure(
    pass_count: int,
    warn_count: int,
    fail_count: int,
    headline: str = "Audit Results",
) -> go.Figure:
    """Create NYT-style scorecard as a Plotly figure.

    Args:
        pass_count: Number of passing checks.
        warn_count: Number of warnings.
        fail_count: Number of failures.
        headline: Chart headline.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Create indicator-style display
    categories = ["Pass", "Review", "Flag"]
    values = [pass_count, warn_count, fail_count]
    colors = [SEMANTIC_COLORS["pass"], SEMANTIC_COLORS["warn_dark"], SEMANTIC_COLORS["fail"]]

    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=val,
                title={"text": cat, "font": {"size": 14, "color": "#666666"}},
                number={"font": {"size": 48, "color": color}},
                domain={"x": [i / 3, (i + 1) / 3], "y": [0, 1]},
            )
        )

    layout = create_nyt_layout(headline)
    layout["height"] = 150
    layout["margin"] = {"l": 20, "r": 20, "t": 60, "b": 20}

    fig.update_layout(**layout)

    return fig
