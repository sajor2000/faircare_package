"""
FairCareAI Plotly Visualization Components

Interactive charts with NYT/D3 editorial aesthetic and ghosting support.
Design Philosophy: Clean, direct-labeled charts that tell a story.

WCAG 2.1 AA Compliance:
- All charts support alt text generation for screen readers
- Color choices from colorblind-safe Okabe-Ito palette
- Sufficient contrast ratios for text and data elements
"""

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from sklearn.metrics import auc, roc_curve

from .themes import (
    COLORSCALES,
    EDITORIAL_COLORS,
    FAIRCAREAI_BRAND,
    GHOSTING_CONFIG,
    GROUP_COLORS,
    LEGEND_POSITIONS,
    SEMANTIC_COLORS,
    SUBPLOT_SPACING,
    TYPOGRAPHY,
    GhostingConfig,
    calculate_chart_height,
    register_plotly_template,
)

register_plotly_template()


# =============================================================================
# BRANDING HELPER
# =============================================================================


def add_source_annotation(
    fig: go.Figure,
    source_note: str | None = None,
) -> go.Figure:
    """Add FairCareAI source annotation to a figure.

    Args:
        fig: Plotly Figure object.
        source_note: Custom source note (uses brand default if None).

    Returns:
        Figure with source annotation added.
    """
    effective_source = source_note if source_note is not None else FAIRCAREAI_BRAND["source_note"]
    fig.add_annotation(
        text=effective_source,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.12,
        showarrow=False,
        font={"size": TYPOGRAPHY["source_size"], "color": EDITORIAL_COLORS["slate"]},
        xanchor="left",
    )
    return fig


# =============================================================================
# ALT TEXT GENERATION FOR WCAG 2.1 AA COMPLIANCE
# =============================================================================


def generate_forest_plot_alt_text(
    metrics_df: pl.DataFrame,
    metric: str,
    title: str,
    flagged_threshold: float = 0.8,
) -> str:
    """Generate accessible alt text for forest plot.

    Args:
        metrics_df: DataFrame with metrics data.
        metric: Metric being displayed.
        title: Chart title.
        flagged_threshold: Threshold for flagging groups.

    Returns:
        Descriptive alt text for screen readers.
    """
    df = metrics_df.filter(pl.col("group") != "_overall")
    n_groups = len(df)

    if n_groups == 0:
        return f"{title}. No data available for visualization."

    values = df[metric].to_list()
    groups = df["group"].to_list()

    min_val = min(values)
    max_val = max(values)
    flagged = [g for g, v in zip(groups, values) if v < flagged_threshold]

    alt_text = (
        f"{title}. Forest plot comparing {metric.upper()} across {n_groups} demographic groups. "
        f"Values range from {min_val:.1%} to {max_val:.1%}. "
    )

    if flagged:
        alt_text += f"{len(flagged)} group(s) below {flagged_threshold:.0%} threshold: {', '.join(flagged)}."
    else:
        alt_text += f"All groups meet the {flagged_threshold:.0%} threshold."

    return alt_text


def generate_calibration_alt_text(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_labels: np.ndarray | None = None,
    title: str = "Calibration Curve",
) -> str:
    """Generate accessible alt text for calibration plot.

    Args:
        y_true: True outcomes.
        y_prob: Predicted probabilities.
        group_labels: Optional group labels.
        title: Chart title.

    Returns:
        Descriptive alt text for screen readers.
    """
    n_samples = len(y_true)

    if group_labels is not None:
        n_groups = len(np.unique(group_labels))
        alt_text = (
            f"{title}. Calibration curves showing predicted vs observed probabilities "
            f"for {n_groups} demographic groups across {n_samples:,} samples. "
            "Perfect calibration follows the diagonal line from (0,0) to (1,1). "
        )
    else:
        alt_text = (
            f"{title}. Calibration curve showing predicted vs observed probabilities "
            f"for {n_samples:,} samples. "
            "Points closer to the diagonal indicate better calibration."
        )

    return alt_text


def generate_roc_alt_text(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_labels: np.ndarray,
    title: str = "ROC Curves",
) -> str:
    """Generate accessible alt text for ROC curves.

    Args:
        y_true: True outcomes.
        y_prob: Predicted probabilities.
        group_labels: Group labels.
        title: Chart title.

    Returns:
        Descriptive alt text for screen readers.
    """
    unique_groups = np.unique(group_labels)
    auc_values = {}

    for group in unique_groups:
        mask = group_labels == group
        y_t = y_true[mask]
        y_p = y_prob[mask]

        if len(np.unique(y_t)) >= 2:
            fpr, tpr, _ = roc_curve(y_t, y_p)
            auc_values[str(group)] = auc(fpr, tpr)

    alt_text = (
        f"{title}. ROC curves showing sensitivity vs false positive rate "
        f"for {len(unique_groups)} demographic groups. "
        "Higher area under curve (AUC) indicates better discrimination. "
    )

    if auc_values:
        auc_text = "; ".join([f"{k}: {v:.2f}" for k, v in auc_values.items()])
        alt_text += f"AUC values: {auc_text}."

    return alt_text


def generate_heatmap_alt_text(
    disparity_df: pl.DataFrame,
    metric: str,
    title: str,
) -> str:
    """Generate accessible alt text for disparity heatmap.

    Args:
        disparity_df: DataFrame with disparity data.
        metric: Metric being displayed.
        title: Chart title.

    Returns:
        Descriptive alt text for screen readers.
    """
    df = disparity_df.filter(pl.col("metric") == metric)

    if len(df) == 0:
        return f"{title}. No disparity data available."

    max_disparity = df["difference"].abs().max()
    n_significant = df.filter(pl.col("statistically_significant")).shape[0]

    alt_text = (
        f"{title}. Heatmap showing pairwise {metric.upper()} disparities between groups. "
        f"Maximum disparity: {max_disparity:.1%}. "
        f"{n_significant} comparison(s) are statistically significant (p < 0.05). "
        "Blue indicates negative difference, red indicates positive difference."
    )

    return alt_text


def _get_status_color(value: float, threshold: float = 0.8, warn_threshold: float = 0.7) -> str:
    """Get semantic color based on metric value."""
    if value >= threshold:
        return SEMANTIC_COLORS["pass"]
    elif value >= warn_threshold:
        return SEMANTIC_COLORS["warn"]
    return SEMANTIC_COLORS["fail"]


def create_forest_plot(
    metrics_df: pl.DataFrame,
    metric: str = "tpr",
    title: str | None = None,
    subtitle: str | None = None,
    enable_ghosting: bool = True,
    ghosting_config: GhostingConfig | None = None,
    show_safe_zone: bool = True,
    safe_zone_min: float = 0.8,
    reference_line: float | None = None,
    source_note: str | None = None,
) -> go.Figure:
    """
    Create NYT-style forest plot with CI whiskers and ghosting.

    Features:
    - Horizontal lollipop/forest plot style
    - Confidence interval whiskers
    - Green "safe zone" shading for acceptable range
    - Ghosting (opacity reduction) for low sample sizes
    - Direct labeling (no legend needed)
    """
    ghost_cfg = ghosting_config or GHOSTING_CONFIG

    # Validate required columns exist
    required_cols = ["group", metric, "n"]
    missing_cols = [col for col in required_cols if col not in metrics_df.columns]
    if missing_cols:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required columns: {', '.join(missing_cols)}",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["fail"]),
        )
        return fig

    df = metrics_df.filter(pl.col("group") != "_overall").to_pandas()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        return fig

    if title is None:
        metric_names = {
            "tpr": "True Positive Rate (Sensitivity)",
            "fpr": "False Positive Rate",
            "ppv": "Positive Predictive Value",
            "npv": "Negative Predictive Value",
            "accuracy": "Accuracy",
        }
        title = metric_names.get(metric, metric.upper())

    fig = go.Figure()

    # Add safe zone (green shaded region)
    if show_safe_zone:
        fig.add_vrect(
            x0=safe_zone_min,
            x1=1.0,
            fillcolor=SEMANTIC_COLORS["safe_zone"],
            line=dict(color=SEMANTIC_COLORS["safe_zone_border"], width=1),
            layer="below",
            annotation_text="Acceptable",
            annotation_position="top right",
            annotation_font=dict(
                size=TYPOGRAPHY["tick_size"], color=SEMANTIC_COLORS["text_secondary"]
            ),
        )

    # Add reference line if specified
    if reference_line is not None:
        fig.add_vline(
            x=reference_line,
            line=dict(color=SEMANTIC_COLORS["text_secondary"], width=1, dash="dash"),
            annotation_text=f"Reference: {reference_line:.0%}",
            annotation_position="top",
            annotation_font=dict(size=TYPOGRAPHY["tick_size"]),
        )

    # Sort by metric value for visual hierarchy
    if metric not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Metric '{metric}' not found in data",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["fail"]),
        )
        return fig

    df = df.sort_values(metric, ascending=True)

    y_labels = []
    for _idx, row in df.iterrows():
        group = row["group"]
        value = row[metric]
        n = int(row["n"])

        opacity = ghost_cfg.get_opacity(n) if enable_ghosting else 1.0
        badge = ghost_cfg.get_badge(n)

        # Color based on value threshold
        color = _get_status_color(value, safe_zone_min, safe_zone_min - 0.1)

        # Create label with sample size integrated (prevents overlap)
        label = f"{group}  (n={n:,})"
        y_labels.append(label)

        hover_text = f"<b>{group}</b><br>{metric.upper()}: {value:.1%}<br>Sample size: {n:,}"
        if badge:
            hover_text += f"<br><i>{badge}</i>"

        # Add CI whiskers first (so dots appear on top)
        ci_lower = row.get("ci_lower", value - 0.05)
        ci_upper = row.get("ci_upper", value + 0.05)

        # Horizontal CI line
        fig.add_trace(
            go.Scatter(
                x=[ci_lower, ci_upper],
                y=[label, label],
                mode="lines",
                line=dict(width=2, color=color),
                opacity=opacity * 0.6,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # CI caps (small vertical lines at ends)
        for ci_val in [ci_lower, ci_upper]:
            fig.add_trace(
                go.Scatter(
                    x=[ci_val, ci_val],
                    y=[label, label],
                    mode="markers",
                    marker=dict(symbol="line-ns", size=8, line=dict(width=2, color=color)),
                    opacity=opacity * 0.6,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Main point marker
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[label],
                mode="markers+text",
                marker=dict(
                    size=16,  # Larger markers
                    color=color,
                    opacity=opacity,
                    line=dict(width=2, color="white"),
                ),
                text=[f"{value:.0%}"],
                textposition="middle right",
                textfont=dict(size=TYPOGRAPHY["annotation_size"], color=SEMANTIC_COLORS["text"]),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
            )
        )

    # Generate alt text for WCAG 2.1 AA compliance
    alt_text = generate_forest_plot_alt_text(
        metrics_df, metric, title, flagged_threshold=safe_zone_min
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>"
            + (
                f"<br><span style='font-size:14px;color:#666666'>{subtitle}</span>"
                if subtitle
                else ""
            ),
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["heading_size"]),
        ),
        xaxis=dict(
            title=dict(text=metric.upper(), font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickformat=".0%",
            range=[0, 1.05],
            showgrid=True,
            gridcolor=SEMANTIC_COLORS["grid"],
            gridwidth=1,
        ),
        yaxis=dict(
            title=dict(text="Group", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            showgrid=False,
            categoryorder="array",
            categoryarray=y_labels,
        ),
        template="faircareai",
        height=calculate_chart_height(len(df), "forest"),
        margin=dict(l=180, r=100, t=100, b=80),  # Wider left margin for labels with sample sizes
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    add_source_annotation(fig, source_note)
    return fig


def create_disparity_heatmap(
    disparity_df: pl.DataFrame,
    metric: str = "tpr",
    title: str | None = None,
    source_note: str | None = None,
) -> go.Figure:
    """
    Create a heatmap showing pairwise disparities between groups.

    Red cells indicate statistically significant disparities.
    """
    if title is None:
        title = f"Pairwise {metric.upper()} Disparities"

    # Validate required columns exist
    required_cols = [
        "metric",
        "reference_group",
        "comparison_group",
        "difference",
        "statistically_significant",
    ]
    missing_cols = [col for col in required_cols if col not in disparity_df.columns]
    if missing_cols:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required columns: {', '.join(missing_cols)}",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["fail"]),
        )
        return fig

    # Filter for the specified metric
    df = disparity_df.filter(pl.col("metric") == metric).to_pandas()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No disparity data available", x=0.5, y=0.5, showarrow=False)
        return fig

    # Get unique groups
    groups = sorted(set(df["reference_group"].tolist() + df["comparison_group"].tolist()))
    n_groups = len(groups)

    # Create matrix
    matrix = np.zeros((n_groups, n_groups))
    sig_matrix = np.zeros((n_groups, n_groups))

    for _, row in df.iterrows():
        ref_group = row["reference_group"]
        comp_group = row["comparison_group"]

        # Validate groups exist before calling .index()
        if ref_group not in groups or comp_group not in groups:
            continue

        ref_idx = groups.index(ref_group)
        comp_idx = groups.index(comp_group)
        matrix[ref_idx, comp_idx] = row["difference"]
        matrix[comp_idx, ref_idx] = -row["difference"]
        sig_matrix[ref_idx, comp_idx] = row["statistically_significant"]
        sig_matrix[comp_idx, ref_idx] = row["statistically_significant"]

    # Create annotations for cell values
    annotations = []
    for i in range(n_groups):
        for j in range(n_groups):
            if i != j:
                val = matrix[i, j]
                sig = sig_matrix[i, j]
                text = f"{val:+.1%}"
                if sig:
                    text += " *"
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=text,
                        showarrow=False,
                        font=dict(
                            size=TYPOGRAPHY["tick_size"],
                            color="white" if abs(val) > 0.1 else SEMANTIC_COLORS["text"],
                        ),
                    )
                )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=groups,
            y=groups,
            colorscale=COLORSCALES["diverging_disparity"],
            zmid=0,
            zmin=-0.3,
            zmax=0.3,
            showscale=True,
            colorbar=dict(
                title="Difference",
                tickformat="+.0%",
                len=0.6,
            ),
            hovertemplate=("<b>%{y}</b> vs <b>%{x}</b><br>Difference: %{z:+.1%}<extra></extra>"),
        )
    )

    # Generate alt text for WCAG 2.1 AA compliance
    alt_text = generate_heatmap_alt_text(disparity_df, metric, title)

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:{TYPOGRAPHY['body_size']}px;color:#666'>* indicates statistical significance (p < 0.05)</span>",
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["subheading_size"]),
        ),
        xaxis=dict(
            title=dict(text="Comparison Group", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickangle=-45,
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        yaxis=dict(
            title=dict(text="Reference Group", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            autorange="reversed",
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        annotations=annotations,
        template="faircareai",
        height=calculate_chart_height(n_groups, "bar"),
        margin=dict(l=120, r=40, t=120, b=100),  # Proper margins for title and source
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    add_source_annotation(fig, source_note)
    return fig


def create_metric_comparison_chart(
    metrics_df: pl.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Performance Metrics by Group",
) -> go.Figure:
    """
    Create grouped bar chart comparing multiple metrics across groups.
    """
    # Default metrics
    if metrics is None:
        metrics = ["tpr", "fpr", "ppv"]

    # Validate required columns exist
    if "group" not in metrics_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Missing required column: 'group'",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["fail"]),
        )
        return fig

    df = metrics_df.filter(pl.col("group") != "_overall").to_pandas()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        return fig

    groups = df["group"].tolist()

    fig = go.Figure()

    metric_colors = {
        "tpr": SEMANTIC_COLORS["primary"],
        "fpr": SEMANTIC_COLORS["fail"],
        "ppv": SEMANTIC_COLORS["secondary"],
        "npv": GROUP_COLORS[4],
        "accuracy": GROUP_COLORS[5],
    }

    metric_labels = {
        "tpr": "Sensitivity (TPR)",
        "fpr": "False Positive Rate",
        "ppv": "Precision (PPV)",
        "npv": "NPV",
        "accuracy": "Accuracy",
    }

    for metric in metrics:
        if metric in df.columns:
            fig.add_trace(
                go.Bar(
                    name=metric_labels.get(metric, metric.upper()),
                    x=groups,
                    y=df[metric],
                    marker_color=metric_colors.get(metric, GROUP_COLORS[0]),
                    text=[f"{v:.0%}" for v in df[metric]],
                    textposition="outside",
                    textfont=dict(size=TYPOGRAPHY["tick_size"]),
                    hovertemplate=(
                        f"<b>%{{x}}</b><br>"
                        f"{metric_labels.get(metric, metric)}: %{{y:.1%}}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["subheading_size"]),
        ),
        barmode="group",
        xaxis=dict(
            title=dict(text="Group", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        yaxis=dict(
            title=dict(text="Value", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickformat=".0%",
            range=[0, 1.15],
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        legend=LEGEND_POSITIONS["top_horizontal"],
        template="faircareai",
        height=400,
    )

    return fig


def create_summary_scorecard(
    pass_count: int,
    warn_count: int,
    fail_count: int,
    n_samples: int,
    threshold: float,
    model_name: str = "Model",
) -> go.Figure:
    """Create executive summary scorecard with NYT editorial styling."""
    fig = go.Figure()

    # Create three indicator cards with better styling
    cards = [
        ("PASS", pass_count, SEMANTIC_COLORS["pass"], SEMANTIC_COLORS["pass_light"]),
        ("WARN", warn_count, SEMANTIC_COLORS["warn_dark"], SEMANTIC_COLORS["warn_light"]),
        ("FAIL", fail_count, SEMANTIC_COLORS["fail"], SEMANTIC_COLORS["fail_light"]),
    ]

    for i, (label, value, color, bg_color) in enumerate(cards):
        # Background rectangle
        fig.add_shape(
            type="rect",
            x0=i / 3 + 0.02,
            x1=(i + 1) / 3 - 0.02,
            y0=0.2,
            y1=0.95,
            xref="paper",
            yref="paper",
            fillcolor=bg_color,
            line=dict(color=color, width=2),
            layer="below",
        )

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=value,
                title=dict(
                    text=label,
                    font=dict(
                        family=TYPOGRAPHY["data_font"],
                        size=14,
                        color=SEMANTIC_COLORS["text_secondary"],
                    ),
                ),
                number=dict(
                    font=dict(
                        family=TYPOGRAPHY["data_font"],
                        size=56,
                        color=color,
                    ),
                ),
                domain={"x": [i / 3 + 0.02, (i + 1) / 3 - 0.02], "y": [0.25, 0.9]},
            )
        )

    # Metadata annotation
    fig.add_annotation(
        text=f"<b>N = {n_samples:,}</b> samples  •  Threshold = {threshold:.0%}",
        x=0.5,
        y=0.08,
        showarrow=False,
        font=dict(
            family=TYPOGRAPHY["data_font"],
            size=TYPOGRAPHY["annotation_size"],
            color=SEMANTIC_COLORS["text_secondary"],
        ),
        xref="paper",
        yref="paper",
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Equity Audit Summary</b><br><span style='font-size:14px;color:#666666'>{model_name}</span>",
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["heading_size"]),
            x=0,
            xanchor="left",
        ),
        template="faircareai",
        height=280,
        margin=dict(l=20, r=20, t=80, b=40),
    )

    return fig


def create_calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_labels: np.ndarray | None = None,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    source_note: str | None = None,
) -> go.Figure:
    """
    Create calibration plot showing predicted vs actual probabilities.

    If group_labels provided, shows separate curves per group.
    """
    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=SEMANTIC_COLORS["text_secondary"], dash="dash", width=1),
            name="Perfect calibration",
            showlegend=True,
        )
    )

    def compute_calibration(y_t, y_p, bins):
        """Compute calibration bins."""
        bin_edges = np.linspace(0, 1, bins + 1)

        mean_predicted = []
        fraction_positive = []
        counts = []

        for i in range(bins):
            mask = (y_p >= bin_edges[i]) & (y_p < bin_edges[i + 1])
            if mask.sum() > 0:
                mean_predicted.append(y_p[mask].mean())
                fraction_positive.append(y_t[mask].mean())
                counts.append(mask.sum())
            else:
                mean_predicted.append(np.nan)
                fraction_positive.append(np.nan)
                counts.append(0)

        return np.array(mean_predicted), np.array(fraction_positive), np.array(counts)

    if group_labels is None:
        # Single calibration curve
        mean_pred, frac_pos, counts = compute_calibration(y_true, y_prob, n_bins)

        fig.add_trace(
            go.Scatter(
                x=mean_pred,
                y=frac_pos,
                mode="lines+markers",
                line=dict(color=SEMANTIC_COLORS["primary"], width=2),
                marker=dict(size=8, color=SEMANTIC_COLORS["primary"]),
                name="Model",
                hovertemplate=("Predicted: %{x:.1%}<br>Actual: %{y:.1%}<extra></extra>"),
            )
        )
    else:
        # Separate curves per group
        unique_groups = np.unique(group_labels)
        for i, group in enumerate(unique_groups):
            mask = group_labels == group

            # Check for empty masks before array operations
            if not mask.any():
                continue

            mean_pred, frac_pos, counts = compute_calibration(y_true[mask], y_prob[mask], n_bins)

            color = GROUP_COLORS[i % len(GROUP_COLORS)]

            fig.add_trace(
                go.Scatter(
                    x=mean_pred,
                    y=frac_pos,
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=8, color=color),
                    name=str(group),
                    hovertemplate=(
                        f"<b>{group}</b><br>"
                        "Predicted: %{x:.1%}<br>"
                        "Actual: %{y:.1%}<extra></extra>"
                    ),
                )
            )

    # Generate alt text for WCAG 2.1 AA compliance
    alt_text = generate_calibration_alt_text(y_true, y_prob, group_labels, title)

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:{TYPOGRAPHY['body_size']}px;color:#666'>Closer to diagonal = better calibrated</span>",
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["subheading_size"]),
        ),
        xaxis=dict(
            title=dict(
                text="Mean Predicted Probability", font=dict(size=TYPOGRAPHY["axis_title_size"])
            ),
            tickformat=".0%",
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
            range=[0, 1],
        ),
        yaxis=dict(
            title=dict(text="Fraction of Positives", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickformat=".0%",
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
            range=[0, 1],
        ),
        legend=LEGEND_POSITIONS["top_horizontal"],
        template="faircareai",
        height=500,
        margin=dict(l=80, r=40, t=120, b=100),  # Proper margins for title and source
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    add_source_annotation(fig, source_note)
    return fig


def create_roc_curve_by_group(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_labels: np.ndarray,
    title: str = "ROC Curves by Demographic Group",
    source_note: str | None = None,
) -> go.Figure:
    """
    Create ROC curves for each demographic group.
    """
    fig = go.Figure()

    # Diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=SEMANTIC_COLORS["text_secondary"], dash="dash", width=1),
            name="Random (AUC = 0.50)",
            showlegend=True,
        )
    )

    unique_groups = np.unique(group_labels)

    for i, group in enumerate(unique_groups):
        mask = group_labels == group
        y_t = y_true[mask]
        y_p = y_prob[mask]

        if len(np.unique(y_t)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_t, y_p)
        roc_auc = auc(fpr, tpr)

        color = GROUP_COLORS[i % len(GROUP_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{group} (AUC = {roc_auc:.2f})",
                hovertemplate=(
                    f"<b>{group}</b><br>FPR: %{{x:.1%}}<br>TPR: %{{y:.1%}}<extra></extra>"
                ),
            )
        )

    # Generate alt text for WCAG 2.1 AA compliance
    alt_text = generate_roc_alt_text(y_true, y_prob, group_labels, title)

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["subheading_size"]),
        ),
        xaxis=dict(
            title=dict(text="False Positive Rate", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickformat=".0%",
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
            range=[0, 1],
        ),
        yaxis=dict(
            title=dict(text="True Positive Rate", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickformat=".0%",
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
            range=[0, 1],
        ),
        legend=LEGEND_POSITIONS["bottom_right_inset"],
        template="faircareai",
        height=500,
        margin=dict(l=80, r=40, t=100, b=100),  # Proper margins for title and source
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    add_source_annotation(fig, source_note)
    return fig


def create_sample_size_waterfall(
    metrics_df: pl.DataFrame,
    title: str = "Sample Size Distribution",
) -> go.Figure:
    """
    Create waterfall/bar chart showing sample sizes with ghosting indication.
    """
    df = metrics_df.filter(pl.col("group") != "_overall").to_pandas()
    df = df.sort_values("n", ascending=False)

    ghost_cfg = GHOSTING_CONFIG

    colors = []
    opacities = []

    for _, row in df.iterrows():
        n = int(row["n"])
        opacity = ghost_cfg.get_opacity(n)
        opacities.append(opacity)

        if n >= ghost_cfg.adequate_threshold:
            colors.append(SEMANTIC_COLORS["pass"])
        elif n >= ghost_cfg.moderate_threshold:
            colors.append(SEMANTIC_COLORS["warn_dark"])
        else:
            colors.append(SEMANTIC_COLORS["fail"])

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["group"],
            y=df["n"],
            marker=dict(
                color=colors,
                opacity=opacities,
                line=dict(color="white", width=1),
            ),
            text=[f"{n:,}" for n in df["n"]],
            textposition="outside",
            textfont=dict(size=TYPOGRAPHY["tick_size"], color=SEMANTIC_COLORS["text"]),
            hovertemplate="<b>%{x}</b><br>n = %{y:,}<extra></extra>",
        )
    )

    # Add threshold lines
    fig.add_hline(
        y=ghost_cfg.adequate_threshold,
        line=dict(color=SEMANTIC_COLORS["pass"], width=2, dash="dash"),
        annotation_text=f"Adequate (n≥{ghost_cfg.adequate_threshold})",
        annotation_position="top right",
    )
    fig.add_hline(
        y=ghost_cfg.moderate_threshold,
        line=dict(color=SEMANTIC_COLORS["warn_dark"], width=2, dash="dash"),
        annotation_text=f"Limited (n≥{ghost_cfg.moderate_threshold})",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px;color:#666'>Groups below threshold lines have reduced visual weight</span>",
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["subheading_size"]),
        ),
        xaxis=dict(
            title=dict(text="Group", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickangle=-45,
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        yaxis=dict(
            title=dict(text="Sample Size (n)", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        template="faircareai",
        height=400,
        showlegend=False,
    )

    return fig


def create_equity_dashboard(
    metrics_df: pl.DataFrame,
    disparity_df: pl.DataFrame | None = None,
    metric: str = "tpr",
) -> go.Figure:
    """
    Create comprehensive equity dashboard with multiple views.

    Layout (per CHAI spec):
    - Top left: Subgroup performance (forest plot)
    - Top right: Sample size distribution
    - Bottom left: Fairness metrics radar
    - Bottom right: Disparity from reference
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"{metric.upper()} by Group",
            "Sample Size Distribution",
            "Fairness Metrics Radar",
            "Disparity from Reference",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatterpolar"}, {"type": "bar"}],
        ],
        **SUBPLOT_SPACING["default"],
    )

    # Forest plot data
    df = metrics_df.filter(pl.col("group") != "_overall").to_pandas()
    df = df.sort_values(metric, ascending=True)

    for _, row in df.iterrows():
        group = row["group"]
        value = row[metric]
        n = int(row["n"])
        opacity = GHOSTING_CONFIG.get_opacity(n)
        color = _get_status_color(value)

        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[group],
                mode="markers",
                marker=dict(size=14, color=color, opacity=opacity),
                showlegend=False,
                hovertemplate=f"<b>{group}</b><br>{metric}: {value:.1%}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Sample size bars
    colors = [
        SEMANTIC_COLORS["pass"]
        if n >= 50
        else SEMANTIC_COLORS["warn_dark"]
        if n >= 30
        else SEMANTIC_COLORS["fail"]
        for n in df["n"]
    ]

    fig.add_trace(
        go.Bar(
            x=df["group"],
            y=df["n"],
            marker_color=colors,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Radar chart for fairness metrics (per CHAI spec)
    for i, (_, row) in enumerate(df.iterrows()):
        group = row["group"]
        tpr = row.get("tpr", 0)
        fpr = row.get("fpr", 0)
        ppv = row.get("ppv", 0)

        fig.add_trace(
            go.Scatterpolar(
                r=[tpr, fpr, ppv, tpr],  # Close the polygon
                theta=["TPR", "FPR", "PPV", "TPR"],
                fill="toself",
                name=group,
                line=dict(color=GROUP_COLORS[i % len(GROUP_COLORS)]),
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

    # Disparity bars (if reference available)
    if len(df) > 1:
        # Use first group as reference for disparity calc
        ref_value = df[metric].iloc[-1]  # Highest value as reference
        disparities = df[metric] - ref_value

        bar_colors = [
            SEMANTIC_COLORS["fail"]
            if abs(d) > 0.1
            else SEMANTIC_COLORS["warn_dark"]
            if abs(d) > 0.05
            else SEMANTIC_COLORS["pass"]
            for d in disparities
        ]

        fig.add_trace(
            go.Bar(
                x=df["group"],
                y=disparities,
                marker_color=bar_colors,
                showlegend=False,
                text=[f"{d:+.1%}" for d in disparities],
                textposition="outside",
            ),
            row=2,
            col=2,
        )

        # Add threshold lines using shapes (avoids Plotly bug with polar subplots)
        fig.add_shape(
            type="line",
            y0=0.1,
            y1=0.1,
            x0=0,
            x1=1,
            line=dict(dash="dash", color=SEMANTIC_COLORS["fail"]),
            xref="x4 domain",
            yref="y4",
        )
        fig.add_shape(
            type="line",
            y0=-0.1,
            y1=-0.1,
            x0=0,
            x1=1,
            line=dict(dash="dash", color=SEMANTIC_COLORS["fail"]),
            xref="x4 domain",
            yref="y4",
        )

    fig.update_layout(
        title=dict(
            text="<b>Equity Audit Dashboard</b>",
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["heading_size"]),
        ),
        template="faircareai",
        height=800,
        showlegend=True,
        legend=LEGEND_POSITIONS["bottom_horizontal"],
    )

    # Update polar subplot with JAMA-style font sizing
    fig.update_polars(
        radialaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        )
    )

    # Ensure all subplot axes have explicit font sizing
    fig.update_xaxes(tickfont=dict(size=TYPOGRAPHY["tick_size"]))
    fig.update_yaxes(tickfont=dict(size=TYPOGRAPHY["tick_size"]))

    return fig


def create_subgroup_heatmap(
    metrics_df: pl.DataFrame,
    metric: str = "tpr",
    title: str | None = None,
) -> go.Figure:
    """
    Create heatmap of metric across all subgroups.

    Per CHAI spec - shows metric values with color intensity.
    """
    if title is None:
        title = f"Subgroup {metric.upper()} Heatmap"

    # Validate required columns exist
    required_cols = ["group", "attribute", metric]
    missing_cols = [col for col in required_cols if col not in metrics_df.columns]
    if missing_cols:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required columns: {', '.join(missing_cols)}",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["fail"]),
        )
        return fig

    df = metrics_df.filter(pl.col("group") != "_overall").to_pandas()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        return fig

    # Get unique attributes and groups
    if "attribute" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Missing 'attribute' column in data",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["fail"]),
        )
        return fig

    attributes = df["attribute"].unique().tolist()

    # Build matrix data
    z_data = []
    y_labels = []
    x_labels = []

    for attr in attributes:
        attr_df = df[df["attribute"] == attr]
        groups = attr_df["group"].tolist()
        values = attr_df[metric].tolist()

        if not x_labels:
            x_labels = groups

        y_labels.append(attr)
        z_data.append(values)

    # Use RdYlGn colorscale (red=bad, green=good)
    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(
                title=metric.upper(),
                tickformat=".0%",
            ),
            hovertemplate=(
                f"<b>%{{y}}</b> - %{{x}}<br>{metric.upper()}: %{{z:.1%}}<extra></extra>"
            ),
        )
    )

    # Add text annotations
    for i, row in enumerate(z_data):
        for j, val in enumerate(row):
            fig.add_annotation(
                x=x_labels[j],
                y=y_labels[i],
                text=f"{val:.0%}",
                showarrow=False,
                font=dict(
                    size=TYPOGRAPHY["annotation_size"],
                    color="white" if val < 0.5 or val > 0.85 else SEMANTIC_COLORS["text"],
                ),
            )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["subheading_size"]),
        ),
        xaxis=dict(
            title=dict(text="Group", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickangle=-45,
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        yaxis=dict(
            title=dict(text="Attribute", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        template="faircareai",
        height=calculate_chart_height(len(attributes), "heatmap"),
    )

    return fig


def create_fairness_radar(
    metrics_df: pl.DataFrame,
    title: str = "Fairness Metrics by Group",
) -> go.Figure:
    """
    Create radar/spider chart showing multiple fairness metrics per group.

    Per CHAI spec visualization requirements.
    """
    df = metrics_df.filter(pl.col("group") != "_overall").to_pandas()

    fig = go.Figure()

    metrics = ["tpr", "fpr", "ppv", "npv", "accuracy"]
    metric_labels = ["TPR", "FPR", "PPV", "NPV", "Accuracy"]

    # Filter to available metrics
    available = [m for m in metrics if m in df.columns]
    labels = [metric_labels[metrics.index(m)] for m in available]

    for i, (_, row) in enumerate(df.iterrows()):
        group = row["group"]
        values = [row[m] for m in available]
        values.append(values[0])  # Close the polygon

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels + [labels[0]],
                fill="toself",
                name=group,
                line=dict(color=GROUP_COLORS[i % len(GROUP_COLORS)], width=2),
                opacity=0.7,
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["subheading_size"]),
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".0%",
                tickfont=dict(size=TYPOGRAPHY["tick_size"]),  # JAMA-style clear ticks
            ),
            angularaxis=dict(
                tickfont=dict(size=TYPOGRAPHY["tick_size"]),  # JAMA-style clear labels
            ),
        ),
        legend=LEGEND_POSITIONS["bottom_horizontal"],
        template="faircareai",
        height=500,
    )

    return fig
