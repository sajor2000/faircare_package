"""
FairCareAI Plotly Visualization Components

Interactive charts with publication-ready editorial aesthetic and ghosting support.
Design Philosophy: Clean, direct-labeled charts that tell a story.

WCAG 2.1 AA Compliance:
- All charts support alt text generation for screen readers
- Color choices from colorblind-safe Okabe-Ito palette
- Sufficient contrast ratios for text and data elements

Van Calster et al. (2025) Alignment:
- RECOMMENDED metrics shown by default (AUROC, calibration plot)
- OPTIONAL metrics (Sens, Spec, PPV, NPV) require include_optional=True
- CAUTION metrics (Accuracy, F1) require explicit show_caution=True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from sklearn.metrics import auc, roc_curve

if TYPE_CHECKING:
    pass

from faircareai.core.config import (
    OutputPersona,
    get_axis_labels,
    get_label,
)

from .themes import (
    COLORSCALES,
    GHOSTING_CONFIG,
    GROUP_COLORS,
    LEGEND_POSITIONS,
    SEMANTIC_COLORS,
    SUBPLOT_SPACING,
    TYPOGRAPHY,
    GhostingConfig,
    calculate_chart_height,
    get_contrast_text_color,
    register_plotly_template,
)
from .utils import add_source_annotation, get_metric_category
from .validation import create_error_figure, validate_required_columns

register_plotly_template()


# =============================================================================
# METRIC DISPLAY LABELS
# =============================================================================

# Module-level constant for metric labels (avoids repeated dictionary creation)
METRIC_LABELS: dict[str, str] = {
    "tpr": "True Positive Rate (TPR / Sensitivity)",
    "tnr": "True Negative Rate (TNR / Specificity)",
    "ppv": "Positive Predictive Value (PPV / Precision)",
    "npv": "Negative Predictive Value (NPV)",
    "fpr": "False Positive Rate (FPR)",
    "fnr": "False Negative Rate (FNR)",
    "auroc": "Area Under ROC Curve (AUROC)",
    "auprc": "Area Under Precision-Recall Curve (AUPRC)",
    "brier": "Brier Score",
    "accuracy": "Accuracy",
    "f1": "F1 Score",
    "mcc": "Matthews Correlation Coefficient (MCC)",
    "prevalence": "Prevalence",
    "sensitivity": "Sensitivity (TPR)",
    "specificity": "Specificity (TNR)",
    "precision": "Precision (PPV)",
}


def get_metric_label(metric: str) -> str:
    """Get display label for metric, with fallback to uppercase.

    Args:
        metric: Metric name (e.g., "tpr", "auroc")

    Returns:
        Human-readable display label

    Example:
        >>> get_metric_label("tpr")
        'True Positive Rate (TPR / Sensitivity)'
        >>> get_metric_label("custom_metric")
        'CUSTOM_METRIC'
    """
    return METRIC_LABELS.get(metric.lower(), metric.upper())


# =============================================================================
# VAN CALSTER METRIC FILTERING
# =============================================================================


def _should_show_metric(
    metric: str,
    include_optional: bool = False,
    include_caution: bool = False,
) -> bool:
    """Check if metric should be shown based on Van Calster classification.

    Args:
        metric: Metric name.
        include_optional: Whether to include OPTIONAL metrics.
        include_caution: Whether to include CAUTION metrics (use sparingly).

    Returns:
        True if metric should be displayed.
    """
    category = get_metric_category(metric)
    if category == "RECOMMENDED":
        return True
    if category == "OPTIONAL":
        return include_optional
    if category == "CAUTION":
        return include_caution
    # Unknown metrics shown if include_optional (Data Scientist mode)
    return include_optional


def _filter_metrics(
    metrics: list[str],
    include_optional: bool = False,
    include_caution: bool = False,
) -> list[str]:
    """Filter metrics based on Van Calster classification.

    Args:
        metrics: List of metric names.
        include_optional: Whether to include OPTIONAL metrics.
        include_caution: Whether to include CAUTION metrics.

    Returns:
        Filtered list of metrics.
    """
    return [m for m in metrics if _should_show_metric(m, include_optional, include_caution)]


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

    max_disparity_value = df["difference"].abs().max()
    max_disparity = (
        float(max_disparity_value)
        if isinstance(max_disparity_value, int | float | np.floating)
        else 0.0
    )
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
    include_optional: bool = True,
) -> go.Figure:
    """Create publication-ready forest plot with CI whiskers and ghosting.

    Van Calster et al. (2025) Classification:
    - Default metrics (tpr, fpr, ppv) are OPTIONAL
    - AUROC is RECOMMENDED - use create_roc_curve_by_group for AUROC display

    Features:
    - Horizontal lollipop/forest plot style
    - Confidence interval whiskers
    - Green "safe zone" shading for acceptable range
    - Ghosting (opacity reduction) for low sample sizes
    - Direct labeling (no legend needed)

    Args:
        metrics_df: DataFrame with columns [group, metric, n, ci_lower, ci_upper].
        metric: Metric to display (default "tpr" - OPTIONAL metric).
        title: Chart title (auto-generated if None).
        subtitle: Optional subtitle.
        enable_ghosting: Whether to reduce opacity for small samples.
        ghosting_config: Custom ghosting configuration.
        show_safe_zone: Whether to show acceptable range shading.
        safe_zone_min: Minimum value for safe zone (default 0.8).
        reference_line: Optional reference line value.
        source_note: Custom source annotation.
        include_optional: If True, displays the plot. If False and metric is
            OPTIONAL/CAUTION, returns placeholder. Default True for backward
            compatibility (forest plots are commonly used for OPTIONAL metrics).

    Returns:
        Plotly Figure object.

    Note:
        Forest plots typically display OPTIONAL classification metrics.
        For RECOMMENDED metrics, consider using specialized plots like
        calibration curves or decision curves.
    """
    # Check if metric should be shown based on Van Calster classification
    if not include_optional and not _should_show_metric(metric, include_optional=False):
        fig = go.Figure()
        category = get_metric_category(metric)
        fig.add_annotation(
            text=f"Forest plot for '{metric}' requires include_optional=True<br>"
            f"(Van Calster classification: {category})",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        fig.update_layout(
            title=dict(
                text=f"<b>{title or metric.upper()}</b>",
                font=dict(size=TYPOGRAPHY["subheading_size"]),
            ),
            template="faircareai",
            height=300,
        )
        return fig
    ghost_cfg = ghosting_config or GHOSTING_CONFIG

    # Validate required columns exist
    required_cols = ["group", metric, "n"]
    missing_cols = validate_required_columns(metrics_df, required_cols)
    if missing_cols:
        return create_error_figure(
            f"Missing required columns: {', '.join(missing_cols)}",
            title=title or metric.upper(),
        )

    df = metrics_df.filter(pl.col("group") != "_overall")

    if df.height == 0:
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
        title = get_metric_label(metric)

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

    df = df.sort(metric)

    y_labels = []
    for row in df.iter_rows(named=True):
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
        margin=dict(l=220, r=100, t=100, b=140),  # Wide left for long names
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
    missing_cols = validate_required_columns(disparity_df, required_cols)
    if missing_cols:
        return create_error_figure(
            f"Missing required columns: {', '.join(missing_cols)}",
            title=title,
        )

    # Filter for the specified metric
    df = disparity_df.filter(pl.col("metric") == metric)

    if df.height == 0:
        fig = go.Figure()
        fig.add_annotation(text="No disparity data available", x=0.5, y=0.5, showarrow=False)
        return fig

    # Get unique groups
    groups = sorted(set(df["reference_group"].to_list() + df["comparison_group"].to_list()))
    n_groups = len(groups)

    # Create O(1) lookup dictionary for group indices
    group_to_idx = {g: i for i, g in enumerate(groups)}

    # Create matrix
    matrix = np.zeros((n_groups, n_groups))
    sig_matrix = np.zeros((n_groups, n_groups))

    for row in df.iter_rows(named=True):
        ref_group = row["reference_group"]
        comp_group = row["comparison_group"]

        # Validate groups exist in lookup dict
        if ref_group not in group_to_idx or comp_group not in group_to_idx:
            continue

        ref_idx = group_to_idx[ref_group]
        comp_idx = group_to_idx[comp_group]
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
            tickangle=-40,
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text="Reference Group", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            autorange="reversed",
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        annotations=annotations,
        template="faircareai",
        height=calculate_chart_height(n_groups, "bar"),
        margin=dict(l=120, r=80, t=120, b=160),  # Bottom margin for rotated labels
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    add_source_annotation(fig, source_note)
    return fig


def create_metric_comparison_chart(
    metrics_df: pl.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Performance Metrics by Group",
    include_optional: bool = False,
    persona: OutputPersona = OutputPersona.DATA_SCIENTIST,
) -> go.Figure:
    """Create grouped bar chart comparing multiple metrics across groups.

    Van Calster et al. (2025) Classification:
    - OPTIONAL: sensitivity (tpr), specificity, ppv, npv
    - CAUTION: accuracy (not shown by default)

    Args:
        metrics_df: DataFrame with metrics data per group.
        metrics: List of metrics to display. If None, uses defaults based on include_optional.
        title: Chart title.
        include_optional: If True, shows OPTIONAL metrics (tpr, fpr, ppv).
            If False, shows only RECOMMENDED metrics (none for this chart type).
            Default False for Governance persona compatibility.
        persona: OutputPersona for label terminology (default DATA_SCIENTIST).

    Returns:
        Plotly Figure object.

    Note:
        This chart displays classification metrics which are OPTIONAL per
        Van Calster et al. (2025). Consider using calibration plots and
        AUROC for primary reporting.
    """
    # Default metrics based on include_optional flag
    if metrics is None:
        if include_optional:
            # Data Scientist mode: show OPTIONAL classification metrics
            metrics = ["tpr", "fpr", "ppv"]
        else:
            # Governance mode: no default metrics (this chart type is OPTIONAL)
            # Return informative placeholder
            fig = go.Figure()
            fig.add_annotation(
                text="Classification metrics require include_optional=True<br>"
                "(Van Calster: sensitivity/specificity are OPTIONAL metrics)",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
            )
            fig.update_layout(
                title=dict(text=f"<b>{title}</b>", font=dict(size=TYPOGRAPHY["subheading_size"])),
                template="faircareai",
                height=300,
            )
            return fig

    # Validate required columns exist
    missing_cols = validate_required_columns(metrics_df, ["group"])
    if missing_cols:
        return create_error_figure(
            f"Missing required column: {', '.join(missing_cols)}",
            title=title,
        )

    df = metrics_df.filter(pl.col("group") != "_overall")

    if df.height == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        return fig

    groups = df["group"].to_list()

    fig = go.Figure()

    metric_colors = {
        "tpr": SEMANTIC_COLORS["primary"],
        "fpr": SEMANTIC_COLORS["fail"],
        "ppv": SEMANTIC_COLORS["secondary"],
        "npv": GROUP_COLORS[4],
        "accuracy": GROUP_COLORS[5],
    }

    # Persona-aware metric labels
    def get_metric_display_label(m: str) -> str:
        """Get persona-appropriate label for a metric."""
        # Map internal metric names to terminology keys
        metric_key_map = {
            "tpr": "sensitivity",
            "fpr": "specificity",  # FPR will use fallback
            "ppv": "ppv",
            "npv": "npv",
            "accuracy": "accuracy",
        }
        key = metric_key_map.get(m, m)
        label = get_label(key, persona, "name")
        # If label is same as key, it wasn't found - use fallback
        if label == key:
            fallback_labels = {
                "tpr": "Sensitivity (TPR)"
                if persona == OutputPersona.DATA_SCIENTIST
                else "Detection Rate",
                "fpr": "False Positive Rate"
                if persona == OutputPersona.DATA_SCIENTIST
                else "False Alarm Rate",
                "ppv": "PPV"
                if persona == OutputPersona.DATA_SCIENTIST
                else "Positive Predictive Value",
                "npv": "NPV"
                if persona == OutputPersona.DATA_SCIENTIST
                else "Negative Predictive Value",
                "accuracy": "Accuracy"
                if persona == OutputPersona.DATA_SCIENTIST
                else "Correct Predictions",
            }
            return fallback_labels.get(m, m.upper())
        return label

    for metric in metrics:
        if metric in df.columns:
            display_label = get_metric_display_label(metric)
            bar_color = metric_colors.get(metric, GROUP_COLORS[0])
            metric_values = df[metric].to_list()
            fig.add_trace(
                go.Bar(
                    name=display_label,
                    x=groups,
                    y=metric_values,
                    marker_color=bar_color,
                    text=[f"{v:.0%}" for v in metric_values],
                    textposition="inside",
                    textfont=dict(color=get_contrast_text_color(bar_color), size=TYPOGRAPHY["tick_size"]),
                    hovertemplate=(f"<b>%{{x}}</b><br>{display_label}: %{{y:.1%}}<extra></extra>"),
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
        margin=dict(l=100, r=50, t=100, b=140),
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
    """Create executive summary scorecard with publication-ready editorial styling."""
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
        margin=dict(l=80, r=40, t=80, b=80),
    )

    return fig


def create_calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_labels: np.ndarray | None = None,
    n_bins: int = 10,
    title: str | None = None,
    source_note: str | None = None,
    persona: OutputPersona = OutputPersona.DATA_SCIENTIST,
) -> go.Figure:
    """Create calibration plot showing predicted vs actual probabilities.

    Van Calster et al. (2025) Classification: RECOMMENDED
    - Calibration plots are essential for all reports
    - Always shown regardless of include_optional setting

    Args:
        y_true: True binary outcomes (0 or 1).
        y_prob: Predicted probabilities.
        group_labels: Optional array of group labels for stratified curves.
        n_bins: Number of calibration bins (default 10).
        title: Chart title. Uses persona-appropriate default if None.
        source_note: Custom source annotation.
        persona: OutputPersona for label terminology (default DATA_SCIENTIST).

    Returns:
        Plotly Figure object.

    Note:
        This is a RECOMMENDED metric per Van Calster et al. (2025).
        If group_labels provided, shows separate curves per group.
    """
    # Get persona-appropriate labels
    if title is None:
        title = get_label("calibration", persona, "name")
    x_label, y_label = get_axis_labels("calibration", persona)
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

    def compute_calibration(
        y_t: np.ndarray,
        y_p: np.ndarray,
        bins: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # Persona-appropriate subtitle
    subtitle = (
        "Closer to diagonal = better calibrated"
        if persona == OutputPersona.DATA_SCIENTIST
        else "Points on the line mean predictions match reality"
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:{TYPOGRAPHY['body_size']}px;color:#666'>{subtitle}</span>",
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["subheading_size"]),
        ),
        xaxis=dict(
            title=dict(
                text=x_label or "Mean Predicted Probability",
                font=dict(size=TYPOGRAPHY["axis_title_size"]),
            ),
            tickformat=".0%",
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
            range=[0, 1],
        ),
        yaxis=dict(
            title=dict(
                text=y_label or "Fraction of Positives",
                font=dict(size=TYPOGRAPHY["axis_title_size"]),
            ),
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
    title: str | None = None,
    source_note: str | None = None,
    persona: OutputPersona = OutputPersona.DATA_SCIENTIST,
) -> go.Figure:
    """Create ROC curves for each demographic group.

    Van Calster et al. (2025) Classification: RECOMMENDED
    - AUROC is the key discrimination measure
    - Always shown regardless of include_optional setting

    Args:
        y_true: True binary outcomes (0 or 1).
        y_prob: Predicted probabilities.
        group_labels: Array of group labels for stratified curves.
        title: Chart title. Uses persona-appropriate default if None.
        source_note: Custom source annotation.
        persona: OutputPersona for label terminology (default DATA_SCIENTIST).

    Returns:
        Plotly Figure object with ROC curves per group.

    Note:
        This is a RECOMMENDED metric per Van Calster et al. (2025).
        Each curve shows AUC in the legend for easy comparison.
    """
    # Get persona-appropriate labels
    if title is None:
        base_title = get_label("auroc", persona, "name")
        title = f"{base_title} by Demographic Group"
    x_label, y_label = get_axis_labels("auroc", persona)
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
            title=dict(
                text=x_label or "False Positive Rate",
                font=dict(size=TYPOGRAPHY["axis_title_size"]),
            ),
            tickformat=".0%",
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
            range=[0, 1],
        ),
        yaxis=dict(
            title=dict(
                text=y_label or "True Positive Rate",
                font=dict(size=TYPOGRAPHY["axis_title_size"]),
            ),
            tickformat=".0%",
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
            range=[0, 1],
        ),
        legend=LEGEND_POSITIONS["bottom_right_inset"],
        template="faircareai",
        height=500,
        margin=dict(l=80, r=100, t=100, b=120),  # Extra bottom/right margin for inset legend
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
    df = metrics_df.filter(pl.col("group") != "_overall").sort("n", descending=True)

    ghost_cfg = GHOSTING_CONFIG

    colors = []
    opacities = []

    for row in df.iter_rows(named=True):
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

    groups = df["group"].to_list()
    n_values = df["n"].to_list()
    fig.add_trace(
        go.Bar(
            x=groups,
            y=n_values,
            marker=dict(
                color=colors,
                opacity=opacities,
                line=dict(color="white", width=1),
            ),
            text=[f"{n:,}" for n in n_values],
            textposition="inside",
            textfont=dict(color=[get_contrast_text_color(c) for c in colors], size=TYPOGRAPHY["tick_size"]),
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
            text=f"<b>{title}</b><br><span style='font-size:14px;color:#666'>Groups below threshold lines have reduced visual weight</span>",
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["subheading_size"]),
        ),
        xaxis=dict(
            title=dict(text="Group", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickangle=-40,
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text="Sample Size (n)", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        ),
        template="faircareai",
        height=400,
        margin=dict(l=80, r=40, t=100, b=160),  # Extra bottom margin for rotated labels
        showlegend=False,
    )

    return fig


def create_equity_dashboard(
    metrics_df: pl.DataFrame,
    disparity_df: pl.DataFrame | None = None,
    metric: str = "tpr",
    include_optional: bool = False,
) -> go.Figure:
    """Create comprehensive equity dashboard with multiple views.

    Van Calster et al. (2025) Classification:
    - This dashboard displays OPTIONAL metrics (sensitivity, PPV, etc.)
    - Requires include_optional=True for full display
    - Default mode shows sample sizes and basic structure only

    Layout (per CHAI spec):
    - Top left: Subgroup performance (forest plot)
    - Top right: Sample size distribution
    - Bottom left: Fairness metrics radar
    - Bottom right: Disparity from reference

    Args:
        metrics_df: DataFrame with metrics data per group.
        disparity_df: Optional DataFrame with disparity calculations.
        metric: Primary metric to display (default "tpr").
        include_optional: If True, shows full dashboard with OPTIONAL metrics.
            If False, shows reduced dashboard with sample size info only.
            Default False for Governance persona compatibility.

    Returns:
        Plotly Figure object.
    """
    if not include_optional:
        # Governance mode: show informative placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="Equity dashboard requires include_optional=True<br>"
            "(Contains OPTIONAL metrics: TPR, FPR, PPV per Van Calster et al. 2025)",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        fig.update_layout(
            title=dict(
                text="<b>Equity Audit Dashboard</b><br>"
                "<span style='font-size:12px;color:#666'>Enable include_optional=True for full view</span>",
                font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["heading_size"]),
            ),
            template="faircareai",
            height=400,
        )
        return fig
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
    df = metrics_df.filter(pl.col("group") != "_overall").sort(metric)

    for row in df.iter_rows(named=True):
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
    n_values = df["n"].to_list()
    colors = [
        SEMANTIC_COLORS["pass"]
        if n >= 50
        else SEMANTIC_COLORS["warn_dark"]
        if n >= 30
        else SEMANTIC_COLORS["fail"]
        for n in n_values
    ]

    groups = df["group"].to_list()
    fig.add_trace(
        go.Bar(
            x=groups,
            y=n_values,
            marker_color=colors,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Radar chart for fairness metrics (per CHAI spec)
    for i, row in enumerate(df.iter_rows(named=True)):
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
    if df.height > 1:
        # Use first group as reference for disparity calc
        metric_values = df[metric].to_list()
        ref_value = metric_values[-1]  # Highest value as reference
        disparities = [v - ref_value for v in metric_values]

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
                x=groups,
                y=disparities,
                marker_color=bar_colors,
                showlegend=False,
                text=[f"{d:+.1%}" for d in disparities],
                textposition="inside",
                textfont=dict(color=[get_contrast_text_color(c) for c in bar_colors], size=10),
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

    # Update polar subplot with publication-ready font sizing
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
    include_optional: bool = True,
) -> go.Figure:
    """Create heatmap of metric across all subgroups.

    Van Calster et al. (2025) Classification:
    - Default metric (tpr) is OPTIONAL
    - Shows metric values with color intensity per CHAI spec

    Args:
        metrics_df: DataFrame with columns [group, attribute, metric].
        metric: Metric to display (default "tpr" - OPTIONAL metric).
        title: Chart title (auto-generated if None).
        include_optional: If True, displays the heatmap. If False and metric is
            OPTIONAL/CAUTION, returns placeholder. Default True for backward
            compatibility.

    Returns:
        Plotly Figure object.
    """
    if title is None:
        title = f"Subgroup {metric.upper()} Heatmap"

    # Check if metric should be shown based on Van Calster classification
    if not include_optional and not _should_show_metric(metric, include_optional=False):
        fig = go.Figure()
        category = get_metric_category(metric)
        fig.add_annotation(
            text=f"Subgroup heatmap for '{metric}' requires include_optional=True<br>"
            f"(Van Calster classification: {category})",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=TYPOGRAPHY["subheading_size"])),
            template="faircareai",
            height=300,
        )
        return fig

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

    df = metrics_df.filter(pl.col("group") != "_overall")

    if df.height == 0:
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

    attributes = df["attribute"].unique().to_list()

    # Build matrix data
    z_data: list[list[float]] = []
    y_labels: list[str] = []
    x_labels: list[str] = []

    for attr in attributes:
        attr_df = df.filter(pl.col("attribute") == attr)
        groups = attr_df["group"].to_list()
        values = attr_df[metric].to_list()

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

    # Add text annotations with WCAG-compliant contrast
    # RdYlGn colorscale: red (0) → yellow (0.5) → green (1)
    # Red and green are dark (need white text), yellow is light (needs dark text)
    for i, row in enumerate(z_data):
        for j, val in enumerate(row):
            # Use dark text for yellow range (0.3-0.7), white text for red/green
            text_color = SEMANTIC_COLORS["text"] if 0.3 <= val <= 0.7 else "white"
            fig.add_annotation(
                x=x_labels[j],
                y=y_labels[i],
                text=f"{val:.0%}",
                showarrow=False,
                font=dict(
                    size=TYPOGRAPHY["annotation_size"],
                    color=text_color,
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
            automargin=True,
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
    include_optional: bool = False,
    persona: OutputPersona = OutputPersona.DATA_SCIENTIST,
) -> go.Figure:
    """Create radar/spider chart showing multiple fairness metrics per group.

    Van Calster et al. (2025) Classification:
    - OPTIONAL: sensitivity (tpr), specificity, ppv, npv
    - CAUTION: accuracy (only shown if explicitly requested)

    Args:
        metrics_df: DataFrame with metrics data per group.
        title: Chart title.
        include_optional: If True, shows OPTIONAL metrics (tpr, fpr, ppv, npv).
            If False, returns placeholder indicating no RECOMMENDED metrics available.
            Default False for Governance persona compatibility.
        persona: OutputPersona for label terminology (default DATA_SCIENTIST).

    Returns:
        Plotly Figure object.

    Note:
        This chart displays classification metrics which are OPTIONAL per
        Van Calster et al. (2025). Consider using calibration plots and
        AUROC for primary reporting.
    """
    if not include_optional:
        # Governance mode: radar chart shows OPTIONAL/CAUTION metrics only
        fig = go.Figure()
        fig.add_annotation(
            text="Fairness radar requires include_optional=True<br>"
            "(Van Calster: TPR/FPR/PPV/NPV are OPTIONAL metrics)",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=TYPOGRAPHY["subheading_size"])),
            template="faircareai",
            height=300,
        )
        return fig

    df = metrics_df.filter(pl.col("group") != "_overall")

    fig = go.Figure()

    # OPTIONAL metrics only (exclude CAUTION metrics like accuracy by default)
    metrics = ["tpr", "fpr", "ppv", "npv"]

    # Persona-aware metric labels
    if persona == OutputPersona.GOVERNANCE:
        metric_labels = [
            "Detection Rate",
            "False Alarm Rate",
            "Positive Predictive Value",
            "Negative Predictive Value",
        ]
    else:
        metric_labels = ["Sensitivity", "FPR", "PPV", "NPV"]

    # Filter to available metrics
    available = [m for m in metrics if m in df.columns]
    labels = [metric_labels[metrics.index(m)] for m in available]

    for i, row in enumerate(df.iter_rows(named=True)):
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
                tickfont=dict(size=TYPOGRAPHY["tick_size"]),  # Clear ticks
            ),
            angularaxis=dict(
                tickfont=dict(size=TYPOGRAPHY["tick_size"]),  # Clear labels
            ),
        ),
        legend=LEGEND_POSITIONS["bottom_horizontal"],
        template="faircareai",
        height=500,
    )

    return fig
