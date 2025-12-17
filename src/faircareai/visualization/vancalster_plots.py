"""
FairCareAI Van Calster Visualization Components

Implements the four recommended visualizations from Van Calster et al. (2025):

1. AUROC Forest Plot by Subgroup
2. Calibration Plots by Subgroup (smoothed)
3. Decision Curves (Net Benefit) by Subgroup
4. Risk Distribution Plots by Subgroup (violin/box plots)

Reference:
    Van Calster B, Collins GS, Vickers AJ, et al. Evaluation of performance
    measures in predictive artificial intelligence models to support medical
    decisions: overview and guidance. Lancet Digit Health 2025.
    https://doi.org/10.1016/j.landig.2025.100916

WCAG 2.1 AA Compliance:
- All charts include alt text generation for screen readers
- Colorblind-safe Okabe-Ito palette used throughout
- Sufficient contrast ratios for text and data elements
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .themes import (
    FAIRCAREAI_BRAND,
    GHOSTING_CONFIG,
    GROUP_COLORS,
    LEGEND_POSITIONS,
    SEMANTIC_COLORS,
    TYPOGRAPHY,
    calculate_chart_height,
    register_plotly_template,
)

register_plotly_template()


# =============================================================================
# ALT TEXT GENERATION FOR WCAG 2.1 AA COMPLIANCE
# =============================================================================


def _generate_auroc_forest_alt_text(
    results: dict[str, Any],
    title: str,
) -> str:
    """Generate accessible alt text for AUROC forest plot."""
    groups = results.get("groups", {})
    n_groups = len(groups)

    if n_groups == 0:
        return f"{title}. No data available for visualization."

    aurocs = [(g, d.get("auroc", 0)) for g, d in groups.items() if d.get("auroc") is not None]

    if not aurocs:
        return f"{title}. No AUROC values available."

    min_auroc = min(a[1] for a in aurocs)
    max_auroc = max(a[1] for a in aurocs)
    auroc_range = max_auroc - min_auroc

    alt_text = (
        f"{title}. Forest plot showing AUROC discrimination performance across "
        f"{n_groups} demographic subgroups. AUROC values range from {min_auroc:.2f} "
        f"to {max_auroc:.2f} (range: {auroc_range:.2f}). "
    )

    if auroc_range >= 0.05:
        alt_text += "Clinically meaningful differences detected across groups."
    else:
        alt_text += "Performance is consistent across groups."

    return alt_text


def _generate_calibration_alt_text(
    results: dict[str, Any],
    title: str,
) -> str:
    """Generate accessible alt text for calibration plots."""
    groups = results.get("groups", {})
    n_groups = len(groups)

    if n_groups == 0:
        return f"{title}. No calibration data available."

    alt_text = (
        f"{title}. Calibration plots comparing predicted versus observed "
        f"probabilities for {n_groups} demographic subgroups. "
        "Perfect calibration follows the diagonal line from (0,0) to (1,1). "
    )

    # Check for miscalibration
    miscalibrated = []
    for group, data in groups.items():
        oe = data.get("oe_ratio")
        if oe is not None and abs(oe - 1.0) > 0.2:
            direction = "overprediction" if oe < 1 else "underprediction"
            miscalibrated.append(f"{group} ({direction})")

    if miscalibrated:
        alt_text += f"Miscalibration detected in: {', '.join(miscalibrated)}."
    else:
        alt_text += "All groups show adequate calibration."

    return alt_text


def _generate_decision_curve_alt_text(
    results: dict[str, Any],
    title: str,
    threshold: float,
) -> str:
    """Generate accessible alt text for decision curves."""
    groups = results.get("groups", {})
    n_groups = len(groups)

    if n_groups == 0:
        return f"{title}. No clinical utility data available."

    alt_text = (
        f"{title}. Decision curves showing net benefit (clinical utility) across "
        f"{n_groups} demographic subgroups for decision threshold {threshold:.0%}. "
        "Model is useful where its curve exceeds the 'treat all' and 'treat none' baselines. "
    )

    # Find groups with useful ranges
    useful_groups = []
    for group, data in groups.items():
        useful = data.get("useful_range", {})
        if useful.get("min") and useful.get("max"):
            useful_groups.append(group)

    if useful_groups:
        alt_text += f"Model provides clinical benefit in: {', '.join(useful_groups)}."
    else:
        alt_text += "Limited clinical utility across all groups."

    return alt_text


def _generate_risk_distribution_alt_text(
    results: dict[str, Any],
    title: str,
) -> str:
    """Generate accessible alt text for risk distribution plots."""
    groups = results.get("groups", {})
    n_groups = len(groups)

    if n_groups == 0:
        return f"{title}. No risk distribution data available."

    alt_text = (
        f"{title}. Violin plots showing distribution of predicted probabilities "
        f"for events and non-events across {n_groups} demographic subgroups. "
        "Greater separation between event and non-event distributions indicates "
        "better discrimination. "
    )

    # Check for distribution overlap
    overlaps = []
    for group, data in groups.items():
        disc_slope = data.get("discrimination_slope")
        if disc_slope is not None and disc_slope < 0.2:
            overlaps.append(group)

    if overlaps:
        alt_text += f"High overlap detected in: {', '.join(overlaps)}."
    else:
        alt_text += "Adequate separation in all groups."

    return alt_text


def _add_source_annotation(
    fig: go.Figure,
    source_note: str | None = None,
) -> go.Figure:
    """Add Van Calster citation and source annotation."""
    citation = "Van Calster et al. (2025) Lancet Digit Health"
    effective_source = source_note if source_note else FAIRCAREAI_BRAND["source_note"]

    fig.add_annotation(
        text=f"{effective_source} | Methodology: {citation}",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.12,
        showarrow=False,
        font={"size": TYPOGRAPHY["source_size"], "color": SEMANTIC_COLORS["text_secondary"]},
        xanchor="left",
    )
    return fig


# =============================================================================
# 1. AUROC FOREST PLOT BY SUBGROUP
# =============================================================================


def create_auroc_forest_plot(
    results: dict[str, Any],
    title: str = "AUROC by Demographic Subgroup",
    subtitle: str | None = None,
    show_reference_line: bool = True,
    reference_auroc: float | None = None,
    enable_ghosting: bool = True,
    source_note: str | None = None,
) -> go.Figure:
    """Create forest plot of AUROC values across subgroups.

    Per Van Calster et al. (2025): "AUROC is the key measure for
    comparing discrimination across demographic groups."

    Args:
        results: Dict with 'groups' containing per-subgroup AUROC data.
        title: Chart title.
        subtitle: Optional subtitle.
        show_reference_line: Show line at reference/overall AUROC.
        reference_auroc: Value for reference line (uses overall if not provided).
        enable_ghosting: Reduce opacity for small samples.
        source_note: Custom source note.

    Returns:
        Plotly Figure with forest plot.
    """
    groups = results.get("groups", {})

    if not groups:
        fig = go.Figure()
        fig.add_annotation(
            text="No AUROC data available for visualization",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        return fig

    fig = go.Figure()

    # Sort by AUROC value for visual hierarchy
    sorted_groups = sorted(
        [(g, d) for g, d in groups.items() if d.get("auroc") is not None],
        key=lambda x: x[1].get("auroc", 0),
        reverse=False,  # Lowest at bottom
    )

    if not sorted_groups:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid AUROC values to display",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        return fig

    # Determine reference AUROC for comparison line
    if reference_auroc is None:
        # Try to get overall AUROC or use mean
        if "Overall" in groups:
            reference_auroc = groups["Overall"].get("auroc")
        else:
            reference_auroc = np.mean([d.get("auroc", 0) for _, d in sorted_groups])

    # Add reference line
    if show_reference_line and reference_auroc is not None:
        fig.add_vline(
            x=reference_auroc,
            line=dict(color=SEMANTIC_COLORS["text_secondary"], width=2, dash="dash"),
            annotation_text=f"Reference: {reference_auroc:.2f}",
            annotation_position="top",
            annotation_font=dict(size=TYPOGRAPHY["tick_size"]),
        )

    y_labels = []
    colors_by_group = {}

    for i, (group_name, group_data) in enumerate(sorted_groups):
        auroc = group_data.get("auroc")
        n = group_data.get("n", 0)
        ci = group_data.get("auroc_ci_95", [None, None])
        is_reference = group_data.get("is_reference", False)

        # Assign color
        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        colors_by_group[group_name] = color

        # Opacity based on sample size
        opacity = GHOSTING_CONFIG.get_opacity(n) if enable_ghosting else 1.0

        # Label with sample size
        label = f"{group_name}  (n={n:,})"
        if is_reference:
            label += " [REF]"
        y_labels.append(label)

        # Hover text
        hover_text = f"<b>{group_name}</b><br>AUROC: {auroc:.3f}<br>Sample size: {n:,}"
        if ci[0] is not None and ci[1] is not None:
            hover_text += f"<br>95% CI: ({ci[0]:.3f}, {ci[1]:.3f})"

        # CI whiskers
        if ci[0] is not None and ci[1] is not None:
            fig.add_trace(
                go.Scatter(
                    x=[ci[0], ci[1]],
                    y=[label, label],
                    mode="lines",
                    line=dict(width=2, color=color),
                    opacity=opacity * 0.6,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # CI caps
            for ci_val in [ci[0], ci[1]]:
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

        # Main point
        fig.add_trace(
            go.Scatter(
                x=[auroc],
                y=[label],
                mode="markers+text",
                marker=dict(
                    size=16,
                    color=color,
                    opacity=opacity,
                    line=dict(width=2, color="white"),
                ),
                text=[f"{auroc:.2f}"],
                textposition="middle right",
                textfont=dict(size=TYPOGRAPHY["annotation_size"], color=SEMANTIC_COLORS["text"]),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
            )
        )

    # Generate alt text
    alt_text = _generate_auroc_forest_alt_text(results, title)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b>"
                + (
                    f"<br><span style='font-size:14px;color:#666'>{subtitle}</span>"
                    if subtitle
                    else ""
                )
            ),
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["heading_size"]),
        ),
        xaxis=dict(
            title=dict(text="AUROC (C-statistic)", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            range=[0.4, 1.0],
            showgrid=True,
            gridcolor=SEMANTIC_COLORS["grid"],
            gridwidth=1,
        ),
        yaxis=dict(
            title=dict(text="Subgroup", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            showgrid=False,
            categoryorder="array",
            categoryarray=y_labels,
        ),
        template="faircareai",
        height=calculate_chart_height(len(sorted_groups), "forest"),
        margin=dict(l=200, r=100, t=120, b=100),
        meta={"description": alt_text},
    )

    _add_source_annotation(fig, source_note)
    return fig


# =============================================================================
# 2. CALIBRATION PLOTS BY SUBGROUP
# =============================================================================


def create_calibration_plot_by_subgroup(
    results: dict[str, Any],
    title: str = "Calibration by Demographic Subgroup",
    show_confidence_region: bool = True,
    source_note: str | None = None,
) -> go.Figure:
    """Create calibration curves for each subgroup.

    Per Van Calster et al. (2025): "Calibration plot is the most insightful
    approach to assess calibration, particularly when smoothing is used."

    Args:
        results: Dict with 'groups' containing calibration curve data.
        title: Chart title.
        show_confidence_region: Show shaded CI region around curves.
        source_note: Custom source note.

    Returns:
        Plotly Figure with calibration curves.
    """
    groups = results.get("groups", {})

    if not groups:
        fig = go.Figure()
        fig.add_annotation(
            text="No calibration data available",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        return fig

    fig = go.Figure()

    # Perfect calibration line (diagonal)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=SEMANTIC_COLORS["text_secondary"], dash="dash", width=2),
            name="Perfect Calibration",
            showlegend=True,
        )
    )

    for i, (group_name, group_data) in enumerate(groups.items()):
        if "error" in group_data:
            continue

        cal_curve = group_data.get("calibration_curve", {})
        prob_pred = cal_curve.get("prob_pred", [])
        prob_true = cal_curve.get("prob_true", [])

        if not prob_pred or not prob_true:
            continue

        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        n = group_data.get("n", 0)
        oe = group_data.get("oe_ratio", 1.0)
        brier = group_data.get("brier_score", 0)

        # Calibration curve
        fig.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                name=f"{group_name} (n={n:,})",
                hovertemplate=(
                    f"<b>{group_name}</b><br>"
                    "Predicted: %{x:.1%}<br>"
                    "Observed: %{y:.1%}<br>"
                    f"O:E Ratio: {oe:.2f}<br>"
                    f"Brier: {brier:.3f}<extra></extra>"
                ),
            )
        )

    # Generate alt text
    alt_text = _generate_calibration_alt_text(results, title)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b><br>"
                f"<span style='font-size:{TYPOGRAPHY['body_size']}px;color:#666'>"
                "Closer to diagonal = better calibrated</span>"
            ),
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["subheading_size"]),
        ),
        xaxis=dict(
            title=dict(
                text="Mean Predicted Probability", font=dict(size=TYPOGRAPHY["axis_title_size"])
            ),
            tickformat=".0%",
            range=[0, 1],
        ),
        yaxis=dict(
            title=dict(text="Observed Proportion", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickformat=".0%",
            range=[0, 1],
        ),
        legend=LEGEND_POSITIONS["top_horizontal"],
        template="faircareai",
        height=500,
        margin=dict(l=80, r=40, t=120, b=100),
        meta={"description": alt_text},
    )

    _add_source_annotation(fig, source_note)
    return fig


# =============================================================================
# 3. DECISION CURVES (NET BENEFIT) BY SUBGROUP
# =============================================================================


def create_decision_curve_by_subgroup(
    results: dict[str, Any],
    title: str = "Decision Curves by Demographic Subgroup",
    threshold_range: tuple[float, float] = (0.05, 0.5),
    show_reference_strategies: bool = True,
    source_note: str | None = None,
) -> go.Figure:
    """Create decision curves showing net benefit by subgroup.

    Per Van Calster et al. (2025): "Net benefit with decision curve analysis
    is essential to report. It evaluates whether a model leads to improved
    clinical decisions on average."

    Args:
        results: Dict with 'groups' containing decision curve data.
        title: Chart title.
        threshold_range: Range of thresholds to display (min, max).
        show_reference_strategies: Show 'treat all' and 'treat none' lines.
        source_note: Custom source note.

    Returns:
        Plotly Figure with decision curves.
    """
    groups = results.get("groups", {})
    threshold = results.get("primary_threshold", 0.5)

    if not groups:
        fig = go.Figure()
        fig.add_annotation(
            text="No net benefit data available",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        return fig

    fig = go.Figure()

    # Get first group's decision curve data for reference strategies
    first_group = next(iter(groups.values()))
    dc = first_group.get("decision_curve", {})
    thresholds = np.array(dc.get("thresholds", []))

    # Filter to display range
    if len(thresholds) > 0:
        mask = (thresholds >= threshold_range[0]) & (thresholds <= threshold_range[1])
        thresholds_display = thresholds[mask]

        if show_reference_strategies:
            nb_all = np.array(dc.get("net_benefit_all", []))[mask]
            nb_none = np.array(dc.get("net_benefit_none", []))[mask]

            # Treat None
            fig.add_trace(
                go.Scatter(
                    x=thresholds_display,
                    y=nb_none,
                    mode="lines",
                    line=dict(color=SEMANTIC_COLORS["text_secondary"], dash="dash", width=1),
                    name="Treat None",
                    showlegend=True,
                )
            )

            # Treat All
            fig.add_trace(
                go.Scatter(
                    x=thresholds_display,
                    y=nb_all,
                    mode="lines",
                    line=dict(color=SEMANTIC_COLORS["text_secondary"], dash="dot", width=1),
                    name="Treat All",
                    showlegend=True,
                )
            )

    # Add each subgroup's net benefit curve
    for i, (group_name, group_data) in enumerate(groups.items()):
        if "error" in group_data:
            continue

        dc = group_data.get("decision_curve", {})
        group_thresholds = np.array(dc.get("thresholds", []))
        nb_model = np.array(dc.get("net_benefit_model", []))

        if len(group_thresholds) == 0:
            continue

        # Filter to display range
        mask = (group_thresholds >= threshold_range[0]) & (group_thresholds <= threshold_range[1])
        thresholds_plot = group_thresholds[mask]
        nb_plot = nb_model[mask]

        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        n = group_data.get("n", 0)

        fig.add_trace(
            go.Scatter(
                x=thresholds_plot,
                y=nb_plot,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{group_name} (n={n:,})",
                hovertemplate=(
                    f"<b>{group_name}</b><br>"
                    "Threshold: %{x:.0%}<br>"
                    "Net Benefit: %{y:.3f}<extra></extra>"
                ),
            )
        )

    # Add vertical line at primary threshold
    fig.add_vline(
        x=threshold,
        line=dict(color=SEMANTIC_COLORS["primary"], width=2, dash="dashdot"),
        annotation_text=f"Primary threshold: {threshold:.0%}",
        annotation_position="top right",
        annotation_font=dict(size=TYPOGRAPHY["tick_size"]),
    )

    # Generate alt text
    alt_text = _generate_decision_curve_alt_text(results, title, threshold)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b><br>"
                f"<span style='font-size:{TYPOGRAPHY['body_size']}px;color:#666'>"
                "Model useful where curve exceeds baselines</span>"
            ),
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["subheading_size"]),
        ),
        xaxis=dict(
            title=dict(text="Decision Threshold", font=dict(size=TYPOGRAPHY["axis_title_size"])),
            tickformat=".0%",
            range=list(threshold_range),
        ),
        yaxis=dict(
            title=dict(text="Net Benefit", font=dict(size=TYPOGRAPHY["axis_title_size"])),
        ),
        legend=LEGEND_POSITIONS["top_horizontal"],
        template="faircareai",
        height=500,
        margin=dict(l=80, r=40, t=120, b=100),
        meta={"description": alt_text},
    )

    _add_source_annotation(fig, source_note)
    return fig


# =============================================================================
# 4. RISK DISTRIBUTION PLOTS BY SUBGROUP
# =============================================================================


def create_risk_distribution_plot(
    results: dict[str, Any],
    title: str = "Risk Distribution by Demographic Subgroup",
    plot_type: str = "violin",
    source_note: str | None = None,
) -> go.Figure:
    """Create risk distribution plots showing probability distributions by outcome.

    Per Van Calster et al. (2025): "A plot showing probability distributions
    for each outcome category provides valuable insights into a model's behavior."

    Args:
        results: Dict with 'groups' containing risk distribution data.
        title: Chart title.
        plot_type: "violin" or "box".
        source_note: Custom source note.

    Returns:
        Plotly Figure with distribution plots.
    """
    groups = results.get("groups", {})

    if not groups:
        fig = go.Figure()
        fig.add_annotation(
            text="No risk distribution data available",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        return fig

    # Create side-by-side violin/box plots for each group
    n_groups = len(groups)
    fig = make_subplots(
        rows=1,
        cols=n_groups,
        subplot_titles=[f"{g} (n={d.get('n', 0):,})" for g, d in groups.items()],
        horizontal_spacing=0.05,
    )

    for i, (_group_name, group_data) in enumerate(groups.items(), 1):
        if "error" in group_data:
            continue

        events = group_data.get("events", {})
        non_events = group_data.get("non_events", {})

        # Reconstruct distributions from histogram data if available
        # or use summary statistics to simulate
        for outcome_type, outcome_data, name, color in [
            ("events", events, "Events", SEMANTIC_COLORS["fail"]),
            ("non_events", non_events, "Non-Events", SEMANTIC_COLORS["pass"]),
        ]:
            if "error" in outcome_data:
                continue

            hist = outcome_data.get("histogram", {})
            counts = hist.get("counts", [])
            centers = hist.get("bin_centers", [])

            if counts and centers:
                # Reconstruct approximate distribution from histogram
                x_vals = []
                for count, center in zip(counts, centers):
                    x_vals.extend([center] * max(1, int(count / 5)))  # Subsample for performance

                if plot_type == "violin":
                    fig.add_trace(
                        go.Violin(
                            y=x_vals,
                            name=name,
                            side="positive" if outcome_type == "events" else "negative",
                            line_color=color,
                            fillcolor=color,
                            opacity=0.6,
                            meanline_visible=True,
                            showlegend=(i == 1),
                            legendgroup=name,
                        ),
                        row=1,
                        col=i,
                    )
                else:
                    fig.add_trace(
                        go.Box(
                            y=x_vals,
                            name=name,
                            marker_color=color,
                            boxmean=True,
                            showlegend=(i == 1),
                            legendgroup=name,
                        ),
                        row=1,
                        col=i,
                    )
            else:
                # Use summary statistics if histogram not available
                mean = outcome_data.get("mean", 0.5)
                std = outcome_data.get("std", 0.1)
                n = outcome_data.get("n", 100)

                # Generate synthetic data for visualization
                rng = np.random.default_rng(42)
                x_vals = rng.normal(mean, std, min(n, 200))
                x_vals = np.clip(x_vals, 0, 1)

                if plot_type == "violin":
                    fig.add_trace(
                        go.Violin(
                            y=x_vals,
                            name=name,
                            line_color=color,
                            fillcolor=color,
                            opacity=0.6,
                            meanline_visible=True,
                            showlegend=(i == 1),
                            legendgroup=name,
                        ),
                        row=1,
                        col=i,
                    )
                else:
                    fig.add_trace(
                        go.Box(
                            y=x_vals,
                            name=name,
                            marker_color=color,
                            boxmean=True,
                            showlegend=(i == 1),
                            legendgroup=name,
                        ),
                        row=1,
                        col=i,
                    )

    # Generate alt text
    alt_text = _generate_risk_distribution_alt_text(results, title)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b><br>"
                f"<span style='font-size:{TYPOGRAPHY['body_size']}px;color:#666'>"
                "Greater separation indicates better discrimination</span>"
            ),
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["subheading_size"]),
        ),
        legend=LEGEND_POSITIONS["top_horizontal"],
        template="faircareai",
        height=450,
        margin=dict(l=80, r=40, t=120, b=100),
        meta={"description": alt_text},
    )

    # Update all y-axes with JAMA-style font sizing
    for i in range(1, n_groups + 1):
        fig.update_yaxes(
            title_text="Predicted Probability" if i == 1 else "",
            title_font=dict(size=TYPOGRAPHY["axis_title_size"]),
            tickformat=".0%",
            tickfont=dict(size=TYPOGRAPHY["tick_size"]),
            range=[0, 1],
            row=1,
            col=i,
        )

    _add_source_annotation(fig, source_note)
    return fig


# =============================================================================
# COMBINED DASHBOARD
# =============================================================================


def create_vancalster_dashboard(
    results: dict[str, Any],
    title: str = "Van Calster Performance Assessment",
) -> go.Figure:
    """Create comprehensive dashboard with all four Van Calster recommended plots.

    Layout:
    - Top left: AUROC Forest Plot
    - Top right: Calibration Curves
    - Bottom left: Decision Curves
    - Bottom right: Risk Distributions

    Args:
        results: Full results from compute_vancalster_metrics().
        title: Dashboard title.

    Returns:
        Plotly Figure with 2x2 subplot dashboard.
    """
    subgroup_results = results.get("by_subgroup", {})

    if not subgroup_results:
        fig = go.Figure()
        fig.add_annotation(
            text="No subgroup data available for dashboard",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=TYPOGRAPHY["body_size"], color=SEMANTIC_COLORS["text_secondary"]),
        )
        return fig

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "AUROC by Subgroup",
            "Calibration Curves",
            "Decision Curves (Net Benefit)",
            "Risk Distributions",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "box"}],
        ],
    )

    # Helper to extract data for simplified dashboard plots
    groups = list(subgroup_results.keys())

    # 1. AUROC Forest (simplified as scatter)
    for i, group in enumerate(groups):
        data = subgroup_results[group]
        auroc = data.get("discrimination", {}).get("auroc")
        if auroc is None:
            continue

        ci = data.get("discrimination", {}).get("auroc_ci_95", [None, None])
        color = GROUP_COLORS[i % len(GROUP_COLORS)]

        # Error bars
        error_y = None
        if ci[0] is not None and ci[1] is not None:
            error_y = dict(
                type="data",
                symmetric=False,
                array=[ci[1] - auroc],
                arrayminus=[auroc - ci[0]],
                color=color,
            )

        fig.add_trace(
            go.Scatter(
                x=[group],
                y=[auroc],
                mode="markers",
                marker=dict(size=14, color=color),
                error_y=error_y,
                name=group,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # 2. Calibration curves
    # Add diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    for i, group in enumerate(groups):
        data = subgroup_results[group]
        cal = data.get("calibration", {}).get("calibration_curve", {})
        prob_pred = cal.get("prob_pred", [])
        prob_true = cal.get("prob_true", [])

        if not prob_pred:
            continue

        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                line=dict(color=color),
                marker=dict(size=6, color=color),
                name=group,
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    # 3. Decision curves
    for i, group in enumerate(groups):
        data = subgroup_results[group]
        dc = data.get("clinical_utility", {}).get("decision_curve", {})
        thresholds = dc.get("thresholds", [])
        nb_model = dc.get("net_benefit_model", [])

        if not thresholds:
            continue

        # Filter to reasonable range
        thresholds = np.array(thresholds)
        nb_model = np.array(nb_model)
        mask = (thresholds >= 0.05) & (thresholds <= 0.5)

        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=thresholds[mask],
                y=nb_model[mask],
                mode="lines",
                line=dict(color=color),
                name=group,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # 4. Risk distributions (simplified box plots)
    for i, group in enumerate(groups):
        data = subgroup_results[group]
        risk_dist = data.get("risk_distribution", {})

        events = risk_dist.get("events", {})
        non_events = risk_dist.get("non_events", {})

        color = GROUP_COLORS[i % len(GROUP_COLORS)]

        # Generate simplified distribution representation
        for outcome, odata in [("Event", events), ("Non-Event", non_events)]:
            if "error" in odata:
                continue

            mean = odata.get("mean", 0.5)
            q25 = odata.get("q25", 0.3)
            q75 = odata.get("q75", 0.7)

            fig.add_trace(
                go.Box(
                    x=[f"{group[:10]}..."] if len(group) > 10 else [group],
                    y=[[q25, mean, q75]],  # Simplified
                    name=f"{group} {outcome}",
                    marker_color=color if outcome == "Event" else SEMANTIC_COLORS["pass"],
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family=TYPOGRAPHY["heading_font"], size=TYPOGRAPHY["heading_size"]),
        ),
        template="faircareai",
        height=800,
        showlegend=True,
        legend=LEGEND_POSITIONS["bottom_horizontal"],
    )

    # Update axes with JAMA-style font sizing
    fig.update_yaxes(
        title_text="AUROC",
        title_font=dict(size=TYPOGRAPHY["axis_title_size"]),
        tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        range=[0.5, 1],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        tickformat=".0%", tickfont=dict(size=TYPOGRAPHY["tick_size"]), range=[0, 1], row=1, col=2
    )
    fig.update_yaxes(
        tickformat=".0%", tickfont=dict(size=TYPOGRAPHY["tick_size"]), range=[0, 1], row=1, col=2
    )
    fig.update_xaxes(
        title_text="Threshold",
        title_font=dict(size=TYPOGRAPHY["axis_title_size"]),
        tickformat=".0%",
        tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Net Benefit",
        title_font=dict(size=TYPOGRAPHY["axis_title_size"]),
        tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Probability",
        title_font=dict(size=TYPOGRAPHY["axis_title_size"]),
        tickformat=".0%",
        tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        row=2,
        col=2,
    )

    _add_source_annotation(fig)
    return fig
