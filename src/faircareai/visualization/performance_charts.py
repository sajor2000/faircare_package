"""
FairCareAI Performance Visualization Module

Publication-quality visualizations for model performance per TRIPOD+AI.
Designed for governance committee presentations to lay stakeholders.

Methodology: Van Calster et al. (2025), TRIPOD+AI (Collins et al. 2024).
"""

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from faircareai.visualization.themes import (
    FAIRCAREAI_COLORS,
    GOVERNANCE_DISCLAIMER_SHORT,
    LEGEND_POSITIONS,
    TYPOGRAPHY,
    apply_faircareai_theme,
)

if TYPE_CHECKING:
    from faircareai.core.results import AuditResults


def plot_discrimination_curves(results: "AuditResults") -> go.Figure:
    """Plot ROC and Precision-Recall curves side by side.

    TRIPOD+AI 2.1: Discrimination metrics visualization.

    Args:
        results: AuditResults from FairCareAudit.run().

    Returns:
        Plotly Figure with side-by-side curves.
    """
    perf = results.overall_performance
    disc = perf.get("discrimination", {})

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("ROC Curve", "Precision-Recall Curve"),
        horizontal_spacing=0.12,
    )

    # === ROC Curve ===
    roc_data = disc.get("roc_curve", {})
    fpr = roc_data.get("fpr", [])
    tpr = roc_data.get("tpr", [])
    auroc = disc.get("auroc", 0)
    auroc_ci = disc.get("auroc_ci_fmt", "")

    if fpr and tpr:
        # ROC curve
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Model (AUROC={auroc:.3f})",
                line=dict(color=FAIRCAREAI_COLORS["primary"], width=2.5),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Diagonal reference
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color=FAIRCAREAI_COLORS["gray"], width=1.5, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Update ROC axes
    fig.update_xaxes(
        title_text="False Positive Rate (1 - Specificity)",
        range=[0, 1],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="True Positive Rate (Sensitivity)",
        range=[0, 1],
        row=1,
        col=1,
    )

    # === Precision-Recall Curve ===
    pr_data = disc.get("pr_curve", {})
    recall = pr_data.get("recall", [])
    precision = pr_data.get("precision", [])
    auprc = disc.get("auprc", 0)
    prevalence = disc.get("prevalence", 0)

    if recall and precision:
        # PR curve
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"Model (AUPRC={auprc:.3f})",
                line=dict(color=FAIRCAREAI_COLORS["secondary"], width=2.5),
                hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Baseline (prevalence)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[prevalence, prevalence],
                mode="lines",
                name=f"Baseline (Prevalence={prevalence:.3f})",
                line=dict(color=FAIRCAREAI_COLORS["gray"], width=1.5, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Update PR axes
    fig.update_xaxes(
        title_text="Recall (Sensitivity)",
        range=[0, 1],
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="Precision (PPV)",
        range=[0, 1],
        row=1,
        col=2,
    )

    # Generate alt text for WCAG 2.1 AA compliance
    auprc = disc.get("auprc", 0)
    alt_text = (
        f"Model discrimination curves. Left: ROC curve with AUROC = {auroc:.3f} {auroc_ci}. "
        f"Right: Precision-Recall curve with AUPRC = {auprc:.3f}. "
        "Higher values indicate better model discrimination."
    )

    # Apply theme and layout
    fig = apply_faircareai_theme(fig)
    fig.update_layout(
        title=dict(
            text=f"Model Discrimination: AUROC = {auroc:.3f} {auroc_ci}",
            x=0.5,
        ),
        height=450,
        showlegend=True,
        legend=LEGEND_POSITIONS["bottom_horizontal"],
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    # Add annotation
    fig.add_annotation(
        text=GOVERNANCE_DISCLAIMER_SHORT,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.35,
        showarrow=False,
        font=dict(size=14, color=FAIRCAREAI_COLORS["gray"]),
    )

    return fig


def plot_calibration_curve(results: "AuditResults") -> go.Figure:
    """Plot calibration curve for overall model.

    TRIPOD+AI 2.2: Calibration visualization.

    Args:
        results: AuditResults from FairCareAudit.run().

    Returns:
        Plotly Figure with calibration curve.
    """
    perf = results.overall_performance
    cal = perf.get("calibration", {})
    cal_curve = cal.get("calibration_curve", {})

    prob_true = cal_curve.get("prob_true", [])
    prob_pred = cal_curve.get("prob_pred", [])

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color=FAIRCAREAI_COLORS["gray"], width=1.5, dash="dash"),
        )
    )

    # Calibration curve
    if prob_true and prob_pred:
        fig.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                name="Model Calibration",
                line=dict(color=FAIRCAREAI_COLORS["primary"], width=2.5),
                marker=dict(size=8),
                hovertemplate=(
                    "Mean Predicted: %{x:.3f}<br>Observed Rate: %{y:.3f}<extra></extra>"
                ),
            )
        )

    # Metrics annotation
    slope = cal.get("calibration_slope", 1.0)
    brier = cal.get("brier_score", 0)
    ici = cal.get("ici", 0)

    metrics_text = (
        f"<b>Calibration Metrics</b><br>"
        f"Slope: {slope:.2f} (ideal: 1.00)<br>"
        f"Brier Score: {brier:.4f}<br>"
        f"ICI: {ici:.4f}"
    )

    fig.add_annotation(
        text=metrics_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        font=dict(size=TYPOGRAPHY["annotation_size"]),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor=FAIRCAREAI_COLORS["gray"],
        borderwidth=1,
    )

    # Generate alt text for WCAG 2.1 AA compliance
    alt_text = (
        f"Model calibration curve. Brier score: {brier:.4f} (lower is better). "
        f"Calibration slope: {slope:.3f} (ideal: 1.0). "
        "Points close to the diagonal indicate well-calibrated predictions."
    )

    # Apply theme
    fig = apply_faircareai_theme(fig)
    fig.update_layout(
        title=dict(text="Model Calibration", x=0.5),
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Observed Outcome Rate",
        height=500,
        width=600,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    # Disclaimer
    fig.add_annotation(
        text=GOVERNANCE_DISCLAIMER_SHORT,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.15,
        showarrow=False,
        font=dict(size=14, color=FAIRCAREAI_COLORS["gray"]),
    )

    return fig


def plot_threshold_analysis(
    results: "AuditResults",
    selected_threshold: float | None = None,
) -> go.Figure:
    """Interactive threshold sensitivity analysis.

    TRIPOD+AI 2.4: Threshold selection transparency.

    Args:
        results: AuditResults from FairCareAudit.run().
        selected_threshold: Threshold to highlight.

    Returns:
        Plotly Figure with threshold analysis.
    """
    perf = results.overall_performance
    thresh_data = perf.get("threshold_analysis", {})
    plot_data = thresh_data.get("plot_data", {})

    thresholds = plot_data.get("thresholds", [])
    sensitivity = plot_data.get("sensitivity", [])
    specificity = plot_data.get("specificity", [])
    ppv = plot_data.get("ppv", [])
    npv = plot_data.get("npv", [])
    pct_flagged = plot_data.get("pct_flagged", [])

    if not thresholds:
        fig = go.Figure()
        fig.add_annotation(
            text="No threshold analysis data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Classification Metrics by Threshold",
            "Percentage of Patients Flagged",
        ),
        vertical_spacing=0.15,
        row_heights=[0.65, 0.35],
    )

    # Classification metrics
    metrics = [
        ("Sensitivity (TPR)", sensitivity, FAIRCAREAI_COLORS["primary"]),
        ("Specificity (1-FPR)", specificity, FAIRCAREAI_COLORS["secondary"]),
        ("PPV (Precision)", ppv, FAIRCAREAI_COLORS["accent"]),
        ("NPV", npv, FAIRCAREAI_COLORS["success"]),
    ]

    for name, values, color in metrics:
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=values,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
                hovertemplate=f"{name}: %{{y:.1%}}<br>Threshold: %{{x:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Percentage flagged
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=pct_flagged,
            mode="lines+markers",
            name="% Flagged",
            line=dict(color=FAIRCAREAI_COLORS["warning"], width=2),
            marker=dict(size=6),
            hovertemplate="Flagged: %{y:.1f}%<br>Threshold: %{x:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Highlight selected threshold
    if selected_threshold is not None:
        for row in [1, 2]:
            fig.add_vline(
                x=selected_threshold,
                line=dict(color=FAIRCAREAI_COLORS["error"], width=2, dash="dash"),
                annotation_text=f"Selected: {selected_threshold:.2f}",
                annotation_position="top",
                row=row,
                col=1,
            )

    # Update axes
    fig.update_xaxes(title_text="Decision Threshold", row=2, col=1)
    fig.update_yaxes(title_text="Metric Value", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="% Flagged", row=2, col=1)

    # Generate alt text for WCAG 2.1 AA compliance
    threshold_text = f" Selected threshold: {selected_threshold:.2f}." if selected_threshold else ""
    alt_text = (
        "Threshold selection impact analysis. "
        "Top: Performance metrics (Sensitivity, Specificity, PPV, NPV) across thresholds. "
        "Bottom: Percentage of patients flagged at each threshold."
        f"{threshold_text}"
    )

    # Apply theme
    fig = apply_faircareai_theme(fig)
    fig.update_layout(
        title=dict(text="Threshold Selection Impact", x=0.5),
        height=600,
        showlegend=True,
        legend=LEGEND_POSITIONS["top_horizontal"],
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    return fig


def plot_decision_curve(results: "AuditResults") -> go.Figure:
    """Plot Decision Curve Analysis for clinical utility.

    TRIPOD+AI 2.5: Clinical utility assessment.

    Args:
        results: AuditResults from FairCareAudit.run().

    Returns:
        Plotly Figure with DCA curves.
    """
    perf = results.overall_performance
    dca = perf.get("decision_curve", {})

    thresholds = dca.get("thresholds", [])
    nb_model = dca.get("net_benefit_model", [])
    nb_all = dca.get("net_benefit_all", [])
    nb_none = dca.get("net_benefit_none", [])
    useful_range = dca.get("useful_range_summary", {})

    fig = go.Figure()

    # Net benefit curves
    if thresholds:
        # Model
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=nb_model,
                mode="lines",
                name="Model",
                line=dict(color=FAIRCAREAI_COLORS["primary"], width=2.5),
                hovertemplate="Net Benefit: %{y:.3f}<br>Threshold: %{x:.2f}<extra></extra>",
            )
        )

        # Treat All
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=nb_all,
                mode="lines",
                name="Treat All",
                line=dict(color=FAIRCAREAI_COLORS["warning"], width=2, dash="dash"),
            )
        )

        # Treat None
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=nb_none,
                mode="lines",
                name="Treat None",
                line=dict(color=FAIRCAREAI_COLORS["gray"], width=2, dash="dot"),
            )
        )

    # Shade useful range
    if useful_range.get("min") is not None and useful_range.get("max") is not None:
        fig.add_vrect(
            x0=useful_range["min"],
            x1=useful_range["max"],
            fillcolor=FAIRCAREAI_COLORS["success"],
            opacity=0.1,
            line_width=0,
            annotation_text="Useful Range",
            annotation_position="top",
        )

    # Generate alt text for WCAG 2.1 AA compliance
    alt_text = (
        "Decision Curve Analysis showing clinical utility of the model. "
        "Compares net benefit of using the model versus treating all or no patients. "
        "Model is useful when its curve is above both 'Treat All' and 'Treat None' strategies."
    )

    # Apply theme
    fig = apply_faircareai_theme(fig)
    fig.update_layout(
        title=dict(
            text="Decision Curve Analysis: When is the Model Clinically Useful?",
            x=0.5,
        ),
        xaxis_title="Threshold Probability",
        yaxis_title="Net Benefit",
        height=500,
        xaxis=dict(range=[0, 1]),
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    # Interpretation annotation
    interpretation = (
        "<b>Interpretation:</b><br>"
        "Model is useful when its curve is above both<br>"
        "'Treat All' and 'Treat None' strategies."
    )

    fig.add_annotation(
        text=interpretation,
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.98,
        showarrow=False,
        font=dict(size=TYPOGRAPHY["annotation_size"]),
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=FAIRCAREAI_COLORS["gray"],
        borderwidth=1,
    )

    # Disclaimer
    fig.add_annotation(
        text=GOVERNANCE_DISCLAIMER_SHORT,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.12,
        showarrow=False,
        font=dict(size=14, color=FAIRCAREAI_COLORS["gray"]),
    )

    return fig


def plot_confusion_matrix(results: "AuditResults") -> go.Figure:
    """Plot confusion matrix heatmap.

    Args:
        results: AuditResults from FairCareAudit.run().

    Returns:
        Plotly Figure with confusion matrix.
    """
    perf = results.overall_performance
    cm = perf.get("confusion_matrix", {})

    matrix = cm.get("matrix", [[0, 0], [0, 0]])
    threshold = cm.get("threshold", 0.5)

    # Extract values
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp

    # Create text annotations
    text = [
        [
            f"TN<br>{tn:,}<br>({tn / total * 100:.1f}%)",
            f"FP<br>{fp:,}<br>({fp / total * 100:.1f}%)",
        ],
        [
            f"FN<br>{fn:,}<br>({fn / total * 100:.1f}%)",
            f"TP<br>{tp:,}<br>({tp / total * 100:.1f}%)",
        ],
    ]

    # Color scale (darker for higher values)
    z = [[tn, fp], [fn, tp]]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14),
            colorscale=[
                [0, "white"],
                [0.5, FAIRCAREAI_COLORS["secondary"]],
                [1, FAIRCAREAI_COLORS["primary"]],
            ],
            showscale=False,
            hoverongaps=False,
        )
    )

    # Generate alt text for WCAG 2.1 AA compliance
    alt_text = (
        f"Confusion matrix at threshold {threshold:.2f}. "
        f"True Negatives: {tn}, False Positives: {fp}, "
        f"False Negatives: {fn}, True Positives: {tp}."
    )

    # Apply theme
    fig = apply_faircareai_theme(fig)
    fig.update_layout(
        title=dict(text=f"Confusion Matrix at Threshold = {threshold:.2f}", x=0.5),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        width=500,
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    return fig


def plot_performance_summary(results: "AuditResults") -> go.Figure:
    """Create a single-page performance summary for governance.

    Args:
        results: AuditResults from FairCareAudit.run().

    Returns:
        Plotly Figure with key performance metrics.
    """
    perf = results.overall_performance
    disc = perf.get("discrimination", {})
    cal = perf.get("calibration", {})
    cls = perf.get("classification_at_threshold", {})

    # Create 2x2 subplot
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Discrimination (AUROC)",
            "Calibration",
            "Classification Metrics",
            "Clinical Utility",
        ),
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "bar"}, {"type": "indicator"}],
        ],
        vertical_spacing=0.2,
        horizontal_spacing=0.15,
    )

    # AUROC gauge
    auroc = disc.get("auroc", 0)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=auroc,
            number={"suffix": "", "valueformat": ".3f"},
            gauge=dict(
                axis=dict(range=[0.5, 1]),
                bar=dict(color=FAIRCAREAI_COLORS["primary"]),
                steps=[
                    {"range": [0.5, 0.7], "color": FAIRCAREAI_COLORS["error"]},
                    {"range": [0.7, 0.8], "color": FAIRCAREAI_COLORS["warning"]},
                    {"range": [0.8, 1], "color": FAIRCAREAI_COLORS["success"]},
                ],
                threshold=dict(
                    line=dict(color="black", width=2),
                    thickness=0.75,
                    value=0.8,
                ),
            ),
        ),
        row=1,
        col=1,
    )

    # Calibration gauge (Brier score - lower is better)
    brier = cal.get("brier_score", 0.25)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=brier,
            number={"suffix": "", "valueformat": ".3f"},
            title={"text": "Brier Score"},
            gauge=dict(
                axis=dict(range=[0, 0.5]),
                bar=dict(color=FAIRCAREAI_COLORS["secondary"]),
                steps=[
                    {"range": [0, 0.1], "color": FAIRCAREAI_COLORS["success"]},
                    {"range": [0.1, 0.2], "color": FAIRCAREAI_COLORS["warning"]},
                    {"range": [0.2, 0.5], "color": FAIRCAREAI_COLORS["error"]},
                ],
            ),
        ),
        row=1,
        col=2,
    )

    # Classification metrics bar chart
    metrics = {
        "Sensitivity": cls.get("sensitivity", 0),
        "Specificity": cls.get("specificity", 0),
        "PPV": cls.get("ppv", 0),
        "NPV": cls.get("npv", 0),
    }

    fig.add_trace(
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=[
                FAIRCAREAI_COLORS["primary"],
                FAIRCAREAI_COLORS["secondary"],
                FAIRCAREAI_COLORS["accent"],
                FAIRCAREAI_COLORS["success"],
            ],
            text=[f"{v:.1%}" for v in metrics.values()],
            textposition="outside",
        ),
        row=2,
        col=1,
    )

    # % Flagged indicator
    pct_flagged = cls.get("pct_flagged", 0)
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=pct_flagged,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "% Flagged High Risk"},
        ),
        row=2,
        col=2,
    )

    # Generate alt text for WCAG 2.1 AA compliance
    auroc = disc.get("auroc", 0)
    brier = cal.get("brier_score", 0)
    sensitivity = cls.get("sensitivity", 0)
    specificity = cls.get("specificity", 0)
    ppv = cls.get("ppv", 0)
    npv = cls.get("npv", 0)
    pct_flagged = cls.get("pct_flagged", 0)
    alt_text = (
        f"Model performance summary for {results.config.model_name}. "
        f"AUROC: {auroc:.3f}. Brier Score: {brier:.4f}. "
        f"Sensitivity: {sensitivity:.1%}, Specificity: {specificity:.1%}, "
        f"PPV: {ppv:.1%}, NPV: {npv:.1%}. "
        f"{pct_flagged:.1f}% of patients flagged as high risk."
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Model Performance Summary - {results.config.model_name}",
            x=0.5,
        ),
        height=600,
        showlegend=False,
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )
    fig.update_yaxes(title_text="Value", range=[0, 1.1], tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Metric", row=2, col=1)

    # Apply theme
    fig = apply_faircareai_theme(fig)

    return fig
