"""
FairCareAI Governance Dashboard Module

Executive summary visualizations for governance committee presentations.
Designed for stakeholders reviewing fairness analysis results.

Visual indicators provide at-a-glance status relative to configured thresholds.
Healthcare organizations interpret results based on their clinical context.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from faircareai.core.config import FairnessMetric
from faircareai.visualization.themes import (
    COLORSCALES,
    FAIRCAREAI_COLORS,
    SUBPLOT_SPACING,
    TYPOGRAPHY,
    apply_faircareai_theme,
    get_contrast_text_color,
)

if TYPE_CHECKING:
    from faircareai.core.results import AuditResults


def create_executive_summary(results: "AuditResults") -> go.Figure:
    """Create single-page executive summary for governance committee.

    This visualization is designed to be understood by non-technical
    stakeholders in under 30 seconds.

    Args:
        results: AuditResults from FairCareAudit.run().

    Returns:
        Plotly Figure with executive summary.
    """
    gov = results.governance_recommendation
    perf = results.overall_performance
    disc = perf.get("discrimination", {})

    # Determine overall status
    status = gov.get("status", "REVIEW")
    status_colors = {
        "READY": FAIRCAREAI_COLORS["success"],
        "CONDITIONAL": FAIRCAREAI_COLORS["warning"],
        "REVIEW": FAIRCAREAI_COLORS["error"],
    }
    status_color = status_colors.get(status, FAIRCAREAI_COLORS["gray"])

    # Create figure with custom layout
    fig = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"colspan": 3, "type": "indicator"}, None, None],
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"colspan": 3, "type": "table"}, None, None],
        ],
        row_heights=[0.3, 0.35, 0.35],
        vertical_spacing=0.08,
        subplot_titles=(
            "",
            "Model Performance",
            "Fairness Assessment",
            "Data Quality",
            "",
        ),
    )

    # === Row 1: Overall Status ===
    status_text = {
        "READY": "Within Threshold",
        "CONDITIONAL": "Near Threshold",
        "REVIEW": "Outside Threshold",
    }

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=None,
            title=dict(
                text=f"<b style='font-size:36px; color:{status_color}'>{status_text.get(status, status)}</b>",
                font=dict(size=20),
            ),
        ),
        row=1,
        col=1,
    )

    # === Row 2: Key Metrics ===

    # Performance indicator
    auroc = disc.get("auroc", 0)
    perf_color = FAIRCAREAI_COLORS["success"] if auroc >= 0.7 else FAIRCAREAI_COLORS["error"]

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=auroc,
            number={"valueformat": ".2f"},
            gauge=dict(
                axis=dict(range=[0.5, 1], tickformat=".1f"),
                bar=dict(color=perf_color),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    {"range": [0.5, 0.7], "color": COLORSCALES["gauge_steps"]["error"]},
                    {"range": [0.7, 0.8], "color": COLORSCALES["gauge_steps"]["warning"]},
                    {"range": [0.8, 1], "color": COLORSCALES["gauge_steps"]["success"]},
                ],
                threshold=dict(
                    line=dict(color="black", width=3),
                    thickness=0.8,
                    value=0.7,
                ),
            ),
        ),
        row=2,
        col=1,
    )

    # Fairness indicator
    n_warnings = gov.get("n_warnings", 0)
    n_errors = gov.get("n_errors", 0)
    fairness_score = max(0, 100 - n_errors * 30 - n_warnings * 10)
    fairness_color = (
        FAIRCAREAI_COLORS["success"]
        if fairness_score >= 70
        else FAIRCAREAI_COLORS["warning"]
        if fairness_score >= 40
        else FAIRCAREAI_COLORS["error"]
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=fairness_score,
            number={"suffix": "%"},
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color=fairness_color),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    {"range": [0, 40], "color": COLORSCALES["gauge_steps"]["error"]},
                    {"range": [40, 70], "color": COLORSCALES["gauge_steps"]["warning"]},
                    {"range": [70, 100], "color": COLORSCALES["gauge_steps"]["success"]},
                ],
            ),
        ),
        row=2,
        col=2,
    )

    # Data quality indicator
    desc = results.descriptive_stats
    n_total = desc.get("cohort_overview", {}).get("n_total", 0)
    data_quality = "PASS" if n_total >= 1000 else "REVIEW" if n_total >= 100 else "FAIL"
    dq_value = 100 if n_total >= 1000 else 70 if n_total >= 100 else 30
    dq_color = (
        FAIRCAREAI_COLORS["success"]
        if data_quality == "PASS"
        else FAIRCAREAI_COLORS["warning"]
        if data_quality == "REVIEW"
        else FAIRCAREAI_COLORS["error"]
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=dq_value,
            number={"suffix": "%"},
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color=dq_color),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    {"range": [0, 40], "color": COLORSCALES["gauge_steps"]["error"]},
                    {"range": [40, 70], "color": COLORSCALES["gauge_steps"]["warning"]},
                    {"range": [70, 100], "color": COLORSCALES["gauge_steps"]["success"]},
                ],
            ),
        ),
        row=2,
        col=3,
    )

    # === Row 3: Key Findings Table ===
    findings = _generate_key_findings(results)

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Finding</b>", "<b>Status</b>", "<b>Observation</b>"],
                fill_color=FAIRCAREAI_COLORS["primary"],
                font=dict(color="white", size=12),
                align="left",
            ),
            cells=dict(
                values=[
                    findings["findings"],
                    findings["statuses"],
                    findings["actions"],
                ],
                fill_color=[
                    ["white"] * len(findings["findings"]),
                    findings["status_colors"],
                    ["white"] * len(findings["findings"]),
                ],
                font=dict(size=14),
                align="left",
                height=35,
            ),
        ),
        row=3,
        col=1,
    )

    # Generate alt text for WCAG 2.1 AA compliance
    gov = results.governance_recommendation
    n_outside = gov.get("n_errors", gov.get("outside_threshold_count", 0))
    n_near = gov.get("n_warnings", gov.get("near_threshold_count", 0))
    perf = results.overall_performance
    disc = perf.get("discrimination", {})
    auroc = disc.get("auroc", 0)
    alt_text = (
        f"Executive summary for {results.config.model_name} governance review. "
        f"AUROC: {auroc:.3f}. {n_outside} metrics outside threshold, {n_near} near threshold. "
        "Review key findings table for detailed breakdown by category."
    )

    # Update layout
    fig = apply_faircareai_theme(fig)
    fig.update_layout(
        title=dict(
            text=f"<b>Governance Review: {results.config.model_name}</b><br>"
            f"<sup>Version {results.config.model_version}</sup>",
            x=0,
            xanchor="left",
            font=dict(size=18),
        ),
        height=800,
        margin=dict(l=80, r=40, t=80, b=140),  # Extra bottom for rotated labels
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )
    return fig


def _generate_key_findings(results: "AuditResults") -> dict[str, list]:
    """Generate key findings for executive summary table."""
    findings = []
    statuses = []
    status_colors = []
    observations = []

    perf = results.overall_performance
    disc = perf.get("discrimination", {})
    gov = results.governance_recommendation

    # AUROC finding
    auroc = disc.get("auroc", 0)
    findings.append(f"Model Discrimination (AUROC = {auroc:.3f})")
    if auroc >= 0.8:
        statuses.append("STRONG")
        status_colors.append(FAIRCAREAI_COLORS["success"])
        observations.append("Discrimination above 0.8 threshold")
    elif auroc >= 0.7:
        statuses.append("ACCEPTABLE")
        status_colors.append(FAIRCAREAI_COLORS["warning"])
        observations.append("Discrimination meets 0.7 threshold")
    else:
        statuses.append("BELOW THRESHOLD")
        status_colors.append(FAIRCAREAI_COLORS["error"])
        observations.append("Discrimination below 0.7 threshold")

    # Fairness finding
    n_errors = gov.get("n_errors", 0)
    n_warnings = gov.get("n_warnings", 0)
    findings.append(f"Fairness Assessment ({n_errors} exceeded, {n_warnings} near threshold)")
    if n_errors == 0 and n_warnings <= 1:
        statuses.append("WITHIN THRESHOLD")
        status_colors.append(FAIRCAREAI_COLORS["success"])
        observations.append("Disparities within acceptable range")
    elif n_errors == 0:
        statuses.append("NEAR THRESHOLD")
        status_colors.append(FAIRCAREAI_COLORS["warning"])
        observations.append("Some metrics approaching threshold")
    else:
        statuses.append("EXCEEDED THRESHOLD")
        status_colors.append(FAIRCAREAI_COLORS["error"])
        observations.append("Disparities exceed configured threshold")

    # Calibration finding
    cal = perf.get("calibration", {})
    brier = cal.get("brier_score", 0.25)
    slope = cal.get("calibration_slope", 1.0)
    findings.append(f"Calibration (Brier={brier:.3f}, Slope={slope:.2f})")
    if brier < 0.15 and 0.8 <= slope <= 1.2:
        statuses.append("WELL CALIBRATED")
        status_colors.append(FAIRCAREAI_COLORS["success"])
        observations.append("Calibration within ideal range")
    elif brier < 0.25:
        statuses.append("ACCEPTABLE")
        status_colors.append(FAIRCAREAI_COLORS["warning"])
        observations.append("Calibration meets acceptable threshold")
    else:
        statuses.append("NEEDS ATTENTION")
        status_colors.append(FAIRCAREAI_COLORS["error"])
        observations.append("Calibration below acceptable threshold")

    # Sample size finding
    desc = results.descriptive_stats
    n_total = desc.get("cohort_overview", {}).get("n_total", 0)
    findings.append(f"Sample Size (N = {n_total:,})")
    if n_total >= 5000:
        statuses.append("ADEQUATE")
        status_colors.append(FAIRCAREAI_COLORS["success"])
        observations.append("Sample size supports robust analysis")
    elif n_total >= 1000:
        statuses.append("LIMITED")
        status_colors.append(FAIRCAREAI_COLORS["warning"])
        observations.append("Sample size acceptable but limited")
    else:
        statuses.append("SMALL")
        status_colors.append(FAIRCAREAI_COLORS["error"])
        observations.append("Sample size may limit reliability")

    return {
        "findings": findings,
        "statuses": statuses,
        "status_colors": status_colors,
        "actions": observations,
    }


def create_go_nogo_scorecard(results: "AuditResults") -> go.Figure:
    """Create checklist-style go/no-go scorecard.

    Args:
        results: AuditResults from FairCareAudit.run().

    Returns:
        Plotly Figure with checklist scorecard.
    """
    # Define checklist items
    checklist = _build_checklist(results)

    n_pass = sum(1 for item in checklist if item["status"] == "PASS")
    n_warn = sum(1 for item in checklist if item["status"] == "WARN")
    n_fail = sum(1 for item in checklist if item["status"] == "FAIL")

    # Determine overall status based on counts
    if n_fail > 0:
        overall = "OUTSIDE THRESHOLD"
        overall_color = FAIRCAREAI_COLORS["error"]
        summary_text = f"{n_fail} criteria outside configured thresholds."
    elif n_warn > 2:
        overall = "NEAR THRESHOLD"
        overall_color = FAIRCAREAI_COLORS["warning"]
        summary_text = f"{n_warn} criteria near configured thresholds."
    else:
        overall = "WITHIN THRESHOLD"
        overall_color = FAIRCAREAI_COLORS["success"]
        summary_text = "All criteria within configured thresholds."

    # Create figure
    fig = go.Figure()

    # Add checklist items as table
    status_icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    status_colors_map = {
        "PASS": FAIRCAREAI_COLORS["success"],
        "WARN": FAIRCAREAI_COLORS["warning"],
        "FAIL": FAIRCAREAI_COLORS["error"],
    }

    categories = [item["category"] for item in checklist]
    criteria = [item["criterion"] for item in checklist]
    statuses = [f"{status_icons[item['status']]} {item['status']}" for item in checklist]
    notes = [item["note"] for item in checklist]
    colors = [status_colors_map[item["status"]] for item in checklist]

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Category</b>", "<b>Criterion</b>", "<b>Status</b>", "<b>Note</b>"],
                fill_color=FAIRCAREAI_COLORS["primary"],
                font=dict(color="white", size=14),
                align="left",
                height=30,
            ),
            cells=dict(
                values=[categories, criteria, statuses, notes],
                fill_color=[["white"] * len(checklist)] * 3 + [colors],
                font=dict(size=14),
                align="left",
                height=28,
            ),
        )
    )

    # Generate alt text for WCAG 2.1 AA compliance
    alt_text = (
        f"Fairness scorecard for {results.config.model_name}. "
        f"Overall status: {overall}. "
        f"Results: {n_pass} criteria pass, {n_warn} near threshold, {n_fail} outside threshold. "
        f"{summary_text}"
    )

    # Update layout
    fig = apply_faircareai_theme(fig)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Fairness Scorecard: {results.config.model_name}</b><br>"
                f"<span style='color:{overall_color}; font-size:24px'>{overall}</span><br>"
                f"<sup>{n_pass} Pass | {n_warn} Near | {n_fail} Outside</sup>"
            ),
            x=0,
            xanchor="left",
            font=dict(size=16),
        ),
        height=500,
        margin=dict(l=80, r=40, t=120, b=80),
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )
    return fig


def _build_checklist(results: "AuditResults") -> list[dict]:
    """Build checklist items for go/no-go scorecard."""
    checklist = []

    perf = results.overall_performance
    disc = perf.get("discrimination", {})
    cal = perf.get("calibration", {})
    desc = results.descriptive_stats
    gov = results.governance_recommendation

    # Performance criteria
    auroc = disc.get("auroc", 0)
    checklist.append(
        {
            "category": "Performance",
            "criterion": "AUROC ≥ 0.70",
            "status": "PASS" if auroc >= 0.7 else "FAIL",
            "note": f"Observed: {auroc:.3f}",
        }
    )

    brier = cal.get("brier_score", 0.25)
    checklist.append(
        {
            "category": "Performance",
            "criterion": "Brier Score ≤ 0.25",
            "status": "PASS" if brier <= 0.25 else "WARN" if brier <= 0.30 else "FAIL",
            "note": f"Observed: {brier:.4f}",
        }
    )

    slope = cal.get("calibration_slope", 1.0)
    checklist.append(
        {
            "category": "Performance",
            "criterion": "Calibration Slope 0.8-1.2",
            "status": "PASS" if 0.8 <= slope <= 1.2 else "WARN",
            "note": f"Observed: {slope:.2f}",
        }
    )

    # Fairness criteria
    n_errors = gov.get("n_errors", 0)
    checklist.append(
        {
            "category": "Fairness",
            "criterion": "No critical disparities",
            "status": "PASS" if n_errors == 0 else "FAIL",
            "note": f"{n_errors} critical issue(s)",
        }
    )

    n_warnings = gov.get("n_warnings", 0)
    checklist.append(
        {
            "category": "Fairness",
            "criterion": "Minimal fairness warnings",
            "status": "PASS" if n_warnings <= 1 else "WARN" if n_warnings <= 3 else "FAIL",
            "note": f"{n_warnings} warning(s)",
        }
    )

    # Metric justification
    has_justification = bool(results.config.fairness_justification)
    checklist.append(
        {
            "category": "Fairness",
            "criterion": "Fairness metric justified",
            "status": "PASS" if has_justification else "WARN",
            "note": "Documented" if has_justification else "Not documented",
        }
    )

    # Data quality criteria
    n_total = desc.get("cohort_overview", {}).get("n_total", 0)
    checklist.append(
        {
            "category": "Data Quality",
            "criterion": "Sample size ≥ 1000",
            "status": "PASS" if n_total >= 1000 else "WARN" if n_total >= 100 else "FAIL",
            "note": f"N = {n_total:,}",
        }
    )

    # Check for small subgroups
    subgroup_ok = True
    for _attr_name, attr_data in results.subgroup_performance.items():
        # Extract groups from nested structure
        groups_data = attr_data.get("groups", attr_data) if isinstance(attr_data, dict) else {}
        for group_name, group_data in groups_data.items():
            # Skip metadata keys
            if group_name in ("attribute", "threshold", "reference", "disparities"):
                continue
            if isinstance(group_data, dict):
                n = group_data.get("n", 0)
                if n < 30:
                    subgroup_ok = False
                    break
        if not subgroup_ok:
            break

    checklist.append(
        {
            "category": "Data Quality",
            "criterion": "All subgroups n ≥ 30",
            "status": "PASS" if subgroup_ok else "WARN",
            "note": "Adequate" if subgroup_ok else "Small subgroups detected",
        }
    )

    # Governance criteria
    has_model_name = bool(results.config.model_name)
    has_version = bool(results.config.model_version)
    checklist.append(
        {
            "category": "Governance",
            "criterion": "Model identified",
            "status": "PASS" if has_model_name and has_version else "WARN",
            "note": f"{results.config.model_name} v{results.config.model_version}",
        }
    )

    return checklist


def create_fairness_dashboard(results: "AuditResults") -> go.Figure:
    """Create comprehensive 4-panel fairness dashboard.

    Args:
        results: AuditResults from FairCareAudit.run().

    Returns:
        Plotly Figure with fairness dashboard.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "AUROC by Subgroup",
            "Selection Rate by Subgroup",
            "Fairness Metrics",
            "Disparity Summary",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "indicator"}],
        ],
        **SUBPLOT_SPACING["dashboard"],
    )

    # Collect data from all sensitive attributes
    all_groups = []
    all_aurocs = []
    all_selection_rates = []
    all_colors = []

    for attr_name, attr_data in results.subgroup_performance.items():
        if not isinstance(attr_data, dict):
            continue

        # Extract groups from nested structure
        groups_data = attr_data.get("groups", attr_data)

        for group_name, group_data in groups_data.items():
            # Skip metadata keys
            if group_name in ("attribute", "threshold", "reference", "disparities"):
                continue
            if not isinstance(group_data, dict) or "error" in group_data:
                continue

            label = f"{attr_name}: {group_name}"
            all_groups.append(label)
            all_aurocs.append(group_data.get("auroc", 0))
            all_selection_rates.append(group_data.get("selection_rate", 0))

            # Color by reference status
            is_ref = group_data.get("is_reference", False)
            all_colors.append(
                FAIRCAREAI_COLORS["primary"] if is_ref else FAIRCAREAI_COLORS["secondary"]
            )

    # === Panel 1: AUROC by Subgroup ===
    if all_aurocs:
        fig.add_trace(
            go.Bar(
                x=all_groups,
                y=all_aurocs,
                marker_color=all_colors,
                text=[f"{a:.3f}" for a in all_aurocs],
                textposition="inside",
                textfont=dict(
                    color=[get_contrast_text_color(c) for c in all_colors],
                    size=TYPOGRAPHY["annotation_size"],
                ),
                hovertemplate="%{x}<br>AUROC: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", row=1, col=1)
        fig.update_yaxes(range=[0.5, 1], row=1, col=1)

    # === Panel 2: Selection Rate by Subgroup ===
    # Get selection rates from fairness metrics
    selection_groups = []
    selection_rates = []
    selection_colors = []

    for attr_name, attr_data in results.fairness_metrics.items():
        if not isinstance(attr_data, dict):
            continue

        group_metrics = attr_data.get("group_metrics", {})
        reference = attr_data.get("reference")

        for group_name, gm in group_metrics.items():
            if not isinstance(gm, dict) or "error" in gm:
                continue

            label = f"{attr_name}: {group_name}"
            selection_groups.append(label)
            selection_rates.append(gm.get("selection_rate", 0))
            selection_colors.append(
                FAIRCAREAI_COLORS["primary"]
                if group_name == reference
                else FAIRCAREAI_COLORS["secondary"]
            )

    if selection_rates:
        fig.add_trace(
            go.Bar(
                x=selection_groups,
                y=selection_rates,
                marker_color=selection_colors,
                text=[f"{r:.1%}" for r in selection_rates],
                textposition="inside",
                textfont=dict(
                    color=[get_contrast_text_color(c) for c in selection_colors],
                    size=TYPOGRAPHY["annotation_size"],
                ),
                hovertemplate="%{x}<br>Selection Rate: %{y:.1%}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(range=[0, max(selection_rates) * 1.3], tickformat=".0%", row=1, col=2)

    # === Panel 3: Fairness Metrics Summary ===
    fairness_labels = []
    fairness_values = []
    fairness_colors = []

    for attr_name, attr_data in results.fairness_metrics.items():
        if not isinstance(attr_data, dict):
            continue

        summary = attr_data.get("summary", {})

        # Equal opportunity
        eo = summary.get("equal_opportunity", {})
        if eo:
            worst = eo.get("worst_diff", 0)
            fairness_labels.append(f"{attr_name}: TPR Diff")
            fairness_values.append(abs(worst))
            fairness_colors.append(
                FAIRCAREAI_COLORS["success"] if abs(worst) <= 0.1 else FAIRCAREAI_COLORS["error"]
            )

        # Equalized odds
        eq = summary.get("equalized_odds", {})
        if eq:
            worst = eq.get("worst_diff", 0)
            fairness_labels.append(f"{attr_name}: EO Diff")
            fairness_values.append(abs(worst))
            fairness_colors.append(
                FAIRCAREAI_COLORS["success"] if worst <= 0.1 else FAIRCAREAI_COLORS["error"]
            )

    if fairness_values:
        fig.add_trace(
            go.Bar(
                x=fairness_labels,
                y=fairness_values,
                marker_color=fairness_colors,
                text=[f"{v:.2f}" for v in fairness_values],
                textposition="inside",
                textfont=dict(
                    color=[get_contrast_text_color(c) for c in fairness_colors],
                    size=TYPOGRAPHY["annotation_size"],
                ),
                hovertemplate="%{x}<br>Disparity: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0.1, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_yaxes(title_text="Absolute Disparity", row=2, col=1)

    # === Panel 4: Overall Summary ===
    gov = results.governance_recommendation
    n_pass = gov.get("n_pass", 0)
    n_warnings = gov.get("n_warnings", 0)
    n_errors = gov.get("n_errors", 0)

    total_checks = n_pass + n_warnings + n_errors
    pass_pct = (n_pass / total_checks * 100) if total_checks > 0 else 0

    status = gov.get("status", "REVIEW")
    status_color = {
        "READY": FAIRCAREAI_COLORS["success"],
        "CONDITIONAL": FAIRCAREAI_COLORS["warning"],
        "REVIEW": FAIRCAREAI_COLORS["error"],
    }.get(status, FAIRCAREAI_COLORS["gray"])

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=pass_pct,
            number={"suffix": "%", "valueformat": ".0f"},
            title={"text": f"Status: {status}"},
            delta={"reference": 80, "relative": False},
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color=status_color),
                steps=[
                    {"range": [0, 40], "color": COLORSCALES["gauge_steps"]["error"]},
                    {"range": [40, 70], "color": COLORSCALES["gauge_steps"]["warning"]},
                    {"range": [70, 100], "color": COLORSCALES["gauge_steps"]["success"]},
                ],
                threshold=dict(
                    line=dict(color="black", width=2),
                    thickness=0.75,
                    value=80,
                ),
            ),
        ),
        row=2,
        col=2,
    )

    # Generate alt text for WCAG 2.1 AA compliance
    n_attributes = len(results.fairness_metrics)
    alt_text = (
        f"Fairness dashboard for {results.config.model_name}. "
        f"Analyzes {n_attributes} sensitive attribute(s). "
        f"Status: {status}. {n_pass} checks pass, {n_warnings} warnings, {n_errors} errors. "
        f"{pass_pct:.0f}% of fairness criteria met."
    )

    # Update layout
    fig = apply_faircareai_theme(fig)
    fig.update_layout(
        title=dict(
            text=f"<b>Fairness Dashboard: {results.config.model_name}</b>",
            x=0,
            xanchor="left",
            font=dict(size=16),
        ),
        height=1000,  # Taller for more spacing
        margin=dict(l=80, r=40, t=100, b=160),  # Extra bottom for rotated labels
        showlegend=False,
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    # Add axis titles for each subplot
    fig.update_xaxes(
        title_text="Subgroup",
        tickangle=-40,
        tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        automargin=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="AUROC Score",
        tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Subgroup",
        tickangle=-40,
        tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        automargin=True,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="Selection Rate (%)",
        tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text="Fairness Metric",
        tickangle=-40,
        tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        automargin=True,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Absolute Disparity",
        tickfont=dict(size=TYPOGRAPHY["tick_size"]),
        row=2,
        col=1,
    )

    # Style subplot titles smaller to avoid collisions
    for annotation in fig["layout"]["annotations"]:
        if hasattr(annotation, "text") and annotation.text in [
            "AUROC by Subgroup",
            "Selection Rate by Subgroup",
            "Fairness Metrics",
            "Disparity Summary",
        ]:
            annotation.font = dict(size=TYPOGRAPHY["annotation_size"], family="Inter, sans-serif")
    return fig


def plot_subgroup_comparison(
    results: "AuditResults",
    metric: str = "auroc",
) -> go.Figure:
    """Plot subgroup performance comparison for a specific metric.

    Args:
        results: AuditResults from FairCareAudit.run().
        metric: Metric to compare ('auroc', 'tpr', 'fpr', 'ppv', 'selection_rate').

    Returns:
        Plotly Figure with subgroup comparison.
    """
    metric_labels = {
        "auroc": "AUROC",
        "tpr": "True Positive Rate (Sensitivity)",
        "fpr": "False Positive Rate",
        "ppv": "Positive Predictive Value",
        "selection_rate": "Selection Rate",
    }

    fig = go.Figure()

    for attr_name, attr_data in results.subgroup_performance.items():
        if not isinstance(attr_data, dict):
            continue

        # Extract groups from nested structure
        groups_data = attr_data.get("groups", attr_data)

        groups = []
        values = []
        errors_low = []
        errors_high = []
        colors = []

        for group_name, group_data in groups_data.items():
            # Skip metadata keys
            if group_name in ("attribute", "threshold", "reference", "disparities"):
                continue
            if not isinstance(group_data, dict) or "error" in group_data:
                continue

            groups.append(group_name)
            val = group_data.get(metric, 0)
            values.append(val)

            # Confidence interval if available
            ci = group_data.get(f"{metric}_ci_95", [None, None])
            if ci[0] is not None:
                errors_low.append(val - ci[0])
                errors_high.append(ci[1] - val)
            else:
                errors_low.append(0)
                errors_high.append(0)

            # Color by reference
            is_ref = group_data.get("is_reference", False)
            colors.append(
                FAIRCAREAI_COLORS["primary"] if is_ref else FAIRCAREAI_COLORS["secondary"]
            )

        # Add bar trace for this attribute
        fig.add_trace(
            go.Bar(
                name=attr_name,
                x=groups,
                y=values,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=errors_high,
                    arrayminus=errors_low,
                )
                if any(errors_high)
                else None,
                marker_color=colors,
                text=[f"{v:.3f}" for v in values],
                textposition="inside",
                textfont=dict(
                    color=[get_contrast_text_color(c) for c in colors],
                    size=TYPOGRAPHY["annotation_size"],
                ),
                hovertemplate=f"{attr_name}: %{{x}}<br>{metric_labels.get(metric, metric)}: %{{y:.3f}}<extra></extra>",
            )
        )

    # Add reference line for certain metrics
    if metric == "auroc":
        fig.add_hline(
            y=0.7,
            line_dash="dash",
            line_color=FAIRCAREAI_COLORS["error"],
            annotation_text="Minimum acceptable",
        )

    # Update layout
    fig = apply_faircareai_theme(fig)
    fig.update_layout(
        title=dict(
            text=f"Subgroup {metric_labels.get(metric, metric)} Comparison",
            x=0,
            xanchor="left",
        ),
        xaxis_title="Subgroup",
        yaxis_title=metric_labels.get(metric, metric),
        height=500,
        barmode="group",
    )

    # Set appropriate y-axis range
    if metric == "auroc":
        fig.update_yaxes(range=[0.5, 1])
    elif metric in ["tpr", "fpr", "ppv", "selection_rate"]:
        fig.update_yaxes(range=[0, 1], tickformat=".0%")

    return fig


# === GOVERNANCE PERSONA FIGURE GENERATORS ===


def create_governance_overall_figures(results: "AuditResults") -> dict[str, Any]:
    """Create 4 overall performance figures for governance report.

    Returns simplified, large-font figures suitable for non-technical audiences:
    1. AUROC Gauge - Model discrimination with pass/fail threshold
    2. Calibration Plot - Observed vs predicted with slope
    3. Brier Score - Calibration quality gauge
    4. Classification Metrics - Bar chart of key metrics

    Explanatory text is returned separately under '_explanations' key for HTML rendering.

    Args:
        results: AuditResults from FairCareAudit.run().

    Returns:
        Dict mapping figure title to Plotly Figure, plus '_explanations' dict.
    """
    figures: dict[str, Any] = {}

    perf = results.overall_performance
    disc = perf.get("discrimination", {})
    cal = perf.get("calibration", {})

    # Plain language explanations for governance audiences
    PLAIN_EXPLANATIONS = {
        "auroc": (
            "AUROC measures how well the model separates high-risk from low-risk patients. "
            "Think of it as the model's ability to rank patients correctly. "
            "Score of 0.5 = random guessing (coin flip). Score of 1.0 = perfect ranking. "
            "Healthcare standard: 0.7 or higher is acceptable, 0.8+ is strong."
        ),
        "calibration": (
            "Calibration checks if predicted risks match actual outcomes. "
            "If the model predicts 20% risk for a group, do about 20% actually experience the outcome? "
            "Points closer to the diagonal line = more trustworthy risk estimates. "
            "Why it matters: Under/over-estimating risk can lead to wrong treatment decisions."
        ),
        "brier": (
            "Brier Score measures overall prediction accuracy (0 = perfect, 0.25 = poor). "
            "Lower is better. Think of it as the 'error' in risk predictions. "
            "Score <0.15 = excellent calibration. Score 0.15-0.25 = acceptable. Score >0.25 = needs improvement."
        ),
        "classification": (
            "At the chosen risk threshold, these metrics show what happens to patients: "
            "Sensitivity = % of actual cases correctly identified. "
            "Specificity = % without the condition correctly identified. "
            "PPV = When flagged positive, % who actually have the condition."
        ),
    }

    # 1. AUROC Gauge
    auroc = disc.get("auroc", 0)
    auroc_color = FAIRCAREAI_COLORS["success"] if auroc >= 0.7 else FAIRCAREAI_COLORS["error"]

    fig_auroc = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=auroc,
            number={"valueformat": ".2f", "font": {"size": 44, "color": auroc_color}},
            title={"text": "<b>AUROC</b>", "font": {"size": 20}},
            gauge=dict(
                axis=dict(range=[0.5, 1], tickformat=".1f", tickfont={"size": 14}),
                bar=dict(color=auroc_color),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    {"range": [0.5, 0.7], "color": "#ffebee"},
                    {"range": [0.7, 0.8], "color": "#fff3e0"},
                    {"range": [0.8, 1], "color": "#e8f5e9"},
                ],
                threshold=dict(
                    line=dict(color="black", width=4),
                    thickness=0.8,
                    value=0.7,
                ),
            ),
        )
    )
    fig_auroc.update_layout(
        height=400,
        margin=dict(l=80, r=40, t=90, b=80),
    )
    figures["AUROC"] = fig_auroc

    # 2. Calibration Plot (simplified)
    fig_cal = go.Figure()
    # Add perfect calibration line
    fig_cal.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Perfect Calibration",
            showlegend=True,
        )
    )

    # Get calibration curve data if available (prefer smoothed)
    cal_curve_smoothed = cal.get("calibration_curve_smoothed", {}) or {}
    smoothed_pred = cal_curve_smoothed.get("prob_pred", [])
    smoothed_true = cal_curve_smoothed.get("prob_true", [])

    cal_curve = cal.get("calibration_curve", {}) or {}
    prob_pred = cal_curve.get("prob_pred", [])
    prob_true = cal_curve.get("prob_true", [])

    if smoothed_pred and smoothed_true:
        x_vals = smoothed_pred
        y_vals = smoothed_true
    elif prob_pred and prob_true:
        x_vals = prob_pred
        y_vals = prob_true
    else:
        # Generate approximate calibration line from slope
        slope = cal.get("calibration_slope", 1.0)
        intercept = cal.get("calibration_intercept", 0.0)
        x_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
        y_vals = [max(0, min(1, intercept + slope * x)) for x in x_vals]

    fig_cal.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            line=dict(color=FAIRCAREAI_COLORS["primary"], width=3),
            marker=dict(size=10),
            name="Model Calibration",
            showlegend=True,
        )
    )

    slope = cal.get("calibration_slope", 1.0)
    slope_status = "PASS" if 0.8 <= slope <= 1.2 else "REVIEW"

    slope_color = (
        FAIRCAREAI_COLORS["success"] if 0.8 <= slope <= 1.2 else FAIRCAREAI_COLORS["error"]
    )
    fig_cal.update_layout(
        title=dict(text="<b>Calibration</b>", font=dict(size=20)),
        xaxis=dict(
            title="Predicted Risk (what the model says)",
            range=[0, 1],
            tickfont={"size": 14},
            tickformat=".0%",
        ),
        yaxis=dict(
            title="Observed Rate (what actually happened)",
            range=[0, 1],
            tickfont={"size": 14},
            tickformat=".0%",
        ),
        height=400,
        margin=dict(l=80, r=40, t=90, b=80),
        legend=dict(x=0.02, y=0.98, font=dict(size=14)),
        annotations=[
            dict(
                text=f"<b>Slope: {slope:.2f}</b> ({slope_status})",
                x=0.95,
                y=0.05,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=16, color=slope_color),
            ),
        ],
    )
    figures["Calibration"] = fig_cal

    # 3. Brier Score Gauge
    brier = cal.get("brier_score", 0.25)
    brier_color = (
        FAIRCAREAI_COLORS["success"]
        if brier < 0.15
        else FAIRCAREAI_COLORS["warning"]
        if brier < 0.25
        else FAIRCAREAI_COLORS["error"]
    )

    fig_brier = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=brier,
            number={"valueformat": ".3f", "font": {"size": 44, "color": brier_color}},
            title={"text": "<b>Brier Score</b>", "font": {"size": 20}},
            gauge=dict(
                axis=dict(range=[0, 0.5], tickformat=".2f", tickfont={"size": 14}),
                bar=dict(color=brier_color),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    {"range": [0, 0.15], "color": "#e8f5e9"},
                    {"range": [0.15, 0.25], "color": "#fff3e0"},
                    {"range": [0.25, 0.5], "color": "#ffebee"},
                ],
            ),
        )
    )
    fig_brier.update_layout(
        height=400,
        margin=dict(l=80, r=40, t=90, b=80),
    )
    figures["Brier Score"] = fig_brier

    # 4. Classification Metrics at Threshold
    cls = perf.get("classification_at_threshold", {})
    threshold = cls.get("threshold", 0.5)
    sensitivity = cls.get("sensitivity", 0) * 100
    specificity = cls.get("specificity", 0) * 100
    ppv = cls.get("ppv", 0) * 100

    fig_class = go.Figure()
    metrics = ["Sensitivity", "Specificity", "PPV"]
    values = [sensitivity, specificity, ppv]
    colors = [
        FAIRCAREAI_COLORS["success"]
        if v >= 70
        else FAIRCAREAI_COLORS["warning"]
        if v >= 50
        else FAIRCAREAI_COLORS["error"]
        for v in values
    ]

    fig_class.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f"<b>{v:.0f}%</b>" for v in values],
            textposition="inside",
            textfont=dict(color=[get_contrast_text_color(c) for c in colors], size=14),
        )
    )

    fig_class.update_layout(
        title=dict(
            text=f"<b>Classification Metrics at Threshold {threshold:.2f}</b>", font=dict(size=20)
        ),
        xaxis=dict(title="Performance Metric", tickfont={"size": 14}),
        yaxis=dict(
            title="Performance at Threshold (%)",
            range=[0, 110],
            ticksuffix="%",
            tickfont={"size": 14},
        ),
        height=400,
        margin=dict(l=80, r=40, t=90, b=80),
        showlegend=False,
    )
    figures["Classification"] = fig_class

    # Return explanations separately for HTML rendering
    figures["_explanations"] = PLAIN_EXPLANATIONS

    return figures


def create_governance_subgroup_figures(
    results: "AuditResults",
    primary_metric: FairnessMetric | None = None,
) -> dict[str, dict[str, go.Figure]]:
    """Create subgroup performance figures for governance report.

    For each sensitive attribute, generates 4 figures (Van Calster 4):
    1. AUROC by Subgroup - Discrimination comparison
    2. TPR by Subgroup - Sensitivity comparison (equal opportunity)
    3. FPR by Subgroup - Specificity comparison (equalized odds)
    4. Selection Rate by Subgroup - Demographic parity check

    Each figure includes plain language explanations per the governance spec.
    Charts corresponding to the primary_metric are visually highlighted.

    Args:
        results: AuditResults from FairCareAudit.run().
        primary_metric: The primary fairness metric to highlight. If None,
            uses results.config.primary_fairness_metric.

    Returns:
        Dict mapping attribute name to dict of figure title -> Plotly Figure.
    """
    # Get primary metric from results if not provided
    if primary_metric is None:
        primary_metric = getattr(results.config, "primary_fairness_metric", None)

    # Plain language explanations for Van Calster 4 visualizations
    SUBGROUP_EXPLANATIONS = {
        "auroc": (
            "AUROC by Subgroup: Does the model perform equally well across all demographic groups? "
            "All bars should be similar height (difference <0.05 is ideal). "
            "Lower bars mean the model is less accurate for that group. "
            "Why it matters: We want the model to work well for everyone, not just some groups."
        ),
        "sensitivity": (
            "Sensitivity (True Positive Rate): Of patients who actually develop the outcome, "
            "what percentage does the model correctly identify in each group? "
            "Large differences mean the model 'misses' more cases in certain groups. "
            "Fairness goal: Differences between groups should be <10 percentage points."
        ),
        "fpr": (
            "False Positive Rate: Of patients who DON'T have the outcome, "
            "what percentage are incorrectly flagged as high-risk in each group? "
            "Lower is better (fewer false alarms). "
            "Fairness concern: Higher FPR means a group gets unnecessary interventions/worry."
        ),
        "selection": (
            "Selection Rate: What percentage of each group is flagged as 'high-risk' by the model? "
            "This shows which groups the model identifies for intervention. "
            "Large differences may indicate disparate treatment even if clinically justified. "
            "Consider: Should intervention rates differ by demographics?"
        ),
    }

    all_figures = {}

    for attr_name, attr_data in results.subgroup_performance.items():
        if not isinstance(attr_data, dict):
            continue

        figures = {}

        # Collect data for this attribute
        # The subgroup_performance structure has a nested 'groups' key
        groups_data = attr_data.get("groups", attr_data)
        reference_group = attr_data.get("reference", None)

        groups = []
        auroc_vals = []
        tpr_vals = []
        fpr_vals = []
        selection_vals = []
        is_reference = []

        for group_name, group_data in groups_data.items():
            if not isinstance(group_data, dict) or "error" in group_data:
                continue
            # Skip metadata keys that aren't actual groups
            if group_name in ("attribute", "threshold", "reference", "disparities"):
                continue

            groups.append(group_name)
            auroc_vals.append(group_data.get("auroc", 0))
            tpr_vals.append(group_data.get("tpr", 0))
            fpr_vals.append(group_data.get("fpr", 0))
            selection_vals.append(group_data.get("selection_rate", 0))
            # Check is_reference from group_data or compare with reference_group
            is_ref = group_data.get("is_reference", group_name == reference_group)
            is_reference.append(is_ref)

        if not groups:
            continue

        # Color scheme - highlight reference group
        colors = [
            FAIRCAREAI_COLORS["primary"] if ref else FAIRCAREAI_COLORS["secondary"]
            for ref in is_reference
        ]

        # Determine which charts correspond to the primary metric
        is_tpr_primary = primary_metric in (FairnessMetric.EQUAL_OPPORTUNITY, FairnessMetric.EQUALIZED_ODDS)
        is_fpr_primary = primary_metric == FairnessMetric.EQUALIZED_ODDS
        is_selection_primary = primary_metric == FairnessMetric.DEMOGRAPHIC_PARITY

        # 1. AUROC by Subgroup
        fig_auroc = _create_subgroup_bar_chart(
            groups,
            auroc_vals,
            colors,
            "Model Accuracy (AUROC) by Demographic Group",
            y_range=[0.5, 1],
            threshold_line=0.7,
            threshold_label="Acceptable minimum (0.7)",
            explanation=SUBGROUP_EXPLANATIONS["auroc"],
            y_axis_title="AUROC (Model Accuracy Score)",
            x_axis_title="Demographic Group",
            is_primary_metric=False,  # AUROC not directly a fairness metric
        )
        figures["AUROC by Subgroup"] = fig_auroc

        # 2. TPR (Sensitivity) by Subgroup - Equal Opportunity / Equalized Odds
        fig_tpr = _create_subgroup_bar_chart(
            groups,
            [v * 100 for v in tpr_vals],
            colors,
            "Sensitivity: % of Actual Cases Detected by Group",
            y_range=[0, 100],
            y_suffix="%",
            threshold_line=None,
            explanation=SUBGROUP_EXPLANATIONS["sensitivity"],
            y_axis_title="True Positive Rate (%)",
            x_axis_title="Demographic Group",
            is_primary_metric=is_tpr_primary,
        )
        figures["Sensitivity by Subgroup"] = fig_tpr

        # 3. FPR by Subgroup - Equalized Odds
        fig_fpr = _create_subgroup_bar_chart(
            groups,
            [v * 100 for v in fpr_vals],
            colors,
            "False Alarms: % Incorrectly Flagged by Group",
            y_range=[0, 50],
            y_suffix="%",
            threshold_line=None,
            explanation=SUBGROUP_EXPLANATIONS["fpr"],
            y_axis_title="False Positive Rate (%)",
            x_axis_title="Demographic Group",
            is_primary_metric=is_fpr_primary,
        )
        figures["FPR by Subgroup"] = fig_fpr

        # 4. Selection Rate by Subgroup - Demographic Parity
        fig_sel = _create_subgroup_bar_chart(
            groups,
            [v * 100 for v in selection_vals],
            colors,
            "Intervention Rate: % Flagged as High-Risk by Group",
            y_range=[0, 100],
            y_suffix="%",
            threshold_line=None,
            explanation=SUBGROUP_EXPLANATIONS["selection"],
            y_axis_title="Selection Rate (% flagged)",
            x_axis_title="Demographic Group",
            is_primary_metric=is_selection_primary,
        )
        figures["Selection Rate by Subgroup"] = fig_sel

        all_figures[attr_name] = figures

    return all_figures


def _create_subgroup_bar_chart(
    groups: list[str],
    values: list[float],
    colors: list[str],
    title: str,
    y_range: list[float] | None = None,
    y_suffix: str = "",
    threshold_line: float | None = None,
    threshold_label: str = "",
    explanation: str = "",
    y_axis_title: str = "Value",
    x_axis_title: str = "Group",
    is_primary_metric: bool = False,
) -> go.Figure:
    """Create a simplified bar chart for subgroup comparison.

    Args:
        groups: List of group names.
        values: List of metric values.
        colors: List of bar colors.
        title: Chart title.
        y_range: Y-axis range [min, max].
        y_suffix: Suffix for y-axis tick labels (e.g., "%").
        threshold_line: Optional horizontal threshold line.
        threshold_label: Label for threshold line.
        explanation: Plain language explanation for non-technical audiences.
        y_axis_title: Descriptive label for Y-axis.
        x_axis_title: Descriptive label for X-axis.
        is_primary_metric: If True, adds visual highlighting to indicate
            this chart corresponds to the selected primary fairness metric.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Format text based on whether it's percentage
    if y_suffix == "%":
        text_vals = [f"<b>{v:.0f}%</b>" for v in values]
    else:
        text_vals = [f"<b>{v:.2f}</b>" for v in values]

    fig.add_trace(
        go.Bar(
            x=groups,
            y=values,
            marker_color=colors,
            text=text_vals,
            textposition="inside",
            textfont=dict(color=[get_contrast_text_color(c) for c in colors], size=12),
        )
    )

    # Add threshold line if specified
    if threshold_line is not None:
        fig.add_hline(
            y=threshold_line,
            line_dash="dash",
            line_color=FAIRCAREAI_COLORS["error"],
            annotation_text=threshold_label,
            annotation_position="top right",
            annotation_font=dict(size=14),
        )

    # No in-chart annotations - they overlap with labels
    # Explanation text will be added via HTML wrapper in generator.py

    # Add visual highlighting for primary metric
    if is_primary_metric:
        title_text = f"<b>{title}</b><br><span style='font-size:12px; color:#0072B2;'>★ YOUR SELECTED FAIRNESS METRIC</span>"
        plot_bgcolor = "rgba(0, 114, 178, 0.05)"  # Light blue background
    else:
        title_text = f"<b>{title}</b>"
        plot_bgcolor = "white"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16), x=0, xanchor="left"),
        xaxis=dict(
            title=x_axis_title,
            tickfont={"size": 11},
            tickangle=-40,  # Moderate angle for readability
            title_font=dict(size=13),
            automargin=True,  # Auto-adjust margin for labels
        ),
        yaxis=dict(
            title=y_axis_title,
            range=y_range,
            ticksuffix=y_suffix,
            tickfont={"size": 13},
            title_font=dict(size=13),
        ),
        height=380,  # Good height for chart
        margin=dict(l=80, r=40, t=100, b=160),  # Top: long titles, bottom: rotated labels
        showlegend=False,
        plot_bgcolor=plot_bgcolor,
    )

    return fig


# === VAN CALSTER 2025 FIGURE GENERATORS ===


def create_governance_roc_curve(results: "AuditResults") -> go.Figure | None:
    """Create simplified ROC curve for governance report.

    Van Calster et al. (2025): RECOMMENDED

    Shows model's ranking ability - curve hugging top-left = good.
    Simpler than the full discrimination curves plot.

    Args:
        results: AuditResults from FairCareAudit.run()

    Returns:
        Plotly Figure with ROC curve, or None if data unavailable
    """
    perf = results.overall_performance
    disc = perf.get("discrimination", {})
    roc_data = disc.get("roc_curve", {})

    fpr = roc_data.get("fpr", [])
    tpr = roc_data.get("tpr", [])
    auroc = disc.get("auroc", 0)

    if not fpr or not tpr:
        return None

    fig = go.Figure()

    # Diagonal reference (random guessing)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash", width=2),
            name="Random Guessing (AUC=0.50)",
            showlegend=True,
        )
    )

    # ROC curve with shaded area
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            line=dict(color=FAIRCAREAI_COLORS["primary"], width=3),
            fill="tonexty",
            fillcolor="rgba(0,114,178,0.15)",
            name=f"Model (AUC={auroc:.2f})",
            hovertemplate="FPR: %{x:.1%}<br>TPR: %{y:.1%}<extra></extra>",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=dict(text="<b>ROC Curve</b>", font=dict(size=20)),
        xaxis=dict(
            title="False Positive Rate (% incorrectly flagged)",
            tickformat=".0%",
            range=[0, 1],
            tickfont={"size": 14},
        ),
        yaxis=dict(
            title="True Positive Rate (% correctly identified)",
            tickformat=".0%",
            range=[0, 1],
            tickfont={"size": 14},
        ),
        height=450,
        margin=dict(l=90, r=40, t=70, b=90),
        legend=dict(x=0.6, y=0.15, bgcolor="rgba(255,255,255,0.9)", font=dict(size=14)),
        showlegend=True,
    )

    return fig


def create_governance_probability_distribution(results: "AuditResults") -> go.Figure | None:
    """Create risk score distribution histograms by outcome.

    Van Calster et al. (2025): RECOMMENDED

    Shows separation between positive/negative outcomes.
    Wide gap = good discrimination.

    Args:
        results: AuditResults from FairCareAudit.run()

    Returns:
        Plotly Figure with overlapping histograms, or None if data unavailable
    """
    # Access raw data via _audit reference
    if results._audit is None:
        return None

    try:
        df = results._audit.df
        y_true_col = results._audit.y_true_col
        y_prob_col = results._audit.y_prob_col

        y_true = np.asarray(df[y_true_col].to_numpy())
        y_prob = np.asarray(df[y_prob_col].to_numpy())
    except (AttributeError, KeyError):
        return None

    # Separate by outcome
    probs_positive = y_prob[y_true == 1]
    probs_negative = y_prob[y_true == 0]

    fig = go.Figure()

    # Histogram for negative outcomes
    fig.add_trace(
        go.Histogram(
            x=probs_negative,
            name="No Outcome (Negative)",
            marker_color=FAIRCAREAI_COLORS["success"],
            opacity=0.65,
            xbins=dict(start=0, end=1, size=0.05),
            histnorm="probability density",
        )
    )

    # Histogram for positive outcomes
    fig.add_trace(
        go.Histogram(
            x=probs_positive,
            name="Outcome Occurred (Positive)",
            marker_color=FAIRCAREAI_COLORS["error"],
            opacity=0.65,
            xbins=dict(start=0, end=1, size=0.05),
            histnorm="probability density",
        )
    )

    fig.update_layout(
        title=dict(text="<b>Risk Score Distribution by Outcome</b>", font=dict(size=20)),
        xaxis=dict(
            title="Predicted Risk Score",
            tickformat=".0%",
            range=[0, 1],
            tickfont={"size": 14},
        ),
        yaxis=dict(
            title="Proportion of Patients",
            tickfont={"size": 14},
        ),
        barmode="overlay",
        height=450,
        margin=dict(l=90, r=40, t=70, b=90),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)", font=dict(size=14)),
        showlegend=True,
    )

    return fig
