"""
FairCareAI Governance Dashboard Module

Executive summary visualizations for governance committee presentations.
Designed for stakeholders reviewing fairness analysis results.

Visual indicators provide at-a-glance status relative to configured thresholds.
Healthcare organizations interpret results based on their clinical context.
"""

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from faircareai.visualization.themes import (
    COLORSCALES,
    FAIRCAREAI_COLORS,
    GOVERNANCE_DISCLAIMER_SHORT,
    SUBPLOT_SPACING,
    apply_faircareai_theme,
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
                values=["<b>Finding</b>", "<b>Status</b>", "<b>Action</b>"],
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
                font=dict(size=11),
                align="left",
                height=25,
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
            x=0.5,
            font=dict(size=18),
        ),
        height=700,
        margin=dict(t=100, b=80),
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    # Add disclaimer
    fig.add_annotation(
        text=GOVERNANCE_DISCLAIMER_SHORT,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.08,
        showarrow=False,
        font=dict(size=10, color=FAIRCAREAI_COLORS["gray"]),
    )

    return fig


def _generate_key_findings(results: "AuditResults") -> dict[str, list]:
    """Generate key findings for executive summary table."""
    findings = []
    statuses = []
    status_colors = []
    actions = []

    perf = results.overall_performance
    disc = perf.get("discrimination", {})
    gov = results.governance_recommendation

    # AUROC finding
    auroc = disc.get("auroc", 0)
    findings.append(f"Model Discrimination (AUROC = {auroc:.3f})")
    if auroc >= 0.8:
        statuses.append("EXCELLENT")
        status_colors.append(FAIRCAREAI_COLORS["success"])
        actions.append("No action required")
    elif auroc >= 0.7:
        statuses.append("ACCEPTABLE")
        status_colors.append(FAIRCAREAI_COLORS["warning"])
        actions.append("Monitor performance post-deployment")
    else:
        statuses.append("CONCERN")
        status_colors.append(FAIRCAREAI_COLORS["error"])
        actions.append("Improve model before deployment")

    # Fairness finding
    n_errors = gov.get("n_errors", 0)
    n_warnings = gov.get("n_warnings", 0)
    findings.append(f"Fairness Assessment ({n_errors} errors, {n_warnings} warnings)")
    if n_errors == 0 and n_warnings <= 1:
        statuses.append("PASS")
        status_colors.append(FAIRCAREAI_COLORS["success"])
        actions.append("Continue monitoring")
    elif n_errors == 0:
        statuses.append("REVIEW")
        status_colors.append(FAIRCAREAI_COLORS["warning"])
        actions.append("Address warnings before deployment")
    else:
        statuses.append("FAIL")
        status_colors.append(FAIRCAREAI_COLORS["error"])
        actions.append("Critical issues must be resolved")

    # Calibration finding
    cal = perf.get("calibration", {})
    brier = cal.get("brier_score", 0.25)
    slope = cal.get("calibration_slope", 1.0)
    findings.append(f"Calibration (Brier={brier:.3f}, Slope={slope:.2f})")
    if brier < 0.15 and 0.8 <= slope <= 1.2:
        statuses.append("GOOD")
        status_colors.append(FAIRCAREAI_COLORS["success"])
        actions.append("No recalibration needed")
    elif brier < 0.25:
        statuses.append("ACCEPTABLE")
        status_colors.append(FAIRCAREAI_COLORS["warning"])
        actions.append("Consider recalibration")
    else:
        statuses.append("POOR")
        status_colors.append(FAIRCAREAI_COLORS["error"])
        actions.append("Recalibration required")

    # Sample size finding
    desc = results.descriptive_stats
    n_total = desc.get("cohort_overview", {}).get("n_total", 0)
    findings.append(f"Sample Size (N = {n_total:,})")
    if n_total >= 5000:
        statuses.append("ADEQUATE")
        status_colors.append(FAIRCAREAI_COLORS["success"])
        actions.append("Sufficient for validation")
    elif n_total >= 1000:
        statuses.append("LIMITED")
        status_colors.append(FAIRCAREAI_COLORS["warning"])
        actions.append("Plan external validation")
    else:
        statuses.append("SMALL")
        status_colors.append(FAIRCAREAI_COLORS["error"])
        actions.append("Expand dataset before deployment")

    return {
        "findings": findings,
        "statuses": statuses,
        "status_colors": status_colors,
        "actions": actions,
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
                font=dict(color="white", size=12),
                align="left",
                height=30,
            ),
            cells=dict(
                values=[categories, criteria, statuses, notes],
                fill_color=[["white"] * len(checklist)] * 3 + [colors],
                font=dict(size=11),
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
            x=0.5,
            font=dict(size=16),
        ),
        height=500,
        margin=dict(t=120, b=80),
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    # Add summary annotation
    fig.add_annotation(
        text=f"<b>Summary:</b> {summary_text}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.08,
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=overall_color,
        borderwidth=2,
    )

    # Add disclaimer
    fig.add_annotation(
        text=GOVERNANCE_DISCLAIMER_SHORT,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.18,
        showarrow=False,
        font=dict(size=9, color=FAIRCAREAI_COLORS["gray"]),
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
        for _group_name, group_data in attr_data.items():
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
        **SUBPLOT_SPACING["default"],
    )

    # Collect data from all sensitive attributes
    all_groups = []
    all_aurocs = []
    all_selection_rates = []
    all_colors = []

    for attr_name, attr_data in results.subgroup_performance.items():
        if not isinstance(attr_data, dict):
            continue

        for group_name, group_data in attr_data.items():
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
                textposition="outside",
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
                textposition="outside",
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
                textposition="outside",
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
            text=f"Fairness Dashboard: {results.config.model_name}",
            x=0.5,
        ),
        height=700,
        showlegend=False,
        meta={"description": alt_text},  # WCAG 2.1 screen reader support
    )

    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)

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

        groups = []
        values = []
        errors_low = []
        errors_high = []
        colors = []

        for group_name, group_data in attr_data.items():
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
                textposition="outside",
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
            x=0.5,
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
