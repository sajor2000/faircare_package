"""
FairCareAI - Governance Report Page

Generates governance-ready reports with sign-off workflow.
Designed for clinical stakeholders and governance committees.
"""

from datetime import datetime
from typing import Any

import polars as pl
import streamlit as st

from faircareai.dashboard.components.accessibility import (
    announce_status_change,
    render_semantic_heading,
    render_skip_link,
)
from faircareai.dashboard.components.audience_toggle import (
    render_audience_toggle,
)
from faircareai.visualization.themes import (
    GOVERNANCE_DISCLAIMER_FULL,
    render_scorecard_html,
    render_status_badge,
)


def render_governance_summary(result: Any) -> None:
    """Render governance summary section."""
    render_semantic_heading("Results Summary", level=2, id="summary")

    # Present counts objectively
    total = result.pass_count + result.warn_count + result.fail_count
    summary_text = (
        f"Analysis complete. {total} metrics computed across subgroups. "
        f"{result.pass_count} within threshold, {result.warn_count} near threshold, "
        f"{result.fail_count} outside threshold."
    )

    # Visual status based on counts
    if result.fail_count > 0:
        status_type = "fail"
    elif result.warn_count > 0:
        status_type = "warn"
    else:
        status_type = "pass"

    st.markdown(
        render_status_badge(status_type, "<b>Results Summary</b>"),
        unsafe_allow_html=True,
    )

    st.markdown(f"_{summary_text}_")

    st.write("")

    # Scorecard
    st.markdown(
        render_scorecard_html(result.pass_count, result.warn_count, result.fail_count),
        unsafe_allow_html=True,
    )


def render_key_findings(result: Any) -> None:
    """Render key findings section."""
    render_semantic_heading("Computed Metrics", level=2, id="findings")

    findings = []

    # Overall performance
    metrics_overall = result.metrics_df.filter(pl.col("group") == "_overall").head(1)
    if len(metrics_overall) > 0:
        row = metrics_overall.to_dicts()[0]
        auroc = row.get("auroc")
        if auroc is not None:
            findings.append(
                {
                    "type": "neutral",
                    "text": f"Overall model discrimination AUROC: {auroc:.2f}",
                }
            )

    # Largest disparity
    if result.worst_disparity:
        group, metric, value = result.worst_disparity

        metric_plain = {
            "tpr": "detection rate",
            "fpr": "false alarm rate",
            "ppv": "positive prediction accuracy",
        }.get(metric, metric)

        direction = "lower" if value < 0 else "higher"

        findings.append(
            {
                "type": "neutral",
                "text": (
                    f"Largest disparity: {metric_plain} for {group} patients "
                    f"({abs(value):.1%} {direction} than reference)"
                ),
            }
        )

    # Sample size notes
    if result.warnings:
        findings.append(
            {
                "type": "neutral",
                "text": f"{len(result.warnings)} group(s) have small sample sizes",
            }
        )

    # Threshold summary
    findings.append(
        {
            "type": "neutral",
            "text": f"{result.fail_count} metric(s) outside configured thresholds",
        }
    )

    # Render findings
    for finding in findings:
        color = {
            "positive": "#009E73",
            "neutral": "#666666",
            "concern": "#E65100",
        }.get(finding["type"], "#666666")

        st.markdown(
            f"""<div style="padding: 12px 16px; margin: 8px 0; border-left: 4px solid {color}; background: #FAFAFA;">
            {finding["text"]}
            </div>""",
            unsafe_allow_html=True,
        )


def render_metrics_table(result: Any, audience: str) -> None:
    """Render metrics table section."""
    render_semantic_heading("Detailed Metrics", level=2, id="metrics")

    if audience == "governance":
        st.markdown("""
        The table below shows how the model performs for different patient groups.
        Look for large differences between groups in the same column.
        """)

    # Filter out overall row for display
    display_df = result.metrics_df.filter(pl.col("group") != "_overall")

    if len(display_df) > 0:
        # Select columns based on audience
        if audience == "governance":
            columns = ["attribute", "group", "n", "tpr", "fpr", "ppv"]
            rename_map = {
                "attribute": "Demographic",
                "group": "Group",
                "n": "Sample Size",
                "tpr": "Detection Rate",
                "fpr": "False Alarm Rate",
                "ppv": "Flag Accuracy",
            }
        else:
            columns = [
                "attribute",
                "group",
                "n",
                "n_positive",
                "tpr",
                "tpr_ci_lower",
                "tpr_ci_upper",
                "fpr",
                "ppv",
            ]
            rename_map = {
                "attribute": "Attribute",
                "group": "Group",
                "n": "N",
                "n_positive": "N Positive",
                "tpr": "TPR",
                "tpr_ci_lower": "TPR CI Low",
                "tpr_ci_upper": "TPR CI High",
                "fpr": "FPR",
                "ppv": "PPV",
            }

        # Filter to existing columns only
        available_columns = [c for c in columns if c in display_df.columns]
        table_df = display_df.select(available_columns)

        # Convert to pandas and format
        pandas_df = table_df.to_pandas()

        # Format percentage columns
        pct_cols = ["tpr", "fpr", "ppv", "tpr_ci_lower", "tpr_ci_upper"]
        for col in pct_cols:
            if col in pandas_df.columns:
                pandas_df[col] = pandas_df[col].apply(
                    lambda x: f"{x:.1%}" if x is not None else "N/A"
                )

        # Rename columns
        pandas_df = pandas_df.rename(
            columns={k: v for k, v in rename_map.items() if k in pandas_df.columns}
        )

        st.dataframe(pandas_df, use_container_width=True, hide_index=True)

        # Download button
        csv = pandas_df.to_csv(index=False)
        st.download_button(
            label="Download Metrics CSV",
            data=csv,
            file_name="faircareai_governance_metrics.csv",
            mime="text/csv",
        )


def render_sign_off_section(result: Any) -> None:
    """Render governance sign-off section."""
    render_semantic_heading("Governance Sign-Off", level=2, id="signoff")

    st.markdown("""
    ### Review Checklist

    Before approving this model for deployment, please confirm the following:
    """)

    # Checklist items
    checks = [
        ("reviewed_metrics", "I have reviewed the fairness metrics for all demographic groups"),
        (
            "understood_disparities",
            "I understand the identified disparities and their potential clinical implications",
        ),
        ("acceptable_performance", "The overall model performance meets our clinical requirements"),
        ("monitoring_plan", "A monitoring plan is in place for ongoing fairness evaluation"),
        (
            "documentation",
            "Documentation of this review will be retained per organizational policy",
        ),
    ]

    all_checked = True
    for key, label in checks:
        checked = st.checkbox(label, key=f"signoff_{key}")
        if not checked:
            all_checked = False

    st.markdown("---")

    # Reviewer information
    col1, col2 = st.columns(2)

    with col1:
        reviewer_name = st.text_input("Reviewer Name", placeholder="Dr. Jane Smith")
    with col2:
        reviewer_role = st.selectbox(
            "Role",
            options=[
                "Chief Medical Officer",
                "Clinical Informatics Lead",
                "Governance Committee Chair",
                "Data Science Lead",
                "Quality Officer",
                "Other",
            ],
        )

    comments = st.text_area(
        "Comments or Conditions",
        placeholder="Optional: Add any conditions for approval or notes for the record",
    )

    # Decision buttons
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "Approve for Deployment",
            type="primary",
            disabled=not all_checked or not reviewer_name,
            use_container_width=True,
        ):
            st.session_state["governance_decision"] = "approved"
            st.session_state["governance_reviewer"] = reviewer_name
            st.session_state["governance_role"] = reviewer_role
            st.session_state["governance_comments"] = comments
            st.session_state["governance_timestamp"] = datetime.now().isoformat()
            announce_status_change("Deployment approved")
            st.success("Decision recorded: APPROVED FOR DEPLOYMENT")

    with col2:
        if st.button(
            "Request Changes",
            type="secondary",
            disabled=not reviewer_name,
            use_container_width=True,
        ):
            st.session_state["governance_decision"] = "changes_requested"
            st.session_state["governance_reviewer"] = reviewer_name
            st.session_state["governance_role"] = reviewer_role
            st.session_state["governance_comments"] = comments
            st.session_state["governance_timestamp"] = datetime.now().isoformat()
            announce_status_change("Changes requested")
            st.warning("Decision recorded: CHANGES REQUESTED")

    with col3:
        if st.button(
            "Reject",
            type="secondary",
            disabled=not reviewer_name,
            use_container_width=True,
        ):
            st.session_state["governance_decision"] = "rejected"
            st.session_state["governance_reviewer"] = reviewer_name
            st.session_state["governance_role"] = reviewer_role
            st.session_state["governance_comments"] = comments
            st.session_state["governance_timestamp"] = datetime.now().isoformat()
            announce_status_change("Deployment rejected", priority="assertive")
            st.error("Decision recorded: REJECTED")


def render_export_section(result: Any) -> None:
    """Render export options section."""
    render_semantic_heading("Export Report", level=2, id="export")

    st.markdown("""
    Generate a formal governance report for your records.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Generate HTML Report", use_container_width=True):
            try:
                import faircareai.reports.generator  # noqa: F401

                # Would generate actual report here
                st.info("HTML report generation is available via the API")
            except ImportError:
                st.info("Install report dependencies: pip install faircareai[reports]")

    with col2:
        if st.button("Generate PDF Report", use_container_width=True):
            st.info("PDF export available via CLI: `faircareai export --format pdf`")

    with col3:
        if st.button("Download JSON Data", use_container_width=True):
            # Export as JSON
            import json

            export_data = {
                "model_name": result.model_name
                if hasattr(result, "model_name")
                else "Uploaded Model",
                "audit_date": datetime.now().isoformat(),
                "n_samples": result.n_samples,
                "threshold": result.threshold,
                "pass_count": result.pass_count,
                "warn_count": result.warn_count,
                "fail_count": result.fail_count,
                "metrics": result.metrics_df.to_dicts(),
                "disparities": result.disparities_df.to_dicts()
                if len(result.disparities_df) > 0
                else [],
            }

            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2, default=str),
                file_name="faircareai_audit_results.json",
                mime="application/json",
            )


def render_governance_page() -> None:
    """Render the governance report page."""
    render_skip_link()

    # Check for audit result
    if "audit_result" not in st.session_state:
        st.warning("No audit results available. Please run an analysis first.")
        if st.button("Go to Analysis"):
            st.switch_page("pages/2_analysis.py")
        return

    result = st.session_state["audit_result"]

    # Header
    render_semantic_heading("Governance Report", level=1, id="page-title")
    st.caption(f"Step 3 of 4 | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Methodology note
    with st.expander("Analysis Methodology", expanded=False):
        st.markdown(GOVERNANCE_DISCLAIMER_FULL)

    # Audience toggle
    st.markdown("---")
    audience = render_audience_toggle()
    st.markdown("---")

    # Render sections
    render_governance_summary(result)
    st.markdown("---")
    render_key_findings(result)
    st.markdown("---")
    render_metrics_table(result, audience)
    st.markdown("---")
    render_sign_off_section(result)
    st.markdown("---")
    render_export_section(result)

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back to Analysis", use_container_width=True):
            st.switch_page("pages/2_analysis.py")

    with col2:
        if st.button("Settings", use_container_width=True):
            st.switch_page("pages/4_settings.py")


# Run the page
render_governance_page()
