"""
FairCareAI - Analysis Page

Main analysis dashboard with dual-audience support.
Displays performance metrics, fairness analysis, and visualizations.
"""

import polars as pl
import streamlit as st

from faircareai.dashboard.components.accessibility import (
    accessible_plotly_chart,
    announce_status_change,
    generate_chart_alt_text,
    render_semantic_heading,
    render_skip_link,
)
from faircareai.dashboard.components.audience_toggle import (
    get_metric_display,
    get_section_content,
    render_audience_toggle,
)
from faircareai.dashboard.components.glossary import (
    render_glossary_sidebar,
    render_inline_definition,
)
from faircareai.visualization.themes import (
    GOVERNANCE_DISCLAIMER_SHORT,
    render_scorecard_html,
    render_status_badge,
)


@st.cache_data
def run_audit(_df: pl.DataFrame, threshold: float, group_cols: list[str]):
    """Run and cache audit results."""
    from faircareai.core.audit import FairAudit

    audit = FairAudit(
        _df,
        threshold=threshold,
        model_name="Uploaded Model",
    )
    result = audit.run(group_cols=group_cols)
    return result, audit


def render_executive_summary(result, audience: str) -> None:
    """Render executive summary section."""
    render_semantic_heading("Results Summary", level=2, id="summary")

    # Present counts objectively
    if audience == "governance":
        status_desc = (
            f"{result.pass_count} metrics within threshold, "
            f"{result.warn_count} near threshold, "
            f"{result.fail_count} outside threshold."
        )
    else:
        status_desc = (
            f"{result.pass_count} pass, {result.warn_count} warn, "
            f"{result.fail_count} outside configured thresholds."
        )

    # Visual status based on counts
    if result.fail_count > 0:
        status_type = "fail"
    elif result.warn_count > 0:
        status_type = "warn"
    else:
        status_type = "pass"

    st.markdown(
        render_status_badge(
            status_type,
            f"<b>Analysis Complete</b><br><span style='font-size:13px'>{status_desc}</span>",
        ),
        unsafe_allow_html=True,
    )

    st.write("")

    # Scorecard
    st.markdown(
        render_scorecard_html(result.pass_count, result.warn_count, result.fail_count),
        unsafe_allow_html=True,
    )

    # Key metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", f"{result.n_samples:,}")
    with col2:
        st.metric("Decision Threshold", f"{result.threshold:.0%}")
    with col3:
        st.metric(
            "Groups Analyzed",
            result.group_col.replace("_", " ").title() if result.group_col else "N/A",
        )

    # Largest disparity callout
    if result.worst_disparity:
        group, metric, value = result.worst_disparity

        if audience == "governance":
            metric_name = {
                "tpr": "detection rate",
                "fpr": "false alarm rate",
                "ppv": "positive prediction accuracy",
            }.get(metric, metric)

            direction = "lower" if value < 0 else "higher"
            st.markdown(
                f"""<div style="background: #FFF3E0; border-left: 4px solid #E65100; padding: 16px; border-radius: 4px; margin: 16px 0;">
                <b style="color: #E65100;">Largest Disparity</b><br>
                The model's {metric_name} for <b>{group}</b> patients is <b>{abs(value):.1%}</b> {direction}
                than the reference group.
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div style="background: #FFF3E0; border-left: 4px solid #E65100; padding: 16px; border-radius: 4px; margin: 16px 0;">
                <b style="color: #E65100;">Largest Disparity</b><br>
                {metric.upper()} for <b>{group}</b>: Δ = {value:+.3f} vs reference
                </div>""",
                unsafe_allow_html=True,
            )


def render_performance_section(result, df: pl.DataFrame, audience: str) -> None:
    """Render overall performance section."""
    section_content = get_section_content("overall_performance", audience, {})

    render_semantic_heading(section_content["title"], level=2, id="performance")
    st.markdown(section_content["description"])

    # Performance metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    # Get metrics from result
    metrics_row = result.metrics_df.filter(pl.col("group") == "_overall").head(1)

    if len(metrics_row) > 0:
        row = metrics_row.to_dicts()[0]

        with col1:
            auroc_display = get_metric_display("auroc", row.get("auroc"), audience)
            st.metric(auroc_display["name"], auroc_display["value"])
            if audience == "governance":
                st.caption("Higher is better (max 1.0)")

        with col2:
            tpr_display = get_metric_display("tpr", row.get("tpr"), audience)
            st.metric(tpr_display["name"], tpr_display["value"])
            if audience == "governance":
                st.caption("% of cases correctly caught")

        with col3:
            fpr_display = get_metric_display("fpr", row.get("fpr"), audience)
            st.metric(fpr_display["name"], fpr_display["value"])
            if audience == "governance":
                st.caption("% of false alarms")

        with col4:
            ppv_display = get_metric_display("ppv", row.get("ppv"), audience)
            st.metric(ppv_display["name"], ppv_display["value"])
            if audience == "governance":
                st.caption("% of flags that are correct")


def render_fairness_section(result, df: pl.DataFrame, group_cols: list[str], audience: str) -> None:
    """Render fairness analysis section."""
    section_content = get_section_content("fairness_summary", audience, {})

    render_semantic_heading(section_content["title"], level=2, id="fairness")
    st.markdown(section_content["description"])

    from faircareai.visualization.plots import create_forest_plot

    # Metric selector
    if audience == "data_scientist":
        metric_options = ["tpr", "fpr", "ppv", "npv"]
        selected_metric = st.selectbox(
            "Select metric to analyze",
            options=metric_options,
            format_func=lambda x: x.upper(),
        )
    else:
        metric_options = {
            "tpr": "Detection Rate (Sensitivity)",
            "fpr": "False Alarm Rate",
            "ppv": "Positive Flag Accuracy",
        }
        selected_metric = st.selectbox(
            "What would you like to examine?",
            options=list(metric_options.keys()),
            format_func=lambda x: metric_options[x],
        )

    # Show inline definition for governance
    if audience == "governance":
        render_inline_definition(selected_metric, audience)

    st.markdown("---")

    # Forest plots for each demographic attribute
    for attr in group_cols:
        render_semantic_heading(
            f"By {attr.replace('_', ' ').title()}",
            level=3,
        )

        attr_metrics = result.metrics_df.filter(
            (pl.col("attribute") == attr) & (pl.col("group") != "_overall")
        )

        if len(attr_metrics) > 0:
            # Add CI columns for forest plot
            ci_lower_col = f"{selected_metric}_ci_lower"
            ci_upper_col = f"{selected_metric}_ci_upper"

            if ci_lower_col in attr_metrics.columns and ci_upper_col in attr_metrics.columns:
                metrics_with_ci = attr_metrics.with_columns(
                    [
                        pl.col(ci_lower_col).alias("ci_lower"),
                        pl.col(ci_upper_col).alias("ci_upper"),
                    ]
                )
            else:
                # Fallback: create dummy CI columns
                metrics_with_ci = attr_metrics.with_columns(
                    [
                        pl.col(selected_metric).alias("ci_lower"),
                        pl.col(selected_metric).alias("ci_upper"),
                    ]
                )

            # Create forest plot
            if audience == "governance":
                subtitle = "Comparison of groups to the reference line"
            else:
                subtitle = "95% CI shown. Ghosted bars indicate small sample sizes."

            fig = create_forest_plot(
                metrics_with_ci,
                metric=selected_metric,
                title=f"{selected_metric.upper()} by {attr.replace('_', ' ').title()}",
                subtitle=subtitle,
                show_safe_zone=selected_metric in ["tpr", "ppv", "npv"],
            )

            # Generate alt text
            alt_text = generate_chart_alt_text(
                "forest_plot",
                {
                    "n_groups": len(attr_metrics),
                    "metric": selected_metric,
                    "attribute": attr,
                    "reference_group": "overall",
                    "range_min": attr_metrics[selected_metric].min(),
                    "range_max": attr_metrics[selected_metric].max(),
                    "flagged_count": len(
                        result.disparities_df.filter(
                            (pl.col("attribute") == attr)
                            & (pl.col("metric") == selected_metric)
                            & (pl.col("status") == "FAIL")
                        )
                    ),
                },
            )

            accessible_plotly_chart(fig, f"forest_{attr}", alt_text)

            # Disparity summary
            attr_disparities = result.disparities_df.filter(
                (pl.col("attribute") == attr) & (pl.col("metric") == selected_metric)
            )

            if len(attr_disparities) > 0:
                flagged = attr_disparities.filter(pl.col("status") == "FAIL")
                review = attr_disparities.filter(pl.col("status") == "WARN")

                if len(flagged) > 0:
                    if audience == "governance":
                        st.info(f"{len(flagged)} group(s) outside configured threshold")
                    else:
                        st.info(f"{len(flagged)} group comparison(s) outside threshold")
                if len(review) > 0:
                    if audience == "governance":
                        st.info(f"{len(review)} group(s) near configured threshold")
                    else:
                        st.info(f"{len(review)} group comparison(s) near threshold")

        st.markdown("---")


def render_analysis_page():
    """Render the main analysis page."""
    render_skip_link()

    # Check for data
    if "uploaded_data" not in st.session_state:
        st.warning("No data loaded. Please upload data first.")
        if st.button("Go to Upload Page"):
            st.switch_page("pages/1_upload.py")
        return

    df = st.session_state["uploaded_data"]
    group_cols = st.session_state.get("selected_demographic_cols", [])

    if not group_cols:
        st.warning("No demographic columns selected. Please select columns first.")
        if st.button("Go to Upload Page"):
            st.switch_page("pages/1_upload.py")
        return

    # Header
    render_semantic_heading("Fairness Analysis", level=1, id="page-title")
    st.caption(f"Step 2 of 4 | {GOVERNANCE_DISCLAIMER_SHORT}")

    # Sidebar: Glossary and configuration
    with st.sidebar:
        st.markdown("### Configuration")

        threshold = st.slider(
            "Decision Threshold",
            0.0,
            1.0,
            0.5,
            0.01,
            help="Classification threshold for binary predictions",
        )

        st.divider()

        render_glossary_sidebar()

    # Audience toggle
    st.markdown("---")
    audience = render_audience_toggle()
    st.markdown("---")

    # Run audit
    result, audit = run_audit(df, threshold, group_cols)
    announce_status_change("Analysis complete")

    # Render sections
    render_executive_summary(result, audience)
    st.markdown("---")
    render_performance_section(result, df, audience)
    st.markdown("---")
    render_fairness_section(result, df, group_cols, audience)

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back to Upload", use_container_width=True):
            st.switch_page("pages/1_upload.py")

    with col2:
        if st.button("Continue to Governance Report", type="primary", use_container_width=True):
            st.session_state["audit_result"] = result
            st.session_state["audit_threshold"] = threshold
            st.switch_page("pages/3_governance.py")


# Run the page
render_analysis_page()
