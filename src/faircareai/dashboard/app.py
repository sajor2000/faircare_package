"""
FairCareAI Streamlit Dashboard - Multi-Page Application

A WCAG 2.1 AA compliant fairness auditing dashboard with dual-audience support.

Personas:
- Governance Committee: Plain-language explanations, high-level findings
- Data Scientists: Technical details, statistical metrics, export options

Design: NYT Editorial Aesthetic with colorblind-safe Okabe-Ito palette.
"""

from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="FairCare Equity Audit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from faircareai.dashboard.components.accessibility import (
    create_skip_link,
    inject_accessibility_css,
)
from faircareai.visualization.themes import inject_streamlit_css

# Inject CSS
inject_streamlit_css()
inject_accessibility_css()


def get_pages_dir() -> Path:
    """Get the pages directory path."""
    return Path(__file__).parent / "pages"


def main() -> None:
    """Main dashboard application with multi-page navigation."""
    pages_dir = get_pages_dir()

    # Define pages using st.navigation
    upload_page = st.Page(
        str(pages_dir / "1_upload.py"),
        title="Data Upload",
        icon="📤",
        default=True,
    )
    analysis_page = st.Page(
        str(pages_dir / "2_analysis.py"),
        title="Analysis",
        icon="📊",
    )
    governance_page = st.Page(
        str(pages_dir / "3_governance.py"),
        title="Governance Report",
        icon="📋",
    )
    settings_page = st.Page(
        str(pages_dir / "4_settings.py"),
        title="Settings",
        icon="⚙️",
    )

    # Create navigation with grouped pages
    pg = st.navigation(
        {
            "Workflow": [upload_page, analysis_page, governance_page],
            "Configuration": [settings_page],
        },
        position="sidebar",
    )

    # Add skip link for accessibility
    st.markdown(create_skip_link(), unsafe_allow_html=True)

    # Sidebar header
    with st.sidebar:
        st.markdown("### FairCareAI")
        st.caption("Algorithmic Fairness Audit")
        st.divider()

    # Run the selected page
    pg.run()


def launch() -> None:
    """Launch the dashboard from command line."""
    import subprocess
    import sys

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__))],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to launch dashboard: {e}") from e


# Legacy support: Allow running the old single-page app
def run_legacy_app() -> None:
    """Run the legacy single-page application.

    This function preserves the original single-page dashboard for
    backwards compatibility and testing.
    """

    from faircareai.dashboard.components.accessibility import render_skip_link
    from faircareai.dashboard.components.audience_toggle import render_audience_toggle
    from faircareai.visualization.themes import (
        GOVERNANCE_DISCLAIMER_SHORT,
        render_scorecard_html,
        render_status_badge,
    )

    render_skip_link()

    @st.cache_data
    def load_demo_data() -> pl.DataFrame:
        """Load and cache synthetic demo data."""
        from faircareai.data.synthetic import generate_icu_mortality_data

        return generate_icu_mortality_data(n_samples=2000, seed=42)

    @st.cache_data
    def run_audit(_df: pl.DataFrame, threshold: float, group_cols: list[str]) -> Any:
        """Run and cache audit results."""
        from faircareai.core.audit import FairCareAudit
        from faircareai.core.config import FairnessConfig

        config = FairnessConfig(model_name="ICU Mortality Prediction Model")
        audit = FairCareAudit(
            _df,
            pred_col="y_prob",
            target_col="y_true",
            config=config,
            threshold=threshold,
        )
        # Accept all detected attributes for demo
        for i, attr in enumerate(audit._suggestions):
            if attr["column"] in group_cols:
                audit.accept_suggested_attributes([i])
        result = audit.run()
        return result, audit

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### Configuration")

        data_source = st.radio(
            "Data Source",
            ["Demo: ICU Mortality", "Upload CSV"],
            index=0,
        )

        threshold = st.slider(
            "Decision Threshold",
            0.0,
            1.0,
            0.5,
            0.01,
            help="Classification threshold for binary predictions",
        )

        st.divider()

        # Audience toggle
        render_audience_toggle()

        st.divider()

        # Metric selector
        st.selectbox(
            "Primary Metric",
            ["tpr", "fpr", "ppv", "npv", "accuracy"],
            format_func=lambda x: {
                "tpr": "Sensitivity (TPR)",
                "fpr": "False Positive Rate",
                "ppv": "Precision (PPV)",
                "npv": "Negative Predictive Value",
                "accuracy": "Accuracy",
            }.get(x, x),
        )

        st.divider()
        st.caption("FairCareAI v0.2.0")

    # Load data
    if data_source == "Demo: ICU Mortality":
        df = load_demo_data()
        group_cols = ["race_ethnicity", "insurance", "language"]
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.info(
                "Upload a CSV file with columns: `y_true`, `y_pred` (or `y_prob`), and demographic attributes."
            )
            return
        df = pl.read_csv(uploaded)
        available_cols = [c for c in df.columns if c not in ["y_true", "y_pred", "y_prob"]]
        group_cols = st.multiselect(
            "Select demographic columns", available_cols, default=available_cols[:3]
        )
        if not group_cols:
            st.warning("Please select at least one demographic column.")
            return

    # Run audit
    result, audit = run_audit(df, threshold, group_cols)

    # Main content header
    st.markdown(
        "<h1 style='margin-bottom: 0;'>FairCare Equity Audit</h1>",
        unsafe_allow_html=True,
    )
    st.caption(f"Model: {result.model_name} | Audit Date: {result.audit_date}")

    # Governance disclaimer
    st.caption(f"ℹ️ {GOVERNANCE_DISCLAIMER_SHORT}")

    # Status badge
    if result.fail_count > 0:
        status_text = "REVIEW SUGGESTED"
        status_type = "fail"
    elif result.warn_count > 0:
        status_text = "CONSIDERATIONS NOTED"
        status_type = "warn"
    else:
        status_text = "NO FLAGS RAISED"
        status_type = "pass"

    st.markdown(
        render_status_badge(status_type, f"<b>{status_text}</b>"),
        unsafe_allow_html=True,
    )

    st.markdown(
        render_scorecard_html(result.pass_count, result.warn_count, result.fail_count),
        unsafe_allow_html=True,
    )

    # Summary stats
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


if __name__ == "__main__":
    main()
