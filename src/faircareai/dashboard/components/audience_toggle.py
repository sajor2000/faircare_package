"""
Audience Mode Toggle for Dual-Persona UX

Supports switching between Data Scientist and Governance Committee views
with appropriate content adaptation.
"""

from typing import Any

import streamlit as st


def get_audience_mode() -> str:
    """Get current audience mode from session state.

    Returns:
        "data_scientist" or "governance"
    """
    value = st.session_state.get("audience_mode")
    if isinstance(value, str) and value in ("data_scientist", "governance"):
        return value
    return "governance"


def set_audience_mode(mode: str) -> None:
    """Set audience mode in session state.

    Args:
        mode: "data_scientist" or "governance"
    """
    if mode in ("data_scientist", "governance"):
        st.session_state["audience_mode"] = mode


def render_audience_toggle() -> str:
    """Render audience mode toggle in Streamlit.

    Returns:
        Current audience mode.
    """
    current = get_audience_mode()

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Governance View",
            type="primary" if current == "governance" else "secondary",
            use_container_width=True,
            help="Plain-language explanations for clinical stakeholders",
        ):
            set_audience_mode("governance")
            st.rerun()

    with col2:
        if st.button(
            "Technical View",
            type="primary" if current == "data_scientist" else "secondary",
            use_container_width=True,
            help="Detailed statistical information for data scientists",
        ):
            set_audience_mode("data_scientist")
            st.rerun()

    return get_audience_mode()


def render_audience_toggle_pills() -> str:
    """Render audience toggle as segmented pills.

    Returns:
        Current audience mode.
    """
    current = get_audience_mode()

    # CSS for pill-style toggle
    st.markdown(
        """
        <style>
        .audience-pills {
            display: flex;
            gap: 0;
            margin: 16px 0;
            border: 1px solid #E0E0E0;
            border-radius: 24px;
            overflow: hidden;
            width: fit-content;
        }
        .audience-pill {
            padding: 8px 20px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            border: none;
            background: #FFFFFF;
            color: #666666;
            transition: all 0.2s ease;
        }
        .audience-pill:hover {
            background: #F5F5F5;
        }
        .audience-pill.active {
            background: #0072B2;
            color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use radio for actual selection
    mode = st.radio(
        "Audience Mode",
        options=["governance", "data_scientist"],
        format_func=lambda x: "Governance" if x == "governance" else "Technical",
        horizontal=True,
        label_visibility="collapsed",
        key="audience_toggle_radio",
    )

    if mode != current:
        set_audience_mode(mode)

    return mode


def get_metric_display(
    metric: str,
    value: Any,
    audience: str,
    ci: tuple | None = None,
    p_value: float | None = None,
    status: str | None = None,
) -> dict:
    """Get metric display formatted for audience.

    Args:
        metric: Metric name.
        value: Metric value.
        audience: "data_scientist" or "governance".
        ci: Optional confidence interval (lower, upper).
        p_value: Optional p-value.
        status: Optional status (pass, warn, fail).

    Returns:
        Dictionary with formatted display components.
    """
    # Metric name formatting
    metric_names = {
        "data_scientist": {
            "auroc": "AUROC",
            "auprc": "AUPRC",
            "tpr": "TPR (Sensitivity)",
            "fpr": "FPR",
            "ppv": "PPV (Precision)",
            "npv": "NPV",
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "brier_score": "Brier Score",
            "calibration_slope": "Calibration Slope",
        },
        "governance": {
            "auroc": "Model Discrimination",
            "auprc": "Precision-Recall Performance",
            "tpr": "Detection Rate",
            "fpr": "False Alarm Rate",
            "ppv": "Positive Flag Accuracy",
            "npv": "Negative Flag Accuracy",
            "sensitivity": "Detection Rate",
            "specificity": "True Negative Rate",
            "brier_score": "Probability Accuracy",
            "calibration_slope": "Confidence Accuracy",
        },
    }

    display_name = metric_names.get(audience, {}).get(metric, metric.upper())

    # Value formatting
    if isinstance(value, (int, float)):
        if metric in ("brier_score", "calibration_slope"):
            formatted_value = f"{value:.3f}"
        elif abs(value) < 0.01:
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = f"{value:.1%}"
    else:
        formatted_value = str(value)

    result = {
        "name": display_name,
        "value": formatted_value,
        "raw_value": value,
    }

    # Add technical details for data scientists
    if audience == "data_scientist":
        if ci:
            result["ci"] = f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]"
        if p_value is not None:
            if p_value < 0.001:
                result["p_value"] = "p < 0.001"
            else:
                result["p_value"] = f"p = {p_value:.3f}"

    # Add interpretation for governance
    if audience == "governance" and status:
        interpretations = {
            "pass": "Within configured threshold",
            "warn": "Near configured threshold",
            "fail": "Outside configured threshold",
        }
        result["interpretation"] = interpretations.get(status, "")

    return result


def get_section_content(
    section: str,
    audience: str,
    data: dict,
) -> dict:
    """Get section content adapted for audience.

    Args:
        section: Section identifier.
        audience: Target audience.
        data: Section data.

    Returns:
        Dictionary with section content.
    """
    sections = {
        "overall_performance": {
            "data_scientist": {
                "title": "Overall Model Performance (TRIPOD+AI Metrics)",
                "description": (
                    "Comprehensive performance evaluation following TRIPOD+AI guidelines. "
                    "Includes discrimination (AUROC, AUPRC), calibration (Brier score, slope), "
                    "and classification metrics at the specified threshold."
                ),
            },
            "governance": {
                "title": "How Well Does the Model Perform?",
                "description": (
                    "These metrics show how accurate the model is overall. "
                    "Think of it like a report card for the model."
                ),
            },
        },
        "fairness_summary": {
            "data_scientist": {
                "title": "Fairness Metrics by Protected Attribute",
                "description": (
                    "Equalized odds, demographic parity, and calibration metrics "
                    "computed across demographic subgroups with bootstrap confidence intervals."
                ),
            },
            "governance": {
                "title": "Is the Model Fair to All Patient Groups?",
                "description": (
                    "We checked whether the model works equally well for different "
                    "groups of patients. Here's what we found."
                ),
            },
        },
        "key_finding": {
            "data_scientist": {
                "template": (
                    "**{group}** shows {metric} = {value:.3f} "
                    "(95% CI: {ci_lower:.3f}-{ci_upper:.3f}), "
                    "diff from reference: {diff:+.3f} ({status})"
                ),
            },
            "governance": {
                "template": (
                    "The model's accuracy differs for **{group}** patients. {interpretation}"
                ),
            },
        },
        "disparity_explanation": {
            "data_scientist": {
                "tpr_low": (
                    "TPR disparity indicates the model misses more positive cases "
                    "in this group (lower sensitivity)."
                ),
                "tpr_high": (
                    "TPR disparity indicates higher detection rate in this group "
                    "(potential over-prediction)."
                ),
                "fpr_high": (
                    "FPR disparity indicates more false positives in this group "
                    "(lower specificity)."
                ),
            },
            "governance": {
                "tpr_low": (
                    "Patients in this group who have the condition are less likely "
                    "to be correctly identified by the model."
                ),
                "tpr_high": (
                    "Patients in this group are more likely to be flagged as high-risk, "
                    "which could lead to more testing or interventions."
                ),
                "fpr_high": (
                    "Patients in this group who don't have the condition are more "
                    "likely to receive a false alarm."
                ),
            },
        },
    }

    section_config = sections.get(section, {})
    audience_config = section_config.get(audience, section_config.get("governance", {}))

    return audience_config


def render_dual_content(
    technical_content: str,
    plain_content: str,
    audience: str | None = None,
) -> None:
    """Render content appropriate for current audience.

    Args:
        technical_content: Content for data scientists.
        plain_content: Content for governance committee.
        audience: Override audience mode.
    """
    if audience is None:
        audience = get_audience_mode()

    if audience == "data_scientist":
        st.markdown(technical_content)
    else:
        st.markdown(plain_content)


def render_with_detail_toggle(
    summary: str,
    detail: str,
    detail_label: str = "Show technical details",
) -> None:
    """Render summary with optional technical details.

    Always shows summary, with expandable technical details.

    Args:
        summary: Main summary content (shown to everyone).
        detail: Technical details (expandable).
        detail_label: Label for detail expander.
    """
    st.markdown(summary)

    with st.expander(detail_label, expanded=False):
        st.markdown(detail)
