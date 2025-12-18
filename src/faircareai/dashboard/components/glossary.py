"""
Metric Glossary for Dual-Audience Communication

Provides technical and plain-language explanations for all metrics.
Supports Data Scientist and Governance Committee personas.
"""

from typing import Any

import streamlit as st

# Comprehensive metric glossary with dual-audience explanations
METRIC_GLOSSARY: dict[str, dict[str, Any]] = {
    # Performance Metrics
    "auroc": {
        "name": "AUROC",
        "full_name": "Area Under the Receiver Operating Characteristic Curve",
        "technical": (
            "AUROC measures the model's ability to discriminate between positive and "
            "negative cases across all possible classification thresholds. Ranges from "
            "0.5 (random) to 1.0 (perfect). Equivalent to the probability that a randomly "
            "chosen positive case ranks higher than a randomly chosen negative case."
        ),
        "plain": (
            "How well can the model tell apart patients who will have the outcome "
            "from those who won't? A score of 0.5 is like flipping a coin, while 1.0 "
            "would be perfect prediction."
        ),
        "example": "An AUROC of 0.80 means the model correctly ranks patients 80% of the time.",
        "why_it_matters": (
            "This is the standard measure of how well the model distinguishes between "
            "patients at high and low risk. Higher is better."
        ),
        "thresholds": {"excellent": 0.9, "good": 0.8, "acceptable": 0.7},
    },
    "auprc": {
        "name": "AUPRC",
        "full_name": "Area Under the Precision-Recall Curve",
        "technical": (
            "AUPRC summarizes the precision-recall trade-off across thresholds. "
            "More informative than AUROC when outcome prevalence is low. "
            "Baseline equals the outcome prevalence."
        ),
        "plain": (
            "Of the patients the model flags as high-risk, how many actually have the "
            "outcome? This metric is especially useful when the outcome is rare."
        ),
        "example": "AUPRC of 0.6 with 10% prevalence is quite good; with 50% prevalence, less so.",
        "why_it_matters": (
            "Particularly important when the outcome is uncommon. Tells you about "
            "the quality of the model's high-risk predictions."
        ),
        "thresholds": {"depends_on_prevalence": True},
    },
    "brier_score": {
        "name": "Brier Score",
        "full_name": "Brier Score",
        "technical": (
            "Mean squared error between predicted probabilities and binary outcomes. "
            "Ranges from 0 (perfect) to 1 (worst). Decomposable into calibration, "
            "refinement, and uncertainty components."
        ),
        "plain": (
            "How close are the model's predicted probabilities to what actually happens? "
            "Lower scores mean more accurate probability estimates."
        ),
        "example": "A Brier score of 0.15 indicates reasonably well-calibrated predictions.",
        "why_it_matters": (
            "Important for trusting the actual probability values, not just the rankings. "
            "Matters when you need to know 'how likely' not just 'more or less likely'."
        ),
        "thresholds": {"excellent": 0.1, "good": 0.2, "acceptable": 0.25},
    },
    "calibration_slope": {
        "name": "Calibration Slope",
        "full_name": "Calibration Slope",
        "technical": (
            "Slope from regressing observed outcomes on log-odds of predicted probabilities. "
            "Ideal value is 1.0. Slope < 1 indicates overconfident predictions "
            "(extremes too extreme); slope > 1 indicates underconfident predictions."
        ),
        "plain": (
            "Are the model's confidence levels accurate? A slope of 1.0 means the "
            "model is appropriately confident. Below 1 means it's overconfident."
        ),
        "example": "Slope of 0.8 means the model is too extreme in its predictions.",
        "why_it_matters": (
            "Tells you whether to trust the actual probability numbers or if they "
            "need adjustment before clinical use."
        ),
        "thresholds": {"ideal": 1.0, "acceptable_range": (0.8, 1.2)},
    },
    # Classification Metrics
    "sensitivity": {
        "name": "Sensitivity",
        "full_name": "Sensitivity (True Positive Rate)",
        "technical": (
            "Proportion of actual positive cases correctly identified. "
            "TPR = TP / (TP + FN). Also called recall or hit rate. "
            "Threshold-dependent metric."
        ),
        "plain": (
            "Of all patients who actually have the condition, what percentage does "
            "the model correctly identify as high-risk?"
        ),
        "example": "90% sensitivity means 9 out of 10 patients with the condition are flagged.",
        "why_it_matters": (
            "Critical when missing a case is dangerous. High sensitivity means "
            "fewer patients with the condition are missed."
        ),
        "alias": "tpr",
        "thresholds": {"high": 0.9, "moderate": 0.8},
    },
    "tpr": {
        "name": "TPR",
        "full_name": "True Positive Rate",
        "technical": "Same as sensitivity. TPR = TP / (TP + FN).",
        "plain": "Same as sensitivity - the proportion of positive cases correctly identified.",
        "example": "See sensitivity.",
        "why_it_matters": "See sensitivity.",
        "alias": "sensitivity",
    },
    "specificity": {
        "name": "Specificity",
        "full_name": "Specificity (True Negative Rate)",
        "technical": (
            "Proportion of actual negative cases correctly identified. "
            "TNR = TN / (TN + FP). Also called selectivity."
        ),
        "plain": (
            "Of all patients who don't have the condition, what percentage does "
            "the model correctly identify as low-risk?"
        ),
        "example": "85% specificity means 85% of healthy patients are not falsely flagged.",
        "why_it_matters": (
            "Important when false alarms are costly. High specificity means "
            "fewer unnecessary interventions for healthy patients."
        ),
        "thresholds": {"high": 0.9, "moderate": 0.8},
    },
    "fpr": {
        "name": "FPR",
        "full_name": "False Positive Rate",
        "technical": (
            "Proportion of negative cases incorrectly classified as positive. "
            "FPR = FP / (FP + TN) = 1 - Specificity."
        ),
        "plain": (
            "Of all patients without the condition, what percentage are incorrectly "
            "flagged as high-risk?"
        ),
        "example": "10% FPR means 1 in 10 healthy patients get a false alarm.",
        "why_it_matters": (
            "High FPR means more patients receive unnecessary worry or interventions. "
            "Lower is generally better."
        ),
        "thresholds": {"low": 0.1, "moderate": 0.2},
    },
    "ppv": {
        "name": "PPV",
        "full_name": "Positive Predictive Value (Precision)",
        "technical": (
            "Proportion of positive predictions that are correct. "
            "PPV = TP / (TP + FP). Highly dependent on prevalence."
        ),
        "plain": (
            "When the model flags someone as high-risk, how often are they actually "
            "at risk? This is what a positive result means in practice."
        ),
        "example": "PPV of 40% means 4 in 10 flagged patients actually have the condition.",
        "why_it_matters": (
            "Tells clinicians what a positive prediction actually means for their "
            "patient. Very important for shared decision-making."
        ),
        "thresholds": {"depends_on_prevalence": True},
    },
    "npv": {
        "name": "NPV",
        "full_name": "Negative Predictive Value",
        "technical": (
            "Proportion of negative predictions that are correct. "
            "NPV = TN / (TN + FN). Also prevalence-dependent."
        ),
        "plain": (
            "When the model says someone is low-risk, how often is that correct? "
            "This is what a negative result means in practice."
        ),
        "example": "NPV of 95% means 95% of patients flagged as low-risk truly are.",
        "why_it_matters": (
            "Helps determine how much to trust a 'no risk' prediction. "
            "Important for ruling out conditions."
        ),
        "thresholds": {"depends_on_prevalence": True},
    },
    # Fairness Metrics
    "demographic_parity": {
        "name": "Demographic Parity",
        "full_name": "Demographic Parity (Statistical Parity)",
        "technical": (
            "Requires equal positive prediction rates across groups: "
            "P(Ŷ=1|A=a) = P(Ŷ=1|A=b) for all groups a, b. "
            "Does not consider actual outcomes."
        ),
        "plain": (
            "Are patients from different groups equally likely to be flagged as "
            "high-risk by the model? This doesn't consider whether the flagging "
            "is correct, just whether it's equal."
        ),
        "example": (
            "If 20% of Group A is flagged but 30% of Group B, there's a 10% demographic parity gap."
        ),
        "why_it_matters": (
            "Can reveal if the model treats groups differently in terms of "
            "prediction rates, regardless of accuracy."
        ),
        "controversy": (
            "Not always the right goal - if true risk differs between groups, "
            "equal prediction rates would mean unequal accuracy."
        ),
    },
    "equalized_odds": {
        "name": "Equalized Odds",
        "full_name": "Equalized Odds",
        "technical": (
            "Requires equal TPR and FPR across groups: "
            "P(Ŷ=1|Y=y,A=a) = P(Ŷ=1|Y=y,A=b) for y ∈ {0,1}. "
            "Considers model errors conditional on true outcome."
        ),
        "plain": (
            "Is the model equally accurate for all groups? Specifically, among "
            "patients who truly have the condition, are all groups equally likely "
            "to be correctly identified? And among those without, are all groups "
            "equally likely to be correctly cleared?"
        ),
        "example": (
            "If sensitivity is 85% for Group A but 70% for Group B, there's a "
            "15% equalized odds gap in TPR."
        ),
        "why_it_matters": (
            "Reveals if the model's errors fall disproportionately on certain groups. "
            "Often considered the most relevant fairness metric for clinical models."
        ),
    },
    "equal_opportunity": {
        "name": "Equal Opportunity",
        "full_name": "Equal Opportunity (Equality of True Positive Rates)",
        "technical": (
            "Requires equal TPR across groups: P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b). "
            "Relaxation of equalized odds focusing only on positive cases."
        ),
        "plain": (
            "Among patients who truly have the condition, is every group equally "
            "likely to be correctly identified? Focuses on not missing sick patients "
            "in any group."
        ),
        "example": (
            "90% of sick patients in Group A are caught vs 75% in Group B = "
            "15% equal opportunity gap."
        ),
        "why_it_matters": (
            "Critical when the consequence of missing a case is severe. "
            "Ensures no group is systematically under-detected."
        ),
    },
    "predictive_parity": {
        "name": "Predictive Parity",
        "full_name": "Predictive Parity (Equal PPV)",
        "technical": (
            "Requires equal PPV across groups: "
            "P(Y=1|Ŷ=1,A=a) = P(Y=1|Ŷ=1,A=b). "
            "Ensures predictions mean the same thing across groups."
        ),
        "plain": (
            "When the model flags someone as high-risk, does that mean the same "
            "thing regardless of their demographic group? A flag should carry "
            "equal weight for everyone."
        ),
        "example": (
            "If a positive flag means 40% true risk for Group A but 60% for "
            "Group B, there's a 20% predictive parity gap."
        ),
        "why_it_matters": (
            "Important for shared decision-making. Clinicians and patients need "
            "to know what a prediction actually means."
        ),
    },
    "calibration_parity": {
        "name": "Calibration Parity",
        "full_name": "Calibration Parity (Equal Calibration Across Groups)",
        "technical": (
            "Requires equal calibration across groups: "
            "E[Y|s(X)=p,A=a] = p for all groups a. "
            "Predicted probabilities should match observed frequencies in each group."
        ),
        "plain": (
            "When the model says there's a 30% chance for patients in any group, "
            "does the actual rate match? The probability estimates should be "
            "equally accurate for everyone."
        ),
        "example": (
            "If '50% risk' predictions are correct 50% of the time for Group A "
            "but only 35% for Group B, calibration differs."
        ),
        "why_it_matters": (
            "Probability estimates guide clinical decisions. They need to be "
            "trustworthy regardless of patient background."
        ),
    },
}


def get_metric_explanation(
    metric: str,
    audience: str = "governance",
) -> dict[str, Any]:
    """Get metric explanation appropriate for audience.

    Args:
        metric: Metric key from METRIC_GLOSSARY.
        audience: "data_scientist" or "governance".

    Returns:
        Dictionary with explanation appropriate for audience.
    """
    metric_info = METRIC_GLOSSARY.get(metric.lower())
    if metric_info is None:
        return {
            "name": metric,
            "explanation": f"Information for {metric} not available.",
        }

    if audience == "data_scientist":
        return {
            "name": metric_info.get("name", metric),
            "full_name": metric_info.get("full_name", ""),
            "explanation": metric_info.get("technical", ""),
            "thresholds": metric_info.get("thresholds", {}),
        }
    else:
        return {
            "name": metric_info.get("name", metric),
            "explanation": metric_info.get("plain", ""),
            "example": metric_info.get("example", ""),
            "why_it_matters": metric_info.get("why_it_matters", ""),
        }


def render_glossary_tooltip(
    metric: str,
    audience: str = "governance",
) -> str:
    """Render a tooltip-style explanation for a metric.

    Args:
        metric: Metric key.
        audience: Target audience.

    Returns:
        HTML string for tooltip content.
    """
    info = get_metric_explanation(metric, audience)

    if audience == "data_scientist":
        return f"""
        <div style="max-width: 400px;">
            <strong>{info.get("full_name", info["name"])}</strong>
            <p style="margin: 8px 0; font-size: 13px; color: #333;">
                {info["explanation"]}
            </p>
        </div>
        """
    else:
        example = info.get("example", "")
        why = info.get("why_it_matters", "")

        return f"""
        <div style="max-width: 400px;">
            <strong>{info["name"]}</strong>
            <p style="margin: 8px 0; font-size: 14px; color: #333;">
                {info["explanation"]}
            </p>
            {f'<p style="font-size: 13px; color: #666; font-style: italic;">Example: {example}</p>' if example else ""}
            {f'<p style="font-size: 13px; color: #0072B2;"><strong>Why it matters:</strong> {why}</p>' if why else ""}
        </div>
        """


def render_glossary_sidebar() -> None:
    """Render full glossary in Streamlit sidebar."""
    audience = st.session_state.get("audience_mode", "governance")

    st.sidebar.markdown("### Metric Glossary")

    # Group metrics by category
    categories = {
        "Performance": ["auroc", "auprc", "brier_score", "calibration_slope"],
        "Classification": ["sensitivity", "specificity", "fpr", "ppv", "npv"],
        "Fairness": [
            "demographic_parity",
            "equalized_odds",
            "equal_opportunity",
            "predictive_parity",
            "calibration_parity",
        ],
    }

    for category, metrics in categories.items():
        with st.sidebar.expander(category, expanded=False):
            for metric in metrics:
                if metric in METRIC_GLOSSARY:
                    info = get_metric_explanation(metric, audience)
                    st.markdown(f"**{info['name']}**")
                    st.markdown(
                        f"<p style='font-size: 12px; color: #666;'>{info['explanation']}</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")


def render_inline_definition(
    metric: str,
    audience: str = "governance",
) -> None:
    """Render inline definition with expandable details.

    Args:
        metric: Metric key.
        audience: Target audience.
    """
    info = get_metric_explanation(metric, audience)

    with st.expander(f"What is {info['name']}?", expanded=False):
        st.markdown(info["explanation"])

        if audience == "governance":
            if info.get("example"):
                st.markdown(f"*Example: {info['example']}*")
            if info.get("why_it_matters"):
                st.info(f"**Why it matters:** {info['why_it_matters']}")
