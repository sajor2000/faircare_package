"""
FairCareAI Fairness Decision Tree Module

Presents fairness metric options by use case based on healthcare AI literature.
Includes documentation of the impossibility theorem and its implications.

Key Points:
- No single metric is universally "correct" (see impossibility theorem)
- Metric choice involves value trade-offs
- Options are presented; your organization decides what applies to your context
"""

from typing import Any

from faircareai.core.config import FairnessMetric, UseCaseType

# === Decision Tree Configuration ===
# Based on healthcare AI fairness literature and clinical considerations

DECISION_TREE: dict[UseCaseType, dict[str, Any]] = {
    UseCaseType.INTERVENTION_TRIGGER: {
        "recommended": FairnessMetric.EQUALIZED_ODDS,
        "rationale": (
            "When the model triggers clinical interventions, equalizing both true positive "
            "rates (ensuring all groups receive equal benefit from correct predictions) and "
            "false positive rates (ensuring all groups bear equal burden of false alarms) "
            "is typically appropriate. This balances benefit distribution with harm minimization."
        ),
        "alternatives": [FairnessMetric.EQUAL_OPPORTUNITY],
        "alternative_rationale": (
            "Equal Opportunity (equalizing TPR only) may be preferred when false positives "
            "are low-cost interventions (e.g., additional screening) and missing cases is "
            "the primary concern."
        ),
        "contraindicated": [FairnessMetric.DEMOGRAPHIC_PARITY],
        "contraindication_reason": (
            "Demographic parity ignores base rate differences and may result in "
            "suboptimal clinical outcomes for groups with different disease prevalence."
        ),
        "clinical_examples": [
            "Sepsis early warning systems",
            "Acute kidney injury prediction",
            "Fall risk assessment triggering nurse check-ins",
        ],
    },
    UseCaseType.RISK_COMMUNICATION: {
        "recommended": FairnessMetric.CALIBRATION,
        "rationale": (
            "When predictions are communicated directly to patients or inform shared "
            "decision-making, calibration ensures the predicted probabilities are accurate "
            "across all groups. A patient told they have 30% risk should actually have 30% "
            "risk, regardless of their demographic group."
        ),
        "alternatives": [FairnessMetric.PREDICTIVE_PARITY],
        "alternative_rationale": (
            "Predictive Parity ensures positive predictions have equal meaning across groups, "
            "which supports trust in risk communication."
        ),
        "contraindicated": [FairnessMetric.DEMOGRAPHIC_PARITY],
        "contraindication_reason": (
            "Communicating artificially adjusted risk scores to achieve demographic parity "
            "would be deceptive and potentially harmful to patient autonomy."
        ),
        "clinical_examples": [
            "Cardiovascular risk calculators (ASCVD)",
            "Cancer risk prediction for screening decisions",
            "Genetic risk scores shared with patients",
        ],
    },
    UseCaseType.RESOURCE_ALLOCATION: {
        "recommended": FairnessMetric.DEMOGRAPHIC_PARITY,
        "rationale": (
            "When the model determines access to limited resources, demographic parity "
            "ensures proportional allocation across groups. This is especially important "
            "when historical disparities may have created unequal disease burdens that "
            "should not be perpetuated."
        ),
        "alternatives": [FairnessMetric.EQUALIZED_ODDS],
        "alternative_rationale": (
            "If the resource allocation is purely need-based and base rate differences "
            "reflect genuine medical need rather than social determinants, equalized odds "
            "may be more appropriate."
        ),
        "contraindicated": [],
        "contraindication_reason": None,
        "clinical_examples": [
            "Care management program enrollment",
            "Organ transplant waiting list prioritization",
            "Mental health resource allocation",
        ],
        "special_considerations": (
            "Resource allocation is highly context-dependent. Consider whether base rate "
            "differences reflect genuine medical need or historical inequities. Consult "
            "ethics committee for high-stakes allocation decisions."
        ),
    },
    UseCaseType.SCREENING: {
        "recommended": FairnessMetric.EQUAL_OPPORTUNITY,
        "rationale": (
            "For screening programs, ensuring equal sensitivity (true positive rate) "
            "across groups guarantees that all groups have equal chance of having their "
            "condition detected. This is especially important for serious conditions "
            "where early detection improves outcomes."
        ),
        "alternatives": [FairnessMetric.EQUALIZED_ODDS],
        "alternative_rationale": (
            "If screening has significant costs or risks (e.g., invasive follow-up tests), "
            "equalizing both TPR and FPR via equalized odds may be preferred."
        ),
        "contraindicated": [FairnessMetric.CALIBRATION],
        "contraindication_reason": (
            "Pure calibration focus may miss opportunities to improve detection rates "
            "in underserved populations where screening historically underperforms."
        ),
        "clinical_examples": [
            "Cancer screening risk stratification",
            "Diabetic retinopathy screening",
            "Depression screening in primary care",
        ],
    },
    UseCaseType.DIAGNOSIS_SUPPORT: {
        "recommended": FairnessMetric.PREDICTIVE_PARITY,
        "rationale": (
            "For diagnostic decision support, ensuring equal positive predictive value "
            "across groups means a positive prediction has the same meaning regardless "
            "of patient demographics. This supports trust in the diagnostic tool."
        ),
        "alternatives": [FairnessMetric.CALIBRATION],
        "alternative_rationale": (
            "Calibration is also important for diagnosis, ensuring predicted probabilities "
            "reflect true disease probability across groups."
        ),
        "contraindicated": [FairnessMetric.DEMOGRAPHIC_PARITY],
        "contraindication_reason": (
            "Diagnoses should reflect medical reality, not demographic quotas. "
            "Demographic parity in diagnosis could lead to over/under-diagnosis."
        ),
        "clinical_examples": [
            "AI-assisted radiology interpretation",
            "Clinical decision support for differential diagnosis",
            "Lab result interpretation algorithms",
        ],
    },
}


# === Impossibility Theorem Documentation ===

IMPOSSIBILITY_THEOREM = """
IMPOSSIBILITY THEOREM FOR FAIRNESS METRICS

Mathematical Reality:
When base rates (prevalence) differ between groups, it is mathematically impossible
to simultaneously satisfy all of the following:
1. Equal selection rates across groups (Demographic Parity)
2. Equal true positive rates across groups (Equal Opportunity)
3. Equal false positive rates across groups
4. Equal positive predictive values across groups (Predictive Parity)
5. Perfect calibration across groups

Key Implications:
- There is NO universally "fair" model when prevalence differs
- Choice of fairness metric is a VALUES decision, not a technical one
- Optimizing for one metric may worsen another
- Trade-offs must be explicitly documented and justified

Clinical Context Matters:
The "right" fairness metric depends on:
- The clinical use case and its consequences
- Who bears the costs of errors (false positives vs false negatives)
- Whether base rate differences reflect medical reality vs historical inequity
- Stakeholder values and priorities

Methodology:
FairCareAI presents metrics organized by use case. Organizations select metrics
based on their clinical context, institutional values, and governance frameworks.
The decision about which metric(s) to prioritize rests with your team.

References:
- Chouldechova A. (2017). Fair prediction with disparate impact
- Kleinberg J. et al. (2017). Inherent Trade-Offs in Algorithmic Fairness
- Corbett-Davies S. & Goel S. (2018). The Measure and Mismeasure of Fairness
"""


def get_fairness_metric_options(
    use_case: UseCaseType | None = None,
    clinical_context: str | None = None,
) -> dict[str, Any]:
    """Get fairness metric options based on use case from healthcare AI literature.

    Args:
        use_case: The clinical use case type.
        clinical_context: Optional additional context.

    Returns:
        Dict containing:
        - primary_option: Literature-based primary option for this use case
        - rationale: Context for why this metric is commonly used
        - alternatives: Other options to consider
        - tradeoff_notes: Considerations when using certain metrics
        - impossibility_note: Reminder about mathematical constraints
    """
    result: dict[str, Any] = {
        "impossibility_note": (
            "No single fairness metric is universally correct. Choice of metric "
            "involves value trade-offs. Your organization determines which "
            "trade-offs align with your clinical context and values. "
            "See get_impossibility_warning() for mathematical background."
        ),
    }

    if use_case is None:
        result["error"] = (
            "Use case type required to retrieve metric options. "
            "Available use cases: " + ", ".join(t.value for t in UseCaseType)
        )
        result["available_use_cases"] = [
            {"type": t.value, "description": _get_use_case_description(t)} for t in UseCaseType
        ]
        return result

    if use_case not in DECISION_TREE:
        result["error"] = f"Unknown use case type: {use_case}"
        return result

    tree_entry = DECISION_TREE[use_case]

    result.update(
        {
            "use_case": use_case.value,
            "primary_option": tree_entry["recommended"].value,
            "rationale": tree_entry["rationale"],
            "alternatives": [m.value for m in tree_entry.get("alternatives", [])],
            "alternative_rationale": tree_entry.get("alternative_rationale"),
            "tradeoff_notes": [m.value for m in tree_entry.get("contraindicated", [])],
            "tradeoff_context": tree_entry.get("contraindication_reason"),
            "clinical_examples": tree_entry.get("clinical_examples", []),
        }
    )

    if "special_considerations" in tree_entry:
        result["special_considerations"] = tree_entry["special_considerations"]

    return result


# Backward compatibility alias
recommend_fairness_metric = get_fairness_metric_options


def _get_use_case_description(use_case: UseCaseType) -> str:
    """Get description for a use case type."""
    descriptions = {
        UseCaseType.INTERVENTION_TRIGGER: (
            "Model predictions directly trigger clinical interventions "
            "(e.g., alerts, escalations, treatment decisions)"
        ),
        UseCaseType.RISK_COMMUNICATION: (
            "Predicted probabilities are communicated to patients or inform shared decision-making"
        ),
        UseCaseType.RESOURCE_ALLOCATION: (
            "Model determines access to limited healthcare resources "
            "(e.g., care management, specialist referrals)"
        ),
        UseCaseType.SCREENING: (
            "Model identifies patients for diagnostic testing or preventive care"
        ),
        UseCaseType.DIAGNOSIS_SUPPORT: ("Model assists clinicians in diagnostic decision-making"),
    }
    return descriptions.get(use_case, "No description available")


def get_impossibility_warning() -> str:
    """Get the full impossibility theorem documentation.

    Returns:
        Formatted string explaining the impossibility theorem.
    """
    return IMPOSSIBILITY_THEOREM


def get_metric_description(metric: FairnessMetric) -> dict[str, Any]:
    """Get detailed description of a fairness metric.

    Args:
        metric: The fairness metric to describe.

    Returns:
        Dict with metric details.
    """
    descriptions = {
        FairnessMetric.DEMOGRAPHIC_PARITY: {
            "name": "Demographic Parity",
            "also_known_as": ["Statistical Parity", "Independence"],
            "definition": (
                "Selection rates are equal across groups. "
                "P(Y_hat=1|A=a) = P(Y_hat=1|A=b) for all groups a, b"
            ),
            "intuition": (
                "All groups receive positive predictions at the same rate, "
                "regardless of actual outcome rates."
            ),
            "when_appropriate": [
                "Resource allocation where historical disparities should not perpetuate",
                "When base rate differences reflect systemic inequity, not medical reality",
            ],
            "when_inappropriate": [
                "When base rate differences reflect genuine medical need",
                "Diagnosis where accuracy matters more than proportionality",
            ],
            "measurement": "Ratio or difference of selection rates between groups",
        },
        FairnessMetric.EQUALIZED_ODDS: {
            "name": "Equalized Odds",
            "also_known_as": ["Conditional Procedure Accuracy Equality"],
            "definition": (
                "Both TPR and FPR are equal across groups. "
                "P(Y_hat=1|Y=y,A=a) = P(Y_hat=1|Y=y,A=b) for y in {0,1}"
            ),
            "intuition": (
                "The model's error rates are equal across groups for both "
                "positive and negative cases."
            ),
            "when_appropriate": [
                "Intervention triggers where both missed cases and false alarms matter",
                "When costs of both error types are significant",
            ],
            "when_inappropriate": [
                "When one type of error is much more costly than another",
            ],
            "measurement": "Maximum of |TPR_diff| and |FPR_diff| across groups",
        },
        FairnessMetric.EQUAL_OPPORTUNITY: {
            "name": "Equal Opportunity",
            "also_known_as": ["True Positive Rate Parity"],
            "definition": ("TPR is equal across groups. P(Y_hat=1|Y=1,A=a) = P(Y_hat=1|Y=1,A=b)"),
            "intuition": (
                "Among people who actually have the condition, all groups "
                "have equal chance of being identified."
            ),
            "when_appropriate": [
                "Screening programs where detection is paramount",
                "When false negatives are much more costly than false positives",
            ],
            "when_inappropriate": [
                "When false positives carry significant costs",
            ],
            "measurement": "Difference in TPR between groups",
        },
        FairnessMetric.PREDICTIVE_PARITY: {
            "name": "Predictive Parity",
            "also_known_as": ["Positive Predictive Value Parity"],
            "definition": ("PPV is equal across groups. P(Y=1|Y_hat=1,A=a) = P(Y=1|Y_hat=1,A=b)"),
            "intuition": (
                "A positive prediction means the same thing regardless of group membership."
            ),
            "when_appropriate": [
                "Diagnostic decision support",
                "When prediction trustworthiness matters across groups",
            ],
            "when_inappropriate": [
                "May conflict with equal opportunity when prevalence differs",
            ],
            "measurement": "Ratio or difference of PPV between groups",
        },
        FairnessMetric.CALIBRATION: {
            "name": "Calibration",
            "also_known_as": ["Test Fairness", "Matching Conditional Frequencies"],
            "definition": (
                "Predicted probabilities match observed frequencies across groups. "
                "E[Y|Y_hat=p,A=a] = E[Y|Y_hat=p,A=b] = p"
            ),
            "intuition": (
                "When the model says 30% risk, patients actually have 30% risk, "
                "regardless of demographic group."
            ),
            "when_appropriate": [
                "Risk communication to patients",
                "Shared decision-making scenarios",
                "When predicted probabilities are used directly",
            ],
            "when_inappropriate": [
                "Binary classification tasks where thresholds are fixed",
            ],
            "measurement": "Calibration curves by group, Expected Calibration Error (ECE)",
        },
    }

    if metric not in descriptions:
        return {"error": f"Unknown metric: {metric}"}

    return descriptions[metric]


def compare_metrics_tradeoffs(
    metric1: FairnessMetric,
    metric2: FairnessMetric,
) -> dict[str, Any]:
    """Compare trade-offs between two fairness metrics.

    Args:
        metric1: First metric.
        metric2: Second metric.

    Returns:
        Dict describing trade-offs and when each is preferred.
    """
    tradeoffs = {
        (FairnessMetric.DEMOGRAPHIC_PARITY, FairnessMetric.EQUALIZED_ODDS): {
            "tension": (
                "When prevalence differs, satisfying demographic parity typically "
                "violates equalized odds and vice versa."
            ),
            "choose_first_when": (
                "Base rate differences reflect historical inequity that should "
                "not influence resource allocation."
            ),
            "choose_second_when": (
                "Base rate differences reflect genuine medical need and "
                "intervention accuracy matters."
            ),
        },
        (FairnessMetric.EQUAL_OPPORTUNITY, FairnessMetric.PREDICTIVE_PARITY): {
            "tension": (
                "With different prevalence, equalizing TPR typically creates "
                "different PPV across groups."
            ),
            "choose_first_when": (
                "Missing cases (false negatives) is the primary concern, "
                "such as screening programs."
            ),
            "choose_second_when": (
                "Positive prediction trustworthiness matters, such as diagnostic decision support."
            ),
        },
        (FairnessMetric.CALIBRATION, FairnessMetric.DEMOGRAPHIC_PARITY): {
            "tension": (
                "Perfect calibration with different prevalence necessarily "
                "produces different selection rates."
            ),
            "choose_first_when": (
                "Predicted probabilities are communicated to patients "
                "or used in shared decision-making."
            ),
            "choose_second_when": (
                "Proportional resource allocation is the goal and base rates may reflect inequity."
            ),
        },
    }

    # Check both orderings
    key = (metric1, metric2)
    if key not in tradeoffs:
        key = (metric2, metric1)
        if key not in tradeoffs:
            return {
                "note": (
                    "Specific trade-off documentation not available for this pair. "
                    "See impossibility theorem for general guidance."
                )
            }

    return tradeoffs[key]


def format_decision_tree_text() -> str:
    """Format the decision tree as readable text for reports.

    Returns:
        Formatted string with all decision tree information.
    """
    lines = [
        "=" * 70,
        "FAIRNESS METRIC OPTIONS BY USE CASE",
        "=" * 70,
        "",
        "This reference presents metric options organized by clinical use case.",
        "Based on healthcare AI fairness literature. Your organization decides.",
        "",
    ]

    for use_case, config in DECISION_TREE.items():
        lines.extend(
            [
                "-" * 70,
                f"USE CASE: {use_case.value.upper().replace('_', ' ')}",
                "-" * 70,
                "",
                f"PRIMARY OPTION: {config['recommended'].value}",
                f"  Context: {config['rationale'][:200]}...",
                "",
            ]
        )

        if config.get("alternatives"):
            alts = [m.value for m in config["alternatives"]]
            lines.append(f"ALTERNATIVES: {', '.join(alts)}")
            if config.get("alternative_rationale"):
                lines.append(f"  {config['alternative_rationale'][:150]}...")
            lines.append("")

        if config.get("contraindicated"):
            contra = [m.value for m in config["contraindicated"]]
            lines.append(f"TRADEOFF CONSIDERATIONS: {', '.join(contra)}")
            if config.get("contraindication_reason"):
                lines.append(f"  {config['contraindication_reason'][:150]}...")
            lines.append("")

        if config.get("clinical_examples"):
            lines.append("EXAMPLES:")
            for ex in config["clinical_examples"][:3]:
                lines.append(f"  - {ex}")
            lines.append("")

    lines.extend(
        [
            "=" * 70,
            "See get_impossibility_warning() for mathematical background.",
            "=" * 70,
        ]
    )

    return "\n".join(lines)
