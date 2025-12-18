"""
FairCareAI Configuration Classes

Defines model types, use cases, fairness metrics, and audit configuration.

Methodology: CHAI RAIC Checkpoint 1 (configurable thresholds).
Note: Thresholds are starting points - organizations set their own based on context.

Van Calster et al. (2025) Metric Classification:
-------------------------------------------------
This module includes MetricDisplayConfig which controls which metrics are shown
based on Van Calster Table 2 recommendations:
  - RECOMMENDED: Essential for all reports (AUROC, calibration plot, net benefit, risk distribution)
  - OPTIONAL: Acceptable for data science (Brier, O:E ratio, sens+spec, PPV+NPV)
  - CAUTION: Improper measures (F1, accuracy, MCC, DOR, Kappa) - use with explicit caveats
"""

from dataclasses import dataclass, field
from enum import Enum

from faircareai.core.constants import (
    VANCALSTER_ALL_CAUTION,
    VANCALSTER_ALL_OPTIONAL,
    VANCALSTER_ALL_RECOMMENDED,
)


class OutputPersona(Enum):
    """Output persona for report generation.

    Determines the level of detail and technical language in generated reports.
    Use DATA_SCIENTIST for full technical output, GOVERNANCE for streamlined
    non-technical output suitable for governance committees.
    """

    DATA_SCIENTIST = "data_scientist"
    """Full technical output with all metrics, confidence intervals, and detailed tables."""

    GOVERNANCE = "governance"
    """Streamlined 3-5 page output with key figures and plain language summaries."""


@dataclass
class MetricDisplayConfig:
    """Configuration for which metrics to display based on Van Calster et al. (2025).

    Van Calster Table 2 classifies performance measures into three categories:
    - RECOMMENDED: Essential for all reports (always shown)
    - OPTIONAL: Acceptable for data science teams (shown when include_optional=True)
    - CAUTION: Improper measures (shown only with explicit show_caution=True + warning)

    Default behavior:
    - Governance persona: RECOMMENDED only (always)
    - Data Scientist persona: RECOMMENDED only (default), OPTIONAL on request

    Example:
        >>> # Default: RECOMMENDED metrics only
        >>> config = MetricDisplayConfig.data_scientist()
        >>> config.should_show("auroc")  # True (RECOMMENDED)
        >>> config.should_show("brier_score")  # False (OPTIONAL, not enabled)

        >>> # Include OPTIONAL metrics
        >>> config = MetricDisplayConfig.data_scientist(include_optional=True)
        >>> config.should_show("brier_score")  # True

        >>> # Governance always shows RECOMMENDED only
        >>> config = MetricDisplayConfig.governance()
        >>> config.should_show("brier_score")  # False (OPTIONAL never shown)
    """

    show_recommended: bool = True
    """Always show Van Calster RECOMMENDED metrics. Cannot be disabled."""

    show_optional: bool = False
    """Show Van Calster OPTIONAL metrics (Brier, O:E ratio, sens+spec, PPV+NPV)."""

    show_caution: bool = False
    """Show Van Calster CAUTION metrics (F1, accuracy, MCC, etc.) with warnings."""

    persona: OutputPersona = OutputPersona.DATA_SCIENTIST
    """The output persona this config is for."""

    @classmethod
    def governance(cls) -> "MetricDisplayConfig":
        """Create config for Governance persona - RECOMMENDED metrics only.

        Governance reports always show only Van Calster RECOMMENDED metrics:
        - AUROC (discrimination)
        - Calibration plot (smoothed)
        - Net Benefit / Decision Curve (clinical utility)
        - Risk Distribution plots

        Returns:
            MetricDisplayConfig with show_optional=False, show_caution=False.
        """
        return cls(
            show_recommended=True,
            show_optional=False,
            show_caution=False,
            persona=OutputPersona.GOVERNANCE,
        )

    @classmethod
    def data_scientist(cls, include_optional: bool = False) -> "MetricDisplayConfig":
        """Create config for Data Scientist persona.

        Default: RECOMMENDED metrics only (same as governance).
        With include_optional=True: Also shows OPTIONAL metrics like Brier score,
        O:E ratio, sensitivity+specificity, PPV+NPV.

        Args:
            include_optional: If True, include Van Calster OPTIONAL metrics.

        Returns:
            MetricDisplayConfig for data scientist output.
        """
        return cls(
            show_recommended=True,
            show_optional=include_optional,
            show_caution=False,
            persona=OutputPersona.DATA_SCIENTIST,
        )

    def should_show(self, metric: str) -> bool:
        """Check if a metric should be shown based on Van Calster classification.

        Args:
            metric: Metric name (lowercase, e.g., "auroc", "brier_score", "f1_score").

        Returns:
            True if the metric should be displayed, False otherwise.
        """
        metric_lower = metric.lower()

        # RECOMMENDED metrics are always shown
        if metric_lower in VANCALSTER_ALL_RECOMMENDED:
            return True

        # OPTIONAL metrics shown only if enabled
        if metric_lower in VANCALSTER_ALL_OPTIONAL:
            return self.show_optional

        # CAUTION metrics shown only if explicitly enabled
        if metric_lower in VANCALSTER_ALL_CAUTION:
            return self.show_caution

        # Unknown metrics: show for data scientist, hide for governance
        return self.persona == OutputPersona.DATA_SCIENTIST

    def get_metric_category(self, metric: str) -> str:
        """Get the Van Calster classification category for a metric.

        Args:
            metric: Metric name (lowercase).

        Returns:
            "RECOMMENDED", "OPTIONAL", "CAUTION", or "UNKNOWN".
        """
        metric_lower = metric.lower()

        if metric_lower in VANCALSTER_ALL_RECOMMENDED:
            return "RECOMMENDED"
        if metric_lower in VANCALSTER_ALL_OPTIONAL:
            return "OPTIONAL"
        if metric_lower in VANCALSTER_ALL_CAUTION:
            return "CAUTION"
        return "UNKNOWN"

    def filter_metrics(self, metrics: list[str]) -> list[str]:
        """Filter a list of metrics to only those that should be shown.

        Args:
            metrics: List of metric names.

        Returns:
            Filtered list containing only metrics that should be displayed.
        """
        return [m for m in metrics if self.should_show(m)]


# =============================================================================
# Van Calster Priority Ordering (Table 2)
# =============================================================================

# OPTIONAL metrics in Van Calster preference order (most to least preferred)
VANCALSTER_OPTIONAL_PRIORITY: list[str] = [
    "brier_score",  # Overall performance (highest priority)
    "scaled_brier",  # Overall performance
    "oe_ratio",  # Calibration-in-the-large
    "calibration_slope",  # Calibration parameter
    "calibration_intercept",  # Calibration parameter
    "ici",  # Integrated Calibration Index
    "eci",  # Expected Calibration Error
    "sensitivity",  # Classification (must show with specificity)
    "specificity",  # Classification (must show with sensitivity)
    "ppv",  # Predictive values (must show with NPV)
    "npv",  # Predictive values (must show with PPV)
]


def sort_metrics_by_priority(
    metrics: list[str],
    priority_list: list[str] | None = None,
) -> list[str]:
    """Sort metrics by Van Calster preference order.

    OPTIONAL metrics are sorted by preference from Van Calster et al. (2025):
    Brier score > O:E ratio > ICI > Sens/Spec > PPV/NPV.

    Args:
        metrics: List of metric names to sort.
        priority_list: Custom priority list. Defaults to VANCALSTER_OPTIONAL_PRIORITY.

    Returns:
        Sorted list with priority metrics first, unknown metrics last.

    Example:
        >>> sort_metrics_by_priority(["ppv", "brier_score", "sensitivity", "oe_ratio"])
        ['brier_score', 'oe_ratio', 'sensitivity', 'ppv']
    """
    if priority_list is None:
        priority_list = VANCALSTER_OPTIONAL_PRIORITY
    priority_map = {m.lower(): i for i, m in enumerate(priority_list)}
    return sorted(metrics, key=lambda m: priority_map.get(m.lower(), 999))


# =============================================================================
# Persona-Tailored Terminology
# =============================================================================


@dataclass
class MetricTerminology:
    """Persona-specific terminology for a metric.

    Provides different labels and descriptions for Data Scientist (technical)
    vs Governance (plain language) personas.

    Attributes:
        data_scientist: Technical name (e.g., "AUROC")
        governance: Plain language name (e.g., "Model Discrimination")
        ds_description: Technical description
        gov_description: Plain language description for non-technical stakeholders
        interpretation: Brief interpretation guidance for governance (e.g., "higher = better")
        x_axis_ds: X-axis label for technical audience
        x_axis_gov: X-axis label for governance/lay audience
        y_axis_ds: Y-axis label for technical audience
        y_axis_gov: Y-axis label for governance/lay audience
    """

    data_scientist: str
    governance: str
    ds_description: str
    gov_description: str
    interpretation: str = ""  # Brief interpretation hint for governance legends
    x_axis_ds: str | None = None
    x_axis_gov: str | None = None
    y_axis_ds: str | None = None
    y_axis_gov: str | None = None


# Comprehensive terminology mapping for all FairCareAI metrics
PERSONA_TERMINOLOGY: dict[str, MetricTerminology] = {
    # RECOMMENDED metrics
    "auroc": MetricTerminology(
        data_scientist="AUROC",
        governance="Model Discrimination",
        ds_description="Area Under ROC Curve (0.5 = random, 1.0 = perfect)",
        gov_description="How well the model separates high-risk from low-risk patients",
        interpretation="higher = better; ≥0.7 acceptable, ≥0.8 good, ≥0.9 excellent",
        x_axis_ds="1 - Specificity (False Positive Rate)",
        x_axis_gov="False Alarm Rate",
        y_axis_ds="Sensitivity (True Positive Rate)",
        y_axis_gov="Detection Rate",
    ),
    "calibration": MetricTerminology(
        data_scientist="Calibration",
        governance="Prediction Accuracy",
        ds_description="Agreement between predicted probabilities and observed outcomes",
        gov_description="Are the predicted risk percentages accurate?",
        interpretation="points on diagonal = accurate; above = under-predicting; below = over-predicting",
        x_axis_ds="Predicted Probability",
        x_axis_gov="Predicted Risk Level",
        y_axis_ds="Observed Proportion",
        y_axis_gov="Actual Outcome Rate",
    ),
    "net_benefit": MetricTerminology(
        data_scientist="Net Benefit",
        governance="Clinical Value Added",
        ds_description="Net benefit = (TP/n) - (FP/n) × (pt/(1-pt))",
        gov_description=(
            "Should we use this model to decide who gets treatment? "
            "Compares: (1) use the model, (2) treat everyone, (3) treat no one. "
            "Shows if using the model leads to better patient outcomes than simple strategies."
        ),
        interpretation=(
            "MODEL USEFUL when its curve is ABOVE both gray lines. "
            "Gray 'Treat All' = give everyone the intervention. "
            "Gray 'Treat None' = give no one the intervention. "
            "Higher curve = more correct decisions per 100 patients. "
            "Where curves cross = model stops adding value at that risk level"
        ),
        x_axis_ds="Threshold Probability",
        x_axis_gov="Risk Cutoff Level",
        y_axis_ds="Net Benefit",
        y_axis_gov="Clinical Value",
    ),
    "risk_distribution": MetricTerminology(
        data_scientist="Risk Distribution",
        governance="Risk Score Spread",
        ds_description="Distribution of predicted probabilities by outcome",
        gov_description="How are patients distributed across risk levels?",
        interpretation="wider gap between Events/Non-Events = better discrimination",
        x_axis_ds="Predicted Probability",
        x_axis_gov="Risk Score",
        y_axis_ds="Density",
        y_axis_gov="Number of Patients",
    ),
    # OPTIONAL metrics
    "sensitivity": MetricTerminology(
        data_scientist="Sensitivity",
        governance="Detection Rate",
        ds_description="True Positive Rate = TP / (TP + FN)",
        gov_description="Of patients who have the condition, what % are correctly identified?",
        interpretation="higher = fewer missed cases; 100% = all cases detected",
    ),
    "specificity": MetricTerminology(
        data_scientist="Specificity",
        governance="Correct Rejection Rate",
        ds_description="True Negative Rate = TN / (TN + FP)",
        gov_description="Of patients without the condition, what % are correctly identified?",
        interpretation="higher = fewer false alarms; 100% = no false alarms",
    ),
    "ppv": MetricTerminology(
        data_scientist="PPV",
        governance="Positive Predictive Value",
        ds_description="Precision = TP / (TP + FP)",
        gov_description="When model predicts high risk, how often is it right?",
        interpretation="higher = more trustworthy positive predictions",
    ),
    "npv": MetricTerminology(
        data_scientist="NPV",
        governance="Negative Predictive Value",
        ds_description="NPV = TN / (TN + FN)",
        gov_description="When model predicts low risk, how often is it right?",
        interpretation="higher = more trustworthy negative predictions",
    ),
    "brier_score": MetricTerminology(
        data_scientist="Brier Score",
        governance="Prediction Error",
        ds_description="Mean squared error of predictions (0 = perfect, 1 = worst)",
        gov_description="How far off are the predictions on average?",
        interpretation="lower = better; 0 = perfect, <0.25 good",
    ),
    "scaled_brier": MetricTerminology(
        data_scientist="Scaled Brier Score",
        governance="Prediction Skill",
        ds_description="Brier skill score: 1 - (Brier / Brier_null)",
        gov_description="How much better is the model than guessing?",
        interpretation="higher = better; >0 beats random guessing",
    ),
    "oe_ratio": MetricTerminology(
        data_scientist="O:E Ratio",
        governance="Observed vs Expected",
        ds_description="Ratio of observed to expected events (1.0 = well-calibrated)",
        gov_description="Does the model over- or under-predict risk?",
        interpretation="1.0 = accurate; <1 over-predicts; >1 under-predicts",
    ),
    "calibration_slope": MetricTerminology(
        data_scientist="Calibration Slope",
        governance="Prediction Reliability",
        ds_description="Slope from logistic calibration (1.0 = perfect)",
        gov_description="Are extreme predictions trustworthy?",
        interpretation="1.0 = ideal; <1 = over-confident; >1 = under-confident",
    ),
    "calibration_intercept": MetricTerminology(
        data_scientist="Calibration Intercept",
        governance="Prediction Bias",
        ds_description="Intercept from logistic calibration (0.0 = perfect)",
        gov_description="Is the model systematically over- or under-predicting?",
        interpretation="0 = no bias; <0 over-predicts; >0 under-predicts",
    ),
    "ici": MetricTerminology(
        data_scientist="ICI",
        governance="Calibration Quality",
        ds_description="Integrated Calibration Index (weighted mean calibration error)",
        gov_description="Overall accuracy of risk predictions",
        interpretation="lower = better; 0 = perfect calibration",
    ),
    "eci": MetricTerminology(
        data_scientist="ECI",
        governance="Expected Calibration Error",
        ds_description="Expected Calibration Error (squared ICI)",
        gov_description="Prediction error measure",
        interpretation="lower = better; 0 = perfect",
    ),
    # General terms
    "threshold": MetricTerminology(
        data_scientist="Threshold",
        governance="Risk Cutoff",
        ds_description="Decision threshold probability",
        gov_description="Risk level used to decide who needs intervention",
        interpretation="lower = more sensitive; higher = more specific",
        x_axis_ds="Threshold Probability",
        x_axis_gov="Risk Cutoff (%)",
    ),
    "decision_curve": MetricTerminology(
        data_scientist="Decision Curve Analysis",
        governance="Clinical Utility Analysis",
        ds_description="Net benefit across threshold probabilities",
        gov_description=(
            "At what risk level should we take action? "
            "This chart shows if using the model helps make better decisions "
            "than treating everyone or treating no one, across different risk cutoffs."
        ),
        interpretation=(
            "X-axis = risk cutoff for taking action (e.g., 20% = intervene if risk ≥20%). "
            "Y-axis = clinical value (correct decisions minus harm from unnecessary treatment). "
            "MODEL ADDS VALUE when colored curve is ABOVE both gray baselines. "
            "Wider gap above gray = more clinical benefit from using the model"
        ),
        x_axis_ds="Threshold Probability",
        x_axis_gov="Risk Cutoff Level",
        y_axis_ds="Net Benefit",
        y_axis_gov="Clinical Value",
    ),
    # CAUTION metrics (for completeness)
    "f1_score": MetricTerminology(
        data_scientist="F1 Score",
        governance="Balance Score",
        ds_description="Harmonic mean of precision and recall",
        gov_description="Balance between detection and false alarms (use with caution)",
        interpretation="⚠ use with caution; higher = better balance",
    ),
    "accuracy": MetricTerminology(
        data_scientist="Accuracy",
        governance="Correct Predictions",
        ds_description="(TP + TN) / Total",
        gov_description="Percentage of correct predictions (misleading for imbalanced data)",
        interpretation="⚠ misleading for rare outcomes; higher ≠ always better",
    ),
    "auprc": MetricTerminology(
        data_scientist="AUPRC",
        governance="Precision-Recall Score",
        ds_description="Area Under Precision-Recall Curve",
        gov_description="Detection performance for rare conditions",
        interpretation="⚠ use with caution; higher = better for rare events",
    ),
}


def get_label(
    metric: str,
    persona: OutputPersona,
    label_type: str = "name",
) -> str:
    """Get persona-appropriate label for a metric.

    Returns technical terminology for Data Scientist persona and plain
    language with interpretation guidance for Governance persona.

    Args:
        metric: Metric key (e.g., "auroc", "sensitivity", "threshold")
        persona: OutputPersona.DATA_SCIENTIST or OutputPersona.GOVERNANCE
        label_type: Type of label to return:
            - "name": Short metric name (default)
            - "name_with_interp": Name with interpretation (governance only)
            - "description": Longer explanatory text
            - "interpretation": How to interpret the metric
            - "x_axis": X-axis label for plots
            - "y_axis": Y-axis label for plots

    Returns:
        Persona-appropriate label string. Falls back to original metric
        name if not found in terminology dictionary.

    Example:
        >>> get_label("auroc", OutputPersona.DATA_SCIENTIST)
        'AUROC'
        >>> get_label("auroc", OutputPersona.GOVERNANCE)
        'Model Discrimination (AUROC) — higher = better; ≥0.7 acceptable, ≥0.8 good'
        >>> get_label("sensitivity", OutputPersona.GOVERNANCE)
        'Detection Rate (Sensitivity) — higher = fewer missed cases'
        >>> get_label("auroc", OutputPersona.GOVERNANCE, "x_axis")
        'False Alarm Rate'
        >>> get_label("auroc", OutputPersona.DATA_SCIENTIST, "interpretation")
        'higher = better; ≥0.7 acceptable, ≥0.8 good, ≥0.9 excellent'
    """
    term = PERSONA_TERMINOLOGY.get(metric.lower())
    if term is None:
        return metric  # Fallback to original

    if persona == OutputPersona.GOVERNANCE:
        if label_type == "name":
            # Include technical term AND interpretation for governance
            base = f"{term.governance} ({term.data_scientist})"
            if term.interpretation:
                return f"{base} — {term.interpretation}"
            return base
        elif label_type == "name_with_interp":
            # Same as name for governance (always includes interpretation)
            base = f"{term.governance} ({term.data_scientist})"
            if term.interpretation:
                return f"{base} — {term.interpretation}"
            return base
        elif label_type == "description":
            return term.gov_description
        elif label_type == "interpretation":
            return term.interpretation
        elif label_type == "x_axis":
            return term.x_axis_gov or term.x_axis_ds or ""
        elif label_type == "y_axis":
            return term.y_axis_gov or term.y_axis_ds or ""
    else:  # DATA_SCIENTIST
        if label_type == "name":
            return term.data_scientist
        elif label_type == "name_with_interp":
            # DS can also get interpretation if requested
            if term.interpretation:
                return f"{term.data_scientist} — {term.interpretation}"
            return term.data_scientist
        elif label_type == "interpretation":
            return term.interpretation
        elif label_type == "description":
            return term.ds_description
        elif label_type == "x_axis":
            return term.x_axis_ds or ""
        elif label_type == "y_axis":
            return term.y_axis_ds or ""

    return metric


def get_axis_labels(
    plot_type: str,
    persona: OutputPersona,
) -> tuple[str, str]:
    """Get persona-appropriate axis labels for a plot type.

    Args:
        plot_type: Type of plot (e.g., "auroc", "calibration", "decision_curve")
        persona: OutputPersona.DATA_SCIENTIST or OutputPersona.GOVERNANCE

    Returns:
        Tuple of (x_axis_label, y_axis_label)

    Example:
        >>> get_axis_labels("auroc", OutputPersona.GOVERNANCE)
        ('False Alarm Rate', 'Detection Rate')
        >>> get_axis_labels("calibration", OutputPersona.DATA_SCIENTIST)
        ('Predicted Probability', 'Observed Proportion')
    """
    x_label = get_label(plot_type, persona, "x_axis")
    y_label = get_label(plot_type, persona, "y_axis")
    return (x_label, y_label)


class ModelType(Enum):
    """Type of ML model being audited."""

    BINARY_CLASSIFIER = "binary_classifier"
    RISK_SCORE = "risk_score"
    MULTICLASS = "multiclass"


class UseCaseType(Enum):
    """Clinical/operational use case for the model.

    The use case determines which fairness metric is most appropriate.
    """

    INTERVENTION_TRIGGER = "intervention_trigger"
    """Model triggers an intervention (e.g., care management outreach)."""

    RISK_COMMUNICATION = "risk_communication"
    """Model communicates risk to patients/providers for shared decision-making."""

    RESOURCE_ALLOCATION = "resource_allocation"
    """Model allocates limited resources (e.g., care coordination slots)."""

    SCREENING = "screening"
    """Model screens for a condition (e.g., cancer screening)."""

    DIAGNOSIS_SUPPORT = "diagnosis_support"
    """Model supports clinical diagnosis."""


class FairnessMetric(Enum):
    """Fairness metric definitions.

    NOTE: The impossibility theorem means no single metric is universally correct.
    Selection is a value judgment that humans must make based on context.
    """

    DEMOGRAPHIC_PARITY = "demographic_parity"
    """Equal selection rates across groups. P(Y_hat=1|A=a) = P(Y_hat=1|A=b)"""

    EQUALIZED_ODDS = "equalized_odds"
    """Equal TPR and FPR across groups. Good for intervention triggers."""

    EQUAL_OPPORTUNITY = "equal_opportunity"
    """Equal TPR across groups (subset of equalized odds)."""

    PREDICTIVE_PARITY = "predictive_parity"
    """Equal PPV across groups. P(Y=1|Y_hat=1,A=a) = P(Y=1|Y_hat=1,A=b)"""

    CALIBRATION = "calibration"
    """Equal calibration across groups. Good for risk communication."""

    INDIVIDUAL_FAIRNESS = "individual_fairness"
    """Similar individuals receive similar predictions."""


@dataclass
class SensitiveAttribute:
    """Defines a sensitive/protected attribute for fairness analysis."""

    name: str
    """Display name (e.g., 'race', 'sex')."""

    column: str
    """Column name in the data."""

    reference: str | None = None
    """Reference group for comparisons."""

    categories: list[str] | None = None
    """Expected category values."""

    attr_type: str = "categorical"
    """'categorical' or 'binary'."""

    is_protected: bool = True
    """Whether this is a legally protected characteristic."""

    clinical_justification: str | None = None
    """CHAI-required justification if attribute influences model."""


@dataclass
class FairnessConfig:
    """Configuration for fairness audit following CHAI RAIC framework.

    IMPORTANT: Default thresholds are evidence-based starting points from
    fairness literature. Health systems should adjust these based on their
    clinical context, risk tolerance, and organizational equity goals.

    FairCareAI provides CHAI-grounded guidance. Final decisions on acceptable
    thresholds rest with the data scientist and health system.
    """

    # Model Identity (CHAI AC1.CR1-4)
    model_name: str
    model_version: str = "1.0.0"
    model_type: ModelType = ModelType.BINARY_CLASSIFIER

    # Intended Use (CHAI AC1.CR1, AC1.CR100)
    intended_use: str = ""
    intended_population: str = ""
    out_of_scope: list[str] = field(default_factory=list)

    # REQUIRED: Fairness Prioritization (CHAI AC1.CR92-93)
    # The data scientist MUST select a metric and provide justification
    # FairCareAI recommends but does not dictate the choice
    primary_fairness_metric: FairnessMetric | None = None
    fairness_justification: str = ""
    use_case_type: UseCaseType | None = None

    # Thresholds for flagging - CONFIGURABLE BY HEALTH SYSTEM
    # Defaults are evidence-based starting points, not requirements
    thresholds: dict = field(
        default_factory=lambda: {
            "min_subgroup_n": 100,  # Adjust based on power requirements
            "demographic_parity_ratio": (
                0.8,
                1.25,
            ),  # 80% rule from EEOC, adjust as appropriate
            "equalized_odds_diff": 0.1,  # Adjust based on clinical impact
            "calibration_diff": 0.05,  # Adjust based on decision context
            "min_auroc": 0.65,  # Adjust based on use case
            "max_missing_rate": 0.10,  # Adjust based on data quality standards
        }
    )

    # Decision threshold(s) for the model
    decision_thresholds: list[float] = field(default_factory=lambda: [0.5])

    # Report settings
    include_chai_mapping: bool = True
    organization_name: str = ""
    report_date: str | None = None

    def validate(self) -> list[str]:
        """Validate config and return list of warnings/errors.

        Returns:
            List of validation messages. Errors start with "ERROR:",
            warnings start with "WARNING:".
        """
        issues = []

        if not self.primary_fairness_metric:
            issues.append(
                "ERROR: primary_fairness_metric is required (CHAI AC1.CR92). "
                "Use audit.suggest_fairness_metric() for recommendations."
            )

        if not self.fairness_justification:
            issues.append(
                "ERROR: fairness_justification is required (CHAI AC1.CR93). "
                "Document why you selected this fairness metric."
            )

        if not self.intended_use:
            issues.append(
                "WARNING: intended_use should be specified (CHAI AC1.CR1). "
                "Describe how the model will be used clinically."
            )

        if not self.intended_population:
            issues.append(
                "WARNING: intended_population should be specified (CHAI AC1.CR3). "
                "Describe the target patient population."
            )

        if not self.use_case_type:
            issues.append(
                "WARNING: use_case_type not specified. "
                "This helps determine appropriate fairness metrics."
            )

        return issues

    def has_errors(self) -> bool:
        """Check if configuration has any errors (not just warnings)."""
        return any(issue.startswith("ERROR:") for issue in self.validate())

    def get_threshold(self, key: str, default: float | None = None) -> float | None:
        """Get a threshold value with optional default."""
        return self.thresholds.get(key, default)
