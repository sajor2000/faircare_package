"""
FairCareAI Configuration Classes

Defines model types, use cases, fairness metrics, and audit configuration.

Methodology: CHAI RAIC Checkpoint 1 (configurable thresholds).
Note: Thresholds are starting points - organizations set their own based on context.
"""

from dataclasses import dataclass, field
from enum import Enum


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
