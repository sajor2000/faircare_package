"""
FairCareAI - Healthcare AI Fairness Auditing

A Python package for auditing machine learning models for fairness
in clinical contexts. Designed for healthcare-specific attributes,
clinical threshold calibration, and HIPAA-compliant local analysis.

Metrics computed per Van Calster et al. (2025) methodology. Healthcare
organizations interpret results based on their clinical context.
"""

__version__ = "0.2.0"

# Core API - Primary entry points
from faircareai.core.audit import FairCareAudit
from faircareai.core.config import (
    FairnessConfig,
    FairnessMetric,
    ModelType,
    SensitiveAttribute,
    UseCaseType,
)
from faircareai.core.results import AuditResults

# Fairness module
from faircareai.fairness.decision_tree import (
    DECISION_TREE,
    get_impossibility_warning,
    recommend_fairness_metric,
)

__all__ = [
    # Core API
    "FairCareAudit",
    "FairnessConfig",
    "AuditResults",
    # Enums
    "FairnessMetric",
    "UseCaseType",
    "ModelType",
    "SensitiveAttribute",
    # Fairness decision tree
    "recommend_fairness_metric",
    "get_impossibility_warning",
    "DECISION_TREE",
    # Dashboard
    "launch",
]


# Lazy imports to avoid loading heavy dependencies on startup
def launch() -> None:
    """Launch the FairCare interactive dashboard."""
    from faircareai.dashboard.app import launch as _launch

    _launch()
