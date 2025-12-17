"""FairCareAI Core Module

Core components for healthcare AI fairness auditing.

Methodology: Van Calster et al. (2025), CHAI RAIC Checkpoint 1.
"""

from faircareai.core.audit import AuditResult, FairCareAudit
from faircareai.core.config import (
    FairnessConfig,
    FairnessMetric,
    ModelType,
    SensitiveAttribute,
    UseCaseType,
)
from faircareai.core.disparity import DisparityResult, compute_disparities
from faircareai.core.metrics import GroupMetrics, compute_group_metrics
from faircareai.core.results import AuditResults
from faircareai.core.statistical import newcombe_wilson_ci, wilson_score_ci

# Legacy alias for backward compatibility
FairAudit = FairCareAudit

__all__ = [
    # Primary API
    "FairCareAudit",
    "FairnessConfig",
    "AuditResults",
    # Enums
    "FairnessMetric",
    "UseCaseType",
    "ModelType",
    "SensitiveAttribute",
    # Metrics
    "compute_group_metrics",
    "GroupMetrics",
    "compute_disparities",
    "DisparityResult",
    # Statistical
    "wilson_score_ci",
    "newcombe_wilson_ci",
    # Legacy
    "FairAudit",
    "AuditResult",
]
