"""
FairCareAI Fairness Module

Fairness metric definitions, decision tree for metric selection,
and impossibility theorem documentation.
"""

from faircareai.fairness.decision_tree import (
    DECISION_TREE,
    get_impossibility_warning,
    recommend_fairness_metric,
)

__all__ = [
    "recommend_fairness_metric",
    "get_impossibility_warning",
    "DECISION_TREE",
]
