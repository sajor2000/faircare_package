"""Tests for RAIC Checkpoint 1 checklist export."""

import json
from pathlib import Path

import pytest

from faircareai.core.config import FairnessConfig, FairnessMetric
from faircareai.core.results import AuditResults


@pytest.fixture
def raic_results() -> AuditResults:
    config = FairnessConfig(
        model_name="Test Model",
        model_version="1.0.0",
        primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
        fairness_justification="Equalized odds chosen for parity monitoring.",
    )
    results = AuditResults(config=config)
    results.overall_performance = {
        "discrimination": {"auroc": 0.81, "auroc_ci_fmt": "(0.75-0.86)"},
        "confusion_matrix": {"tp": 10, "fp": 5, "tn": 80, "fn": 5},
    }
    results.subgroup_performance = {"race": {"groups": {"Black": {}, "White": {}}}}
    results.fairness_metrics = {"race": {"calibration_diff": {"Black": 0.02}}}
    return results


def test_raic_checklist_full_length(raic_results: AuditResults, tmp_path: Path) -> None:
    path = tmp_path / "raic_checkpoint_1.json"
    raic_results.to_raic_checkpoint_1(path)
    payload = json.loads(path.read_text())
    criteria = payload["criteria"]
    assert len(criteria) == 178

    ids = {item["id"] for item in criteria}
    assert "AC1.CR79" in ids
    cr1 = next(item for item in criteria if item["id"] == "AC1.CR1")
    assert "intended use" in cr1["summary"].lower()

    statuses = {item["status"] for item in criteria}
    assert "MET" in statuses
