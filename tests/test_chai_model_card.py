"""Tests for CHAI model card exports."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from faircareai.core.config import FairnessConfig, FairnessMetric
from faircareai.core.results import AuditResults


@pytest.fixture
def chai_results() -> AuditResults:
    config = FairnessConfig(
        model_name="Test Model",
        model_version="1.0.0",
        primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
        fairness_justification="Equalized odds chosen for clinical parity.",
        intended_use="Risk stratification",
        intended_population="Adult inpatient population",
    )
    results = AuditResults(config=config)
    results.run_timestamp = "2025-01-01T00:00:00"
    results.descriptive_stats = {"cohort_overview": {"n_total": 1000, "prevalence_pct": "10%"}}
    results.overall_performance = {
        "discrimination": {"auroc": 0.84, "auprc": 0.32},
        "calibration": {"brier_score": 0.09, "calibration_slope": 0.98},
        "classification_at_threshold": {"sensitivity": 0.71, "specificity": 0.82},
    }
    results.fairness_metrics = {"race": {"equalized_odds_diff": {"Black": 0.05}}}
    return results


def test_chai_model_card_xml_export(chai_results: AuditResults, tmp_path: Path) -> None:
    path = tmp_path / "chai_model_card.xml"
    chai_results.to_chai_model_card(path)
    assert path.exists()

    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"m": "https://mc.chai.org/v0.1/schema.xsd"}
    assert root.tag.endswith("AppliedModelCard")
    assert root.find("m:BasicInfo/m:ModelName", ns) is not None
    assert root.find("m:ReleaseInfo/m:ReleaseStage", ns) is not None
    assert root.find("m:ModelSummary/m:Summary", ns) is not None
    assert root.find("m:UsesAndDirections/m:IntendedUseAndWorkflow", ns) is not None
    assert root.find("m:Warnings/m:KnownRisksAndLimitations", ns) is not None
    assert root.find("m:TrustIngredients/m:AISystemFacts/m:ModelType", ns) is not None
    assert root.find("m:KeyMetrics/m:UsefulnessUsabilityEfficacy/m:Result", ns) is not None
    assert root.find("m:Resources/m:PeerReviewedPublications", ns) is not None
    assert root.find("m:Bibliography", ns) is not None


def test_chai_model_card_json_export(chai_results: AuditResults, tmp_path: Path) -> None:
    path = tmp_path / "chai_model_card.json"
    chai_results.to_chai_model_card_json(path)
    assert path.exists()
    payload = json.loads(path.read_text())
    assert "AppliedModelCard" in payload
    assert "metadata" in payload


def test_chai_model_card_xml_validation(chai_results: AuditResults, tmp_path: Path) -> None:
    xmlschema = pytest.importorskip("xmlschema")
    path = tmp_path / "chai_model_card.xml"
    chai_results.to_chai_model_card(path)

    schema_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "faircareai"
        / "data"
        / "chai"
        / "v0.1"
        / "schema.xsd"
    )
    schema = xmlschema.XMLSchema(str(schema_path))
    schema.validate(str(path))
