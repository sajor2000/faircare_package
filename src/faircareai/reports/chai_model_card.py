"""CHAI Applied Model Card (v0.1) generator with XML schema alignment."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Any

from faircareai.core.exceptions import ConfigurationError
from faircareai.core.logging import get_logger
from faircareai.core.results import AuditResults

logger = get_logger(__name__)

CHAI_SCHEMA_VERSION = "v0.1"
CHAI_SCHEMA_URL = (
    "https://raw.githubusercontent.com/coalition-for-health-ai/mc-schema/main/v0.1/schema.xsd"
)
CHAI_TEMPLATE_URL = "https://mc.chai.org/v0.1/documentation.pdf"
CHAI_SCHEMA_REPO = "https://github.com/coalition-for-health-ai/mc-schema"
CHAI_NAMESPACE = "https://mc.chai.org/v0.1/schema.xsd"


def _as_text(value: Any, default: str = "Not specified") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        return text if text else default
    return str(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip() for v in value if str(v).strip()]
        return "; ".join(items) if items else None
    return str(value)


def _join_list(value: Any, default: str = "Not specified", sep: str = "; ") -> str:
    if value is None:
        return default
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip() for v in value if str(v).strip()]
        return sep.join(items) if items else default
    if isinstance(value, str):
        text = value.strip()
        return text if text else default
    return str(value)


def _normalize_keywords(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _section(overrides: dict[str, Any], key: str) -> dict[str, Any]:
    section = overrides.get(key, {})
    return section if isinstance(section, dict) else {}


def _value(
    primary: dict[str, Any],
    keys: list[str],
    default: Any,
    *,
    legacy: dict[str, Any] | None = None,
    legacy_keys: list[str] | None = None,
) -> Any:
    for key in keys:
        if key in primary and primary[key] not in (None, ""):
            return primary[key]
    if legacy is not None:
        check_keys = legacy_keys or keys
        for key in check_keys:
            if key in legacy and legacy[key] not in (None, ""):
                return legacy[key]
    return default


def _format_flag_summary(flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "No flags raised."
    summaries = []
    for flag in flags:
        message = flag.get("message") or ""
        details = flag.get("details") or ""
        if message and details:
            summaries.append(f"{message} ({details})")
        elif message:
            summaries.append(message)
        elif details:
            summaries.append(details)
    return "; ".join(summaries) if summaries else "Flags raised."


def _default_data_description(results: AuditResults) -> str:
    overview = results.descriptive_stats.get("cohort_overview", {})
    n_total = overview.get("n_total")
    prevalence = overview.get("prevalence_pct")
    if n_total is None and prevalence is None:
        return "Not specified"
    parts = []
    if n_total is not None:
        parts.append(f"n={n_total}")
    if prevalence is not None:
        parts.append(f"prevalence={prevalence}")
    return ", ".join(parts)


def _format_metric_result(pairs: list[tuple[str, Any]]) -> str:
    formatted = []
    for label, value in pairs:
        if value is None:
            formatted.append(f"{label}=N/A")
        elif isinstance(value, float):
            formatted.append(f"{label}={value:.4f}")
        else:
            formatted.append(f"{label}={value}")
    return "; ".join(formatted) if formatted else "Not specified"


def build_chai_model_card_metadata(results: AuditResults) -> dict[str, Any]:
    config = results.config
    run_timestamp = results.run_timestamp or config.report_date or "Not specified"
    return {
        "schema_version": CHAI_SCHEMA_VERSION,
        "schema_url": CHAI_SCHEMA_REPO,
        "template_url": CHAI_TEMPLATE_URL,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "audit_id": results.audit_id,
        "run_timestamp": run_timestamp,
    }


def build_chai_model_card_payload(results: AuditResults) -> dict[str, Any]:
    """Build XSD-aligned payload for CHAI Applied Model Card (v0.1)."""
    config = results.config
    overrides = getattr(config, "model_card", {}) or {}
    run_timestamp = results.run_timestamp or config.report_date or "Not specified"

    model_overview = _section(overrides, "model_overview")
    basic_overrides = _section(overrides, "basic_info")
    release_overrides = _section(overrides, "release_info")
    summary_overrides = _section(overrides, "model_summary")
    uses_overrides = _section(overrides, "uses_and_directions")
    warnings_overrides = _section(overrides, "warnings")
    trust_overrides = _section(overrides, "trust_ingredients")
    resources_overrides = _section(overrides, "resources")
    key_metrics_overrides = _section(overrides, "key_metrics")
    bibliography_override = overrides.get("bibliography")

    transparency_overrides = _section(overrides, "transparency_information")
    trust_transparency = _section(trust_overrides, "transparency_information")
    transparency_overrides = transparency_overrides or trust_transparency

    ai_system_facts = _section(trust_overrides, "ai_system_facts") or trust_overrides

    gov = results.governance_recommendation or {}
    perf = results.overall_performance
    disc = perf.get("discrimination", {})
    cal = perf.get("calibration", {})
    cls = perf.get("classification_at_threshold", {})

    fairness_attrs = ", ".join(results.fairness_metrics.keys()) if results.fairness_metrics else "N/A"

    usefulness_defaults = _format_metric_result(
        [
            ("AUROC", disc.get("auroc")),
            ("AUPRC", disc.get("auprc")),
            ("Threshold", results.threshold),
        ]
    )
    fairness_defaults = _format_metric_result(
        [
            (
                "Primary fairness metric",
                config.primary_fairness_metric.value if config.primary_fairness_metric else "Not specified",
            ),
            ("Flags", len(results.flags)),
            ("Attributes", fairness_attrs),
        ]
    )
    safety_defaults = _format_metric_result(
        [
            ("Brier", cal.get("brier_score")),
            ("Calibration slope", cal.get("calibration_slope")),
            ("Sensitivity", cls.get("sensitivity")),
            ("Specificity", cls.get("specificity")),
        ]
    )

    payload: dict[str, Any] = {
        "BasicInfo": {
            "ModelName": _as_text(
                _value(
                    basic_overrides,
                    ["model_name", "ModelName"],
                    config.model_name,
                    legacy=model_overview,
                    legacy_keys=["name"],
                )
            ),
            "ModelDeveloper": _as_text(
                _value(
                    basic_overrides,
                    ["model_developer", "ModelDeveloper", "developer"],
                    config.organization_name or "Not specified",
                    legacy=model_overview,
                    legacy_keys=["developer"],
                )
            ),
            "DeveloperContact": _as_text(
                _value(
                    basic_overrides,
                    ["developer_contact", "DeveloperContact", "contact"],
                    "Not specified",
                    legacy=model_overview,
                    legacy_keys=["inquiries_or_report_issue"],
                )
            ),
        },
        "ReleaseInfo": {
            "ReleaseStage": _as_text(
                _value(
                    release_overrides,
                    ["release_stage", "ReleaseStage"],
                    "Not specified",
                    legacy=model_overview,
                    legacy_keys=["release_stage"],
                )
            ),
            "ReleaseDate": _as_text(
                _value(
                    release_overrides,
                    ["release_date", "ReleaseDate"],
                    run_timestamp,
                    legacy=model_overview,
                    legacy_keys=["release_date"],
                )
            ),
            "ReleaseVersion": _as_text(
                _value(
                    release_overrides,
                    ["release_version", "ReleaseVersion", "version"],
                    config.model_version,
                    legacy=model_overview,
                    legacy_keys=["version"],
                )
            ),
            "GlobalAvailability": _as_text(
                _value(
                    release_overrides,
                    ["global_availability", "GlobalAvailability"],
                    "Not specified",
                    legacy=model_overview,
                    legacy_keys=["global_availability"],
                )
            ),
        },
        "ModelSummary": {
            "Summary": _as_text(
                _value(
                    summary_overrides,
                    ["summary", "Summary"],
                    f"Governance status: {gov.get('status', 'N/A')}. {gov.get('advisory', '')}".strip(),
                    legacy=model_overview,
                    legacy_keys=["summary"],
                )
            ),
            "Keywords": _normalize_keywords(
                _value(
                    summary_overrides,
                    ["keywords", "Keywords"],
                    model_overview.get("keywords", []),
                )
            ),
        },
        "UsesAndDirections": {
            "IntendedUseAndWorkflow": _as_text(
                _value(
                    uses_overrides,
                    ["intended_use_and_workflow", "IntendedUseAndWorkflow"],
                    config.intended_use or "Not specified",
                )
            ),
            "PrimaryIntendedUsers": _as_text(
                _value(
                    uses_overrides,
                    ["primary_intended_users", "PrimaryIntendedUsers"],
                    "Not specified",
                )
            ),
            "HowToUse": _as_text(
                _value(
                    uses_overrides,
                    ["how_to_use", "HowToUse"],
                    "Not specified",
                )
            ),
            "TargetedPatientPopulation": _as_text(
                _value(
                    uses_overrides,
                    ["targeted_patient_population", "TargetedPatientPopulation"],
                    config.intended_population or "Not specified",
                )
            ),
            "CautionedOutOfScopeSettings": _join_list(
                _value(
                    uses_overrides,
                    ["cautioned_out_of_scope_settings", "CautionedOutOfScopeSettings"],
                    config.out_of_scope,
                )
            ),
        },
        "Warnings": {
            "KnownRisksAndLimitations": _as_text(
                _value(
                    warnings_overrides,
                    ["known_risks_and_limitations", "KnownRisksAndLimitations"],
                    _format_flag_summary(results.flags),
                )
            ),
            "KnownBiasesOrEthicalConsiderations": _as_text(
                _value(
                    warnings_overrides,
                    [
                        "known_biases_or_ethical_considerations",
                        "KnownBiasesOrEthicalConsiderations",
                    ],
                    config.fairness_justification or "Not specified",
                )
            ),
            "ClinicalRiskLevel": _as_text(
                _value(
                    warnings_overrides,
                    ["clinical_risk_level", "ClinicalRiskLevel"],
                    "Not specified",
                )
            ),
        },
        "TrustIngredients": {
            "AISystemFacts": {
                "OutcomesAndOutputs": _as_text(
                    _value(
                        ai_system_facts,
                        ["outcomes_and_outputs", "OutcomesAndOutputs"],
                        "Not specified",
                    )
                ),
                "ModelType": _as_text(
                    _value(
                        ai_system_facts,
                        ["model_type", "ModelType"],
                        config.model_type.value if config.model_type else "Not specified",
                    )
                ),
                "InputDataSource": _as_text(
                    _value(
                        ai_system_facts,
                        ["input_data_source", "InputDataSource"],
                        "Not specified",
                    )
                ),
                "OutputAndInputDataTypes": _as_text(
                    _value(
                        ai_system_facts,
                        ["output_and_input_data_types", "OutputAndInputDataTypes"],
                        "Not specified",
                    )
                ),
                "DevelopmentDataCharacterization": _as_text(
                    _value(
                        ai_system_facts,
                        ["development_data_characterization", "DevelopmentDataCharacterization"],
                        "Not specified",
                    )
                ),
                "BiasMitigationApproaches": _as_text(
                    _value(
                        ai_system_facts,
                        ["bias_mitigation_approaches", "BiasMitigationApproaches"],
                        "Not specified",
                    )
                ),
                "OngoingMaintenance": _as_text(
                    _value(
                        ai_system_facts,
                        ["ongoing_maintenance", "OngoingMaintenance"],
                        "Not specified",
                    )
                ),
            },
            "TransparencyInformation": {
                "FundingSource": _as_text(
                    _value(
                        transparency_overrides,
                        ["funding_source", "FundingSource"],
                        "Not specified",
                    )
                ),
                "StakeholdersConsulted": _as_text(
                    _value(
                        transparency_overrides,
                        ["stakeholders_consulted", "StakeholdersConsulted"],
                        "Not specified",
                    )
                ),
            },
        },
        "KeyMetrics": {
            "UsefulnessUsabilityEfficacy": {
                "MetricGoal": _as_text(
                    _value(
                        _section(key_metrics_overrides, "usefulness_usability_efficacy"),
                        ["metric_goal", "MetricGoal", "goal_of_metrics"],
                        "Assess clinical usefulness and model discrimination.",
                    )
                ),
                "Result": _as_text(
                    _value(
                        _section(key_metrics_overrides, "usefulness_usability_efficacy"),
                        ["result", "Result"],
                        usefulness_defaults,
                    )
                ),
                "Interpretation": _as_text(
                    _value(
                        _section(key_metrics_overrides, "usefulness_usability_efficacy"),
                        ["interpretation", "Interpretation"],
                        gov.get("status", "Not specified"),
                    )
                ),
                "TestType": _as_text(
                    _value(
                        _section(key_metrics_overrides, "usefulness_usability_efficacy"),
                        ["test_type", "TestType"],
                        "Not specified",
                    )
                ),
                "TestingDataDescription": _as_text(
                    _value(
                        _section(key_metrics_overrides, "usefulness_usability_efficacy"),
                        ["testing_data_description", "TestingDataDescription"],
                        _default_data_description(results),
                    )
                ),
                "ValidationProcessAndJustification": _as_text(
                    _value(
                        _section(key_metrics_overrides, "usefulness_usability_efficacy"),
                        ["validation_process_and_justification", "ValidationProcessAndJustification"],
                        config.fairness_justification or "Not specified",
                    )
                ),
            },
            "FairnessEquity": {
                "MetricGoal": _as_text(
                    _value(
                        _section(key_metrics_overrides, "fairness_equity"),
                        ["metric_goal", "MetricGoal", "goal_of_metrics"],
                        "Assess parity across protected groups.",
                    )
                ),
                "Result": _as_text(
                    _value(
                        _section(key_metrics_overrides, "fairness_equity"),
                        ["result", "Result"],
                        fairness_defaults,
                    )
                ),
                "Interpretation": _as_text(
                    _value(
                        _section(key_metrics_overrides, "fairness_equity"),
                        ["interpretation", "Interpretation"],
                        gov.get("advisory", "Not specified"),
                    )
                ),
                "TestType": _as_text(
                    _value(
                        _section(key_metrics_overrides, "fairness_equity"),
                        ["test_type", "TestType"],
                        "Not specified",
                    )
                ),
                "TestingDataDescription": _as_text(
                    _value(
                        _section(key_metrics_overrides, "fairness_equity"),
                        ["testing_data_description", "TestingDataDescription"],
                        _default_data_description(results),
                    )
                ),
                "ValidationProcessAndJustification": _as_text(
                    _value(
                        _section(key_metrics_overrides, "fairness_equity"),
                        ["validation_process_and_justification", "ValidationProcessAndJustification"],
                        config.fairness_justification or "Not specified",
                    )
                ),
            },
            "SafetyReliability": {
                "MetricGoal": _as_text(
                    _value(
                        _section(key_metrics_overrides, "safety_reliability"),
                        ["metric_goal", "MetricGoal", "goal_of_metrics"],
                        "Assess calibration and error tradeoffs.",
                    )
                ),
                "Result": _as_text(
                    _value(
                        _section(key_metrics_overrides, "safety_reliability"),
                        ["result", "Result"],
                        safety_defaults,
                    )
                ),
                "Interpretation": _as_text(
                    _value(
                        _section(key_metrics_overrides, "safety_reliability"),
                        ["interpretation", "Interpretation"],
                        gov.get("status", "Not specified"),
                    )
                ),
                "TestType": _as_text(
                    _value(
                        _section(key_metrics_overrides, "safety_reliability"),
                        ["test_type", "TestType"],
                        "Not specified",
                    )
                ),
                "TestingDataDescription": _as_text(
                    _value(
                        _section(key_metrics_overrides, "safety_reliability"),
                        ["testing_data_description", "TestingDataDescription"],
                        _default_data_description(results),
                    )
                ),
                "ValidationProcessAndJustification": _as_text(
                    _value(
                        _section(key_metrics_overrides, "safety_reliability"),
                        ["validation_process_and_justification", "ValidationProcessAndJustification"],
                        "Not specified",
                    )
                ),
            },
        },
        "Resources": {
            "PeerReviewedPublications": _as_text(
                _value(
                    resources_overrides,
                    ["peer_reviewed_publications", "PeerReviewedPublications"],
                    "Not specified",
                )
            ),
            "PatientConsentOrDisclosure": _as_text(
                _value(
                    resources_overrides,
                    ["patient_consent_or_disclosure", "PatientConsentOrDisclosure"],
                    "Not specified",
                )
            ),
        },
        "Bibliography": _as_text(bibliography_override, "Not specified"),
    }

    regulatory_approval = _optional_text(
        _value(
            release_overrides,
            ["regulatory_approval", "RegulatoryApproval"],
            model_overview.get("regulatory_approval"),
        )
    )
    if regulatory_approval:
        payload["ReleaseInfo"]["RegulatoryApproval"] = regulatory_approval

    foundation_models = _optional_text(
        _value(
            ai_system_facts,
            ["foundation_models", "FoundationModels", "foundation_models_used"],
            None,
        )
    )
    if foundation_models:
        payload["TrustIngredients"]["AISystemFacts"]["FoundationModels"] = foundation_models

    security = _optional_text(_value(ai_system_facts, ["security", "Security"], None))
    if security:
        payload["TrustIngredients"]["AISystemFacts"]["Security"] = security

    transparency = _optional_text(_value(ai_system_facts, ["transparency", "Transparency"], None))
    if transparency:
        payload["TrustIngredients"]["AISystemFacts"]["Transparency"] = transparency

    third_party = _optional_text(
        _value(transparency_overrides, ["third_party_information", "ThirdPartyInformation"], None)
    )
    if third_party:
        payload["TrustIngredients"]["TransparencyInformation"][
            "ThirdPartyInformation"
        ] = third_party

    evaluation_refs = _optional_text(
        _value(resources_overrides, ["evaluation_references", "EvaluationReferences"], None)
    )
    if evaluation_refs:
        payload["Resources"]["EvaluationReferences"] = evaluation_refs

    clinical_trial = _optional_text(
        _value(resources_overrides, ["clinical_trial", "ClinicalTrial"], None)
    )
    if clinical_trial:
        payload["Resources"]["ClinicalTrial"] = clinical_trial

    reimbursement = _optional_text(
        _value(resources_overrides, ["reimbursement_status", "ReimbursementStatus"], None)
    )
    if reimbursement:
        payload["Resources"]["ReimbursementStatus"] = reimbursement

    return payload


def build_chai_model_card_json(results: AuditResults) -> dict[str, Any]:
    return {
        "metadata": build_chai_model_card_metadata(results),
        "AppliedModelCard": build_chai_model_card_payload(results),
    }


def _ns_tag(tag: str) -> str:
    return f"{{{CHAI_NAMESPACE}}}{tag}"


def _add_text(parent: ET.Element, tag: str, value: str | None) -> None:
    child = ET.SubElement(parent, _ns_tag(tag))
    if value is not None:
        child.text = value


def _build_xml(payload: dict[str, Any]) -> bytes:
    ET.register_namespace("", CHAI_NAMESPACE)
    root = ET.Element(_ns_tag("AppliedModelCard"))

    basic = ET.SubElement(root, _ns_tag("BasicInfo"))
    _add_text(basic, "ModelName", payload["BasicInfo"]["ModelName"])
    _add_text(basic, "ModelDeveloper", payload["BasicInfo"]["ModelDeveloper"])
    _add_text(basic, "DeveloperContact", payload["BasicInfo"]["DeveloperContact"])

    release = ET.SubElement(root, _ns_tag("ReleaseInfo"))
    _add_text(release, "ReleaseStage", payload["ReleaseInfo"]["ReleaseStage"])
    _add_text(release, "ReleaseDate", payload["ReleaseInfo"]["ReleaseDate"])
    _add_text(release, "ReleaseVersion", payload["ReleaseInfo"]["ReleaseVersion"])
    _add_text(release, "GlobalAvailability", payload["ReleaseInfo"]["GlobalAvailability"])
    if "RegulatoryApproval" in payload["ReleaseInfo"]:
        _add_text(release, "RegulatoryApproval", payload["ReleaseInfo"]["RegulatoryApproval"])

    summary = ET.SubElement(root, _ns_tag("ModelSummary"))
    _add_text(summary, "Summary", payload["ModelSummary"]["Summary"])
    keywords_el = ET.SubElement(summary, _ns_tag("Keywords"))
    for keyword in payload["ModelSummary"]["Keywords"]:
        _add_text(keywords_el, "Keyword", keyword)

    uses = ET.SubElement(root, _ns_tag("UsesAndDirections"))
    _add_text(uses, "IntendedUseAndWorkflow", payload["UsesAndDirections"]["IntendedUseAndWorkflow"])
    _add_text(uses, "PrimaryIntendedUsers", payload["UsesAndDirections"]["PrimaryIntendedUsers"])
    _add_text(uses, "HowToUse", payload["UsesAndDirections"]["HowToUse"])
    _add_text(
        uses,
        "TargetedPatientPopulation",
        payload["UsesAndDirections"]["TargetedPatientPopulation"],
    )
    _add_text(
        uses,
        "CautionedOutOfScopeSettings",
        payload["UsesAndDirections"]["CautionedOutOfScopeSettings"],
    )

    warnings = ET.SubElement(root, _ns_tag("Warnings"))
    _add_text(warnings, "KnownRisksAndLimitations", payload["Warnings"]["KnownRisksAndLimitations"])
    _add_text(
        warnings,
        "KnownBiasesOrEthicalConsiderations",
        payload["Warnings"]["KnownBiasesOrEthicalConsiderations"],
    )
    _add_text(warnings, "ClinicalRiskLevel", payload["Warnings"]["ClinicalRiskLevel"])

    trust = ET.SubElement(root, _ns_tag("TrustIngredients"))
    ai_facts = ET.SubElement(trust, _ns_tag("AISystemFacts"))
    ai_payload = payload["TrustIngredients"]["AISystemFacts"]
    _add_text(ai_facts, "OutcomesAndOutputs", ai_payload["OutcomesAndOutputs"])
    _add_text(ai_facts, "ModelType", ai_payload["ModelType"])
    if "FoundationModels" in ai_payload:
        _add_text(ai_facts, "FoundationModels", ai_payload["FoundationModels"])
    _add_text(ai_facts, "InputDataSource", ai_payload["InputDataSource"])
    _add_text(ai_facts, "OutputAndInputDataTypes", ai_payload["OutputAndInputDataTypes"])
    _add_text(
        ai_facts,
        "DevelopmentDataCharacterization",
        ai_payload["DevelopmentDataCharacterization"],
    )
    _add_text(ai_facts, "BiasMitigationApproaches", ai_payload["BiasMitigationApproaches"])
    _add_text(ai_facts, "OngoingMaintenance", ai_payload["OngoingMaintenance"])
    if "Security" in ai_payload:
        _add_text(ai_facts, "Security", ai_payload["Security"])
    if "Transparency" in ai_payload:
        _add_text(ai_facts, "Transparency", ai_payload["Transparency"])

    transparency = ET.SubElement(trust, _ns_tag("TransparencyInformation"))
    transparency_payload = payload["TrustIngredients"]["TransparencyInformation"]
    _add_text(transparency, "FundingSource", transparency_payload["FundingSource"])
    if "ThirdPartyInformation" in transparency_payload:
        _add_text(
            transparency,
            "ThirdPartyInformation",
            transparency_payload["ThirdPartyInformation"],
        )
    _add_text(transparency, "StakeholdersConsulted", transparency_payload["StakeholdersConsulted"])

    metrics = ET.SubElement(root, _ns_tag("KeyMetrics"))
    for section_key in [
        "UsefulnessUsabilityEfficacy",
        "FairnessEquity",
        "SafetyReliability",
    ]:
        metric_payload = payload["KeyMetrics"][section_key]
        metric_el = ET.SubElement(metrics, _ns_tag(section_key))
        _add_text(metric_el, "MetricGoal", metric_payload["MetricGoal"])
        _add_text(metric_el, "Result", metric_payload["Result"])
        _add_text(metric_el, "Interpretation", metric_payload["Interpretation"])
        _add_text(metric_el, "TestType", metric_payload["TestType"])
        _add_text(metric_el, "TestingDataDescription", metric_payload["TestingDataDescription"])
        _add_text(
            metric_el,
            "ValidationProcessAndJustification",
            metric_payload["ValidationProcessAndJustification"],
        )

    resources = ET.SubElement(root, _ns_tag("Resources"))
    resources_payload = payload["Resources"]
    if "EvaluationReferences" in resources_payload:
        _add_text(resources, "EvaluationReferences", resources_payload["EvaluationReferences"])
    if "ClinicalTrial" in resources_payload:
        _add_text(resources, "ClinicalTrial", resources_payload["ClinicalTrial"])
    _add_text(resources, "PeerReviewedPublications", resources_payload["PeerReviewedPublications"])
    if "ReimbursementStatus" in resources_payload:
        _add_text(resources, "ReimbursementStatus", resources_payload["ReimbursementStatus"])
    _add_text(
        resources,
        "PatientConsentOrDisclosure",
        resources_payload["PatientConsentOrDisclosure"],
    )

    _add_text(root, "Bibliography", payload["Bibliography"])

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _validate_xml(xml_bytes: bytes) -> None:
    try:
        import xmlschema
    except ImportError:
        logger.warning(
            "xmlschema not installed; skipping CHAI model card validation. "
            "Install with: pip install \"faircareai[compliance]\""
        )
        return

    schema_path = resources.files("faircareai.data").joinpath("chai/v0.1/schema.xsd")
    try:
        schema = xmlschema.XMLSchema(str(schema_path))
        schema.validate(BytesIO(xml_bytes))
    except Exception as exc:  # xmlschema raises various errors
        raise ConfigurationError(
            "chai_model_card",
            f"CHAI model card XML failed schema validation: {exc}",
        ) from exc


def generate_chai_model_card_xml(results: AuditResults, path: str | Path) -> Path:
    """Generate a CHAI Applied Model Card XML file (v0.1 schema)."""
    path = Path(path)
    payload = build_chai_model_card_payload(results)
    xml_bytes = _build_xml(payload)
    _validate_xml(xml_bytes)
    path.write_bytes(xml_bytes)
    return path


def generate_chai_model_card_json(results: AuditResults, path: str | Path) -> Path:
    """Generate a CHAI Applied Model Card JSON file (debug/reference)."""
    path = Path(path)
    payload = build_chai_model_card_json(results)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
