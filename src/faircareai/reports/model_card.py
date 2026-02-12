"""Model card generator for FairCareAI audits (CHAI Applied Model Card aligned)."""

from __future__ import annotations

from pathlib import Path

from faircareai.core.results import AuditResults
from faircareai.reports.chai_model_card import (
    build_chai_model_card_metadata,
    build_chai_model_card_payload,
)


def _format_value(value: object) -> str:
    if value is None:
        return "Not specified"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "Not specified"
    if isinstance(value, dict):
        return "See structured section"
    return str(value)


def _section_lines(title: str, rows: list[tuple[str, object]]) -> list[str]:
    lines = [f"## {title}"]
    for label, value in rows:
        lines.append(f"- **{label}**: {_format_value(value)}")
    lines.append("")
    return lines


def _title_from_camel(text: str) -> str:
    if "_" in text:
        return text.replace("_", " ").title()
    spaced = []
    for idx, char in enumerate(text):
        if idx > 0 and char.isupper() and text[idx - 1].islower():
            spaced.append(" ")
        spaced.append(char)
    return "".join(spaced).title()


def generate_model_card_markdown(results: AuditResults, path: str | Path) -> Path:
    """Generate a CHAI Applied Model Card-aligned Markdown report."""
    path = Path(path)
    payload = build_chai_model_card_payload(results)
    metadata = build_chai_model_card_metadata(results)

    lines: list[str] = ["# CHAI Applied Model Card (FairCareAI)", ""]

    lines += _section_lines(
        "Schema Alignment",
        [
            ("Schema version", metadata.get("schema_version")),
            ("Schema URL", metadata.get("schema_url")),
            ("Template URL", metadata.get("template_url")),
            ("Generated at", metadata.get("generated_at")),
            ("Audit ID", metadata.get("audit_id")),
            ("Run timestamp", metadata.get("run_timestamp")),
        ],
    )

    basic = payload.get("BasicInfo", {})
    release = payload.get("ReleaseInfo", {})
    summary = payload.get("ModelSummary", {})
    lines += _section_lines(
        "Basic Info",
        [
            ("Model name", basic.get("ModelName")),
            ("Model developer", basic.get("ModelDeveloper")),
            ("Developer contact", basic.get("DeveloperContact")),
        ],
    )

    lines += _section_lines(
        "Release Info",
        [
            ("Release stage", release.get("ReleaseStage")),
            ("Release date", release.get("ReleaseDate")),
            ("Release version", release.get("ReleaseVersion")),
            ("Global availability", release.get("GlobalAvailability")),
            ("Regulatory approval", release.get("RegulatoryApproval")),
        ],
    )

    lines += _section_lines(
        "Model Summary",
        [
            ("Summary", summary.get("Summary")),
            ("Keywords", summary.get("Keywords")),
        ],
    )

    uses = payload.get("UsesAndDirections", {})
    lines += _section_lines(
        "Uses and Directions",
        [
            ("Intended use and workflow", uses.get("IntendedUseAndWorkflow")),
            ("Primary intended users", uses.get("PrimaryIntendedUsers")),
            ("How to use", uses.get("HowToUse")),
            ("Targeted patient population", uses.get("TargetedPatientPopulation")),
            ("Cautioned out-of-scope settings", uses.get("CautionedOutOfScopeSettings")),
        ],
    )

    warnings = payload.get("Warnings", {})
    lines += _section_lines(
        "Warnings",
        [
            ("Known risks and limitations", warnings.get("KnownRisksAndLimitations")),
            ("Known biases or ethical considerations", warnings.get("KnownBiasesOrEthicalConsiderations")),
            ("Clinical risk level", warnings.get("ClinicalRiskLevel")),
        ],
    )

    trust = payload.get("TrustIngredients", {})
    ai_facts = trust.get("AISystemFacts", {})
    lines += _section_lines(
        "Trust Ingredients",
        [
            ("Outcomes and outputs", ai_facts.get("OutcomesAndOutputs")),
            ("Model type", ai_facts.get("ModelType")),
            ("Foundation models", ai_facts.get("FoundationModels")),
            ("Input data source", ai_facts.get("InputDataSource")),
            ("Output and input data types", ai_facts.get("OutputAndInputDataTypes")),
            ("Development data characterization", ai_facts.get("DevelopmentDataCharacterization")),
            ("Bias mitigation approaches", ai_facts.get("BiasMitigationApproaches")),
            ("Ongoing maintenance", ai_facts.get("OngoingMaintenance")),
            ("Security", ai_facts.get("Security")),
            ("Transparency", ai_facts.get("Transparency")),
        ],
    )

    transparency = trust.get("TransparencyInformation", {})
    lines += _section_lines(
        "Transparency Information",
        [
            ("Funding source", transparency.get("FundingSource")),
            ("Third-party information", transparency.get("ThirdPartyInformation")),
            ("Stakeholders consulted", transparency.get("StakeholdersConsulted")),
        ],
    )

    key_metrics = payload.get("KeyMetrics", {})
    lines.append("## Key Metrics")
    for section, metrics in key_metrics.items():
        lines.append(f"### {_title_from_camel(section)}")
        if isinstance(metrics, dict):
            lines.append(f"- **Metric goal**: {_format_value(metrics.get('MetricGoal'))}")
            lines.append(f"- **Result**: {_format_value(metrics.get('Result'))}")
            lines.append(f"- **Interpretation**: {_format_value(metrics.get('Interpretation'))}")
            lines.append(f"- **Test type**: {_format_value(metrics.get('TestType'))}")
            lines.append(
                f"- **Testing data description**: {_format_value(metrics.get('TestingDataDescription'))}"
            )
            lines.append(
                "- **Validation process and justification**: "
                f"{_format_value(metrics.get('ValidationProcessAndJustification'))}"
            )
        else:
            lines.append(f"- {_format_value(metrics)}")
        lines.append("")

    resources = payload.get("Resources", {})
    lines += _section_lines(
        "Resources",
        [
            ("Evaluation references", resources.get("EvaluationReferences")),
            ("Clinical trial", resources.get("ClinicalTrial")),
            ("Peer reviewed publications", resources.get("PeerReviewedPublications")),
            ("Reimbursement status", resources.get("ReimbursementStatus")),
            ("Patient consent or disclosure", resources.get("PatientConsentOrDisclosure")),
        ],
    )

    lines += _section_lines(
        "Bibliography",
        [
            ("Bibliography", payload.get("Bibliography")),
        ],
    )

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path
