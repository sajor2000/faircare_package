"""RAIC Checkpoint 1 checklist export for FairCareAI audits."""

from __future__ import annotations

import json
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

from faircareai.core.results import AuditResults


def _has_confidence_intervals(results: AuditResults) -> bool:
    disc = results.overall_performance.get("discrimination", {})
    return bool(
        disc.get("auroc_ci_95")
        or disc.get("auroc_ci_fmt")
        or (disc.get("auroc_ci_lower") is not None and disc.get("auroc_ci_upper") is not None)
    )


def _has_confusion_matrix(results: AuditResults) -> bool:
    return bool(results.overall_performance.get("confusion_matrix"))


def _has_subgroup_calibration(results: AuditResults) -> bool:
    for metrics in results.fairness_metrics.values():
        if isinstance(metrics, dict) and metrics.get("calibration_diff"):
            return True
    return False


def _has_subgroup_analysis(results: AuditResults) -> bool:
    return bool(results.subgroup_performance)


def _format_status(condition: bool, else_status: str = "NOT_EVALUATED") -> str:
    return "MET" if condition else else_status


def _auto_evaluable_items(results: AuditResults) -> dict[str, dict[str, Any]]:
    items = {
        "AC1.CR79": {
            "ls_id": "LS2.ME46",
            "summary": "Decision thresholds are defined and documented.",
            "status": _format_status(results.threshold is not None),
            "evidence": f"Primary threshold: {results.threshold}",
        },
        "AC1.CR83": {
            "ls_id": "LS2.ME12",
            "summary": "Confidence intervals reported for model performance.",
            "status": _format_status(_has_confidence_intervals(results), else_status="PARTIAL"),
            "evidence": "AUROC CI available" if _has_confidence_intervals(results) else "CI not computed",
        },
        "AC1.CR85": {
            "ls_id": "LS2.ME14",
            "summary": "Confusion matrix or equivalent error analysis included.",
            "status": _format_status(_has_confusion_matrix(results)),
            "evidence": "Confusion matrix present" if _has_confusion_matrix(results) else "Missing",
        },
        "AC1.CR88": {
            "ls_id": "LS2.ME34",
            "summary": "Calibration evaluated overall and for sensitive subgroups.",
            "status": _format_status(_has_subgroup_calibration(results), else_status="PARTIAL"),
            "evidence": (
                "Calibration by group available"
                if _has_subgroup_calibration(results)
                else "Overall only"
            ),
        },
        "AC1.CR90": {
            "ls_id": "LS2.ME15",
            "summary": "Counterfactual or sensitivity analysis for subgroup impact.",
            "status": "NOT_EVALUATED",
            "evidence": "Not automated in FairCareAI (manual review recommended).",
        },
        "AC1.CR91": {
            "ls_id": "LS2.ME17",
            "summary": "Performance evaluated across different populations.",
            "status": _format_status(_has_subgroup_analysis(results)),
            "evidence": (
                "Subgroup performance available" if _has_subgroup_analysis(results) else "Missing"
            ),
        },
        "AC1.CR92": {
            "ls_id": "LS2.ME37",
            "summary": "Calibration assessed for protected classes.",
            "status": _format_status(_has_subgroup_calibration(results), else_status="PARTIAL"),
            "evidence": (
                "Calibration differences computed"
                if _has_subgroup_calibration(results)
                else "Not computed"
            ),
        },
        "AC1.CR93": {
            "ls_id": "LS2.ME38",
            "summary": "Primary fairness metric selected and justified.",
            "status": _format_status(
                bool(
                    results.config.primary_fairness_metric
                    and results.config.fairness_justification
                )
            ),
            "evidence": results.config.fairness_justification or "Not provided",
        },
        "AC1.CR95": {
            "ls_id": "LS2.ME43",
            "summary": "Performance and parity assessed across sensitive attributes.",
            "status": _format_status(_has_subgroup_analysis(results)),
            "evidence": (
                "Subgroup fairness metrics available" if _has_subgroup_analysis(results) else "Missing"
            ),
        },
    }
    for key, value in items.items():
        value["id"] = key
    return items


def _load_raic_criteria() -> list[dict[str, Any]]:
    data_path = resources.files("faircareai.data").joinpath("raic/checkpoint_1.json")
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    criteria = payload.get("criteria", payload)
    if not isinstance(criteria, list):
        raise ValueError("RAIC checklist data is invalid or missing criteria list.")
    return criteria


def generate_raic_checkpoint_1_checklist(results: AuditResults, path: str | Path) -> Path:
    """Generate a RAIC Checkpoint 1 checklist JSON export."""
    path = Path(path)
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    auto_items = _auto_evaluable_items(results)
    catalog = _load_raic_criteria()
    criteria: list[dict[str, Any]] = []
    for item in catalog:
        criteria_id = item.get("id")
        if not criteria_id:
            continue
        entry: dict[str, Any] = {
            "id": criteria_id,
            "summary": item.get("summary", ""),
        }
        if "ls_id" in item:
            entry["ls_id"] = item["ls_id"]
        if criteria_id in auto_items:
            auto = auto_items[criteria_id]
            entry["status"] = auto["status"]
            entry["evidence"] = auto["evidence"]
            if "ls_id" in auto and "ls_id" not in entry:
                entry["ls_id"] = auto["ls_id"]
        else:
            entry["status"] = "NOT_EVALUATED"
            entry["evidence"] = "Manual review required."
        criteria.append(entry)

    checklist = {
        "raic_checkpoint": "Checkpoint 1",
        "source_url": "https://www.chai.org/workgroup/responsible-ai/responsible-ai-checklists-raic",
        "documentation_url": "https://chai.org/wp-content/uploads/2025/02/Responsible-AI-Checkpoint-1-CHAI-Responsible-AI-Checklist.pdf",
        "document_version": "v0.3",
        "last_revised": "2024-06-26",
        "generated_at": now,
        "audit_id": results.audit_id,
        "model_name": results.config.model_name,
        "model_version": results.config.model_version,
        "criteria": criteria,
        "reviewer": {
            "name": "",
            "role": "",
            "review_date": "",
            "decision": "",
            "comments": "",
        },
    }

    path.write_text(json.dumps(checklist, indent=2, default=str), encoding="utf-8")
    return path
