"""RAIC Checkpoint 1 checklist export for FairCareAI audits."""

from __future__ import annotations

import json
from datetime import datetime
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


def generate_raic_checkpoint_1_checklist(results: AuditResults, path: str | Path) -> Path:
    """Generate a RAIC Checkpoint 1 checklist JSON export."""
    path = Path(path)
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    auto_items = _auto_evaluable_items(results)
    criteria: list[dict[str, Any]] = []
    for idx in range(1, 179):
        criteria_id = f"AC1.CR{idx}"
        if criteria_id in auto_items:
            criteria.append(auto_items[criteria_id])
            continue
        criteria.append(
            {
                "id": criteria_id,
                "summary": f"See CHAI RAIC Checkpoint 1 checklist item {criteria_id}.",
                "status": "NOT_EVALUATED",
                "evidence": "Manual review required.",
            }
        )

    checklist = {
        "raic_checkpoint": "Checkpoint 1",
        "source_url": "https://www.chai.org/workgroup/responsible-ai/responsible-ai-checklists-raic",
        "documentation_url": "https://chai.org/wp-content/uploads/2025/02/Responsible-AI-Checkpoint-1-CHAI-Responsible-AI-Checklist.pdf",
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
