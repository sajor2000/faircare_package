"""PowerPoint export options for FairCareAI reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PptxOptions:
    """Customize PPTX content and layout."""

    include_title_slide: bool = True
    include_exec_summary: bool = True
    include_key_findings: bool = True
    include_methodology: bool = True

    include_exec_summary_chart: bool = True
    include_scorecard_chart: bool = True
    include_overall_charts: bool = True
    include_subgroup_charts: bool = True
    include_vancalster_dashboard: bool = False

    slide_order: list[str] | None = None
    """Optional order for slide sections.

    Valid keys: title, summary, key_findings, methodology,
    exec_summary_chart, scorecard_chart, overall_charts,
    subgroup_charts, vancalster_dashboard.
    """

    logo_path: str | Path | None = None
    """Optional logo path to place on the title slide."""

    footer_text: str | None = None
    """Optional footer text (overrides default audit ID/date)."""
