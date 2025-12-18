"""FairCareAI Reports Module"""

from faircareai.reports.generator import (
    AuditSummary,
    generate_html_report,
    generate_pdf_report,
    generate_pptx_deck,
)

__all__ = [
    "AuditSummary",
    "generate_pdf_report",
    "generate_pptx_deck",
    "generate_html_report",
]
