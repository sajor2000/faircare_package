"""
FairCareAI Dashboard Components

WCAG 2.1 AA compliant components for dual-audience UX.
"""

from faircareai.dashboard.components.accessibility import (
    accessible_plotly_chart,
    announce_status_change,
    create_data_table_summary,
    create_skip_link,
    generate_chart_alt_text,
)
from faircareai.dashboard.components.audience_toggle import (
    get_audience_mode,
    get_metric_display,
    render_audience_toggle,
)
from faircareai.dashboard.components.editorial_charts import (
    add_editorial_annotations,
    create_editorial_forest_plot,
    generate_chart_headline,
)
from faircareai.dashboard.components.glossary import (
    METRIC_GLOSSARY,
    render_glossary_sidebar,
    render_glossary_tooltip,
)

__all__ = [
    # Accessibility
    "generate_chart_alt_text",
    "accessible_plotly_chart",
    "create_skip_link",
    "announce_status_change",
    "create_data_table_summary",
    # Audience toggle
    "render_audience_toggle",
    "get_audience_mode",
    "get_metric_display",
    # Glossary
    "METRIC_GLOSSARY",
    "render_glossary_tooltip",
    "render_glossary_sidebar",
    # Editorial charts
    "create_editorial_forest_plot",
    "generate_chart_headline",
    "add_editorial_annotations",
]
