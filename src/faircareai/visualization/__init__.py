"""
FairCareAI Visualization Module

Publication-ready, colorblind-safe visualizations supporting two personas:
- Data Scientist: Interactive Plotly charts with full statistical details
- Clinical Stakeholder: Great Tables scorecards with plain-language summaries

Includes Van Calster et al. (2025) recommended visualizations:
- AUROC Forest Plot by Subgroup
- Calibration Plots by Subgroup
- Decision Curves (Net Benefit) by Subgroup
- Risk Distribution Plots by Subgroup

Export utilities for publication-ready figures:
- PNG, PDF, SVG, HTML export with consistent settings
- Batch export to multiple formats
"""

from faircareai.visualization.exporters import (
    ExportFormat,
    FigureExportError,
    export_altair_chart,
    export_figure_bundle,
    export_plotly_figure,
    get_recommended_export_settings,
)
from faircareai.visualization.themes import (
    GHOSTING_CONFIG,
    SEMANTIC_COLORS,
    TYPOGRAPHY,
    GhostingConfig,
    get_plotly_template,
    register_plotly_template,
)
from faircareai.visualization.vancalster_plots import (
    create_auroc_forest_plot,
    create_calibration_plot_by_subgroup,
    create_decision_curve_by_subgroup,
    create_risk_distribution_plot,
    create_vancalster_dashboard,
)

__all__ = [
    # Theme configuration
    "GHOSTING_CONFIG",
    "GhostingConfig",
    "SEMANTIC_COLORS",
    "TYPOGRAPHY",
    "get_plotly_template",
    "register_plotly_template",
    # Van Calster (2025) visualizations
    "create_auroc_forest_plot",
    "create_calibration_plot_by_subgroup",
    "create_decision_curve_by_subgroup",
    "create_risk_distribution_plot",
    "create_vancalster_dashboard",
    # Export utilities
    "export_plotly_figure",
    "export_altair_chart",
    "export_figure_bundle",
    "get_recommended_export_settings",
    "ExportFormat",
    "FigureExportError",
]
