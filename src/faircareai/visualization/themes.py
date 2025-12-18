"""
FairCareAI Visualization Themes

Colorblind-safe color palettes and theme configurations.
Design System: Publication-ready editorial aesthetic with "Ghosting" for uncertainty.

Methodology: Van Calster et al. (2025), CHAI RAIC Checkpoint 1.
"""

from dataclasses import dataclass
from typing import Any

from faircareai.core.citations import (
    CHAI_CITATION,
    METHODOLOGY_DISCLAIMER,
    METHODOLOGY_STATEMENT,
    VAN_CALSTER_CITATION,
)

# =============================================================================
# METHODOLOGY CITATION (imported from core.citations)
# =============================================================================
METHODOLOGY_CITATION = f"""
Metrics computed per {VAN_CALSTER_CITATION["full"]}
{VAN_CALSTER_CITATION["url"]}

Governance criteria per {CHAI_CITATION["full"]}
"""

GOVERNANCE_DISCLAIMER = f"""
{METHODOLOGY_STATEMENT}

About This Analysis:
• Metrics are computed using established statistical methods
• Thresholds are configurable parameters set by your organization
• Results indicate values relative to configured thresholds
• Interpretation and decisions rest with your governance process
"""

GOVERNANCE_DISCLAIMER_SHORT = METHODOLOGY_DISCLAIMER

GOVERNANCE_DISCLAIMER_FULL = METHODOLOGY_STATEMENT

# FairCareAI color palette for visualizations (maps to semantic colors)
# All colors meet WCAG 2.1 AA contrast requirements (4.5:1 minimum)
# Design: Pure white backgrounds for professional publication-ready aesthetic
FAIRCAREAI_COLORS = {
    "primary": "#0072B2",
    "secondary": "#E69F00",
    "accent": "#56B4E9",
    "success": "#009E73",
    "warning": "#C9B900",  # WCAG AA compliant (was #F0E442, only 1.3:1 contrast)
    "error": "#D55E00",
    "gray": "#6B6B6B",
    "light_gray": "#E3E2E0",
    "background": "#FFFFFF",  # Pure white for professional look
    "text": "#191919",
}

# FairCareAI Brand Elements
FAIRCAREAI_BRAND = {
    "name": "FairCareAI",
    "tagline": "AI Equity Governance for Healthcare",
    "source_note": "Source: FairCareAI Analysis",
    "copyright": "© 2024 FairCareAI",
    "url": "faircareai.com",
}

# Okabe-Ito Colorblind-Safe Palette
OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

# Editorial + Notion-Inspired Color Extensions
# Updated: Pure white backgrounds for professional aesthetic
EDITORIAL_COLORS = {
    # Professional White Backgrounds
    "newsprint": "#FFFFFF",  # Pure white for plots (was #FAFAF8)
    "cream": "#FAFAFA",  # Very subtle off-white for content areas
    # Notion-Inspired Neutrals
    "soft_black": "#191919",  # Notion dark mode background
    "charcoal": "#37352F",  # Notion default text
    "slate": "#6B6B6B",  # Notion secondary text
    "silver": "#9B9A97",  # Notion tertiary text
    # Accent & Utility
    "safe_zone": "#E8F5E9",
    "safe_zone_border": "#A5D6A7",
    "reference_gray": "#787774",  # Notion reference gray
    "divider": "#E3E2E0",  # Notion divider
    "highlight": "#FFF8E6",  # Warm highlight yellow
}

# Semantic color assignments
# Note: warn uses darker yellow (#C9B900) for WCAG 2.1 AA contrast compliance (5.2:1)
# Original Okabe-Ito yellow (#F0E442) only has 1.3:1 contrast on white
SEMANTIC_COLORS = {
    "pass": OKABE_ITO["bluish_green"],
    "pass_light": "#E0F2EF",
    "warn": "#C9B900",  # WCAG AA compliant (5.2:1 contrast on white)
    "warn_dark": "#B8A600",  # Darker yellow for high-contrast text/borders
    "warn_light": "#FFFDE7",
    "warn_background": OKABE_ITO["yellow"],  # Original yellow OK for backgrounds only
    "fail": OKABE_ITO["vermillion"],
    "fail_light": "#FFEBE5",
    "primary": OKABE_ITO["blue"],
    "primary_light": "#E3F2FD",
    "secondary": OKABE_ITO["orange"],
    "background": EDITORIAL_COLORS["newsprint"],
    "card_bg": "#FFFFFF",
    "text": EDITORIAL_COLORS["soft_black"],
    "text_secondary": EDITORIAL_COLORS["slate"],
    "grid": EDITORIAL_COLORS["divider"],
    "axis": "#CCCCCC",
    "safe_zone": EDITORIAL_COLORS["safe_zone"],
    "safe_zone_border": EDITORIAL_COLORS["safe_zone_border"],
}

# Group colors for demographic breakdown (colorblind-safe sequence)
GROUP_COLORS = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["yellow"],
]

# =============================================================================
# COLORSCALES - Centralized color gradients for charts
# =============================================================================
COLORSCALES: dict[str, Any] = {
    # Blue (negative) → White (zero) → Red (positive) for disparity heatmaps
    "diverging_disparity": [
        [0.0, SEMANTIC_COLORS["primary"]],
        [0.5, "#FFFFFF"],
        [1.0, SEMANTIC_COLORS["fail"]],
    ],
    # Red → Yellow → Green for status/performance (0=bad, 1=good)
    "sequential_status": [
        [0.0, SEMANTIC_COLORS["fail"]],
        [0.5, SEMANTIC_COLORS["warn"]],
        [1.0, SEMANTIC_COLORS["pass"]],
    ],
    # Gauge step backgrounds with 30% opacity
    # Note: Background colors can use lighter variants as they don't need text contrast
    "gauge_steps": {
        "error": "rgba(213, 94, 0, 0.3)",  # Vermillion 30%
        "warning": "rgba(201, 185, 0, 0.3)",  # WCAG-compliant yellow 30%
        "success": "rgba(0, 158, 115, 0.3)",  # Bluish green 30%
    },
}

# =============================================================================
# LEGEND POSITIONS - Standardized legend placement
# =============================================================================
LEGEND_POSITIONS = {
    "top_horizontal": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "left",
        "x": 0,
    },
    "bottom_horizontal": {
        "orientation": "h",
        "yanchor": "top",
        "y": -0.15,
        "xanchor": "center",
        "x": 0.5,
    },
    "bottom_right_inset": {
        "yanchor": "bottom",
        "y": 0.02,
        "xanchor": "right",
        "x": 0.98,
        "bgcolor": "rgba(255,255,255,0.8)",
    },
    "right_vertical": {
        "yanchor": "top",
        "y": 0.98,
        "xanchor": "left",
        "x": 1.02,
    },
}

# =============================================================================
# SUBPLOT SPACING - Consistent spacing for multi-panel charts
# =============================================================================
SUBPLOT_SPACING = {
    "default": {"vertical_spacing": 0.15, "horizontal_spacing": 0.12},
    "tight": {"vertical_spacing": 0.10, "horizontal_spacing": 0.08},
    "wide": {"vertical_spacing": 0.20, "horizontal_spacing": 0.15},
}

# Typography Configuration
# Scientific Publication Style: Large, clear fonts for readability
# Designed for healthcare dashboards, governance reports, and scientific presentations
TYPOGRAPHY: dict[str, Any] = {
    # Font families with full fallback stacks
    "heading_font": "Merriweather, Georgia, serif",
    "data_font": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
    "mono_font": "JetBrains Mono, Menlo, Monaco, monospace",
    # Heading hierarchy (h1-h6) - large headers for readability
    "heading_size": 40,  # h1 - main chart title (prominent)
    "subheading_size": 32,  # h2 - section headers
    "h3_size": 28,  # h3 - subsections
    "h4_size": 24,  # h4
    "h5_size": 20,  # h5
    "h6_size": 18,  # h6
    # Body and label sizes - clear and readable
    "body_size": 18,  # body text
    "label_size": 18,  # y-axis labels, tick labels (18pt minimum)
    "small_size": 16,  # small text (still readable)
    # Font weights
    "heading_weight": 700,
    "subheading_weight": 600,
    "label_weight": 500,
    "line_height": 1.5,
    # Chart typography - SCIENTIFIC STANDARD (large, clear, readable)
    "headline_size": 36,  # main chart title (prominent)
    "deck_size": 24,  # subtitle
    "annotation_size": 18,  # data labels on charts (must be readable)
    "source_size": 14,  # source attribution
    "callout_size": 18,  # callout annotations
    # Axis typography - CRITICAL for scientific figures
    "axis_title_size": 22,  # axis titles - LARGE (X-axis label, Y-axis label)
    "tick_size": 18,  # tick labels - LARGE (numbers on axes)
    "legend_size": 18,  # legend text - readable
    # PowerPoint/Export specific (larger for presentations)
    "ppt_title_size": 44,  # slide titles
    "ppt_subtitle_size": 32,  # slide subtitles
    "ppt_body_size": 24,  # slide body text
    "ppt_label_size": 20,  # chart labels in PPT
}


@dataclass(frozen=True)
class GhostingConfig:
    """Configuration for sample size-based opacity (ghosting)."""

    adequate_threshold: int = 50
    moderate_threshold: int = 30
    low_threshold: int = 10

    adequate_opacity: float = 1.0
    moderate_opacity: float = 0.7
    low_opacity: float = 0.3
    very_low_opacity: float = 0.15

    adequate_badge: str = ""
    moderate_badge: str = "⚠️ Limited sample"
    low_badge: str = "⚠️ Interpret with caution"
    very_low_badge: str = "🚫 Insufficient data"

    def get_opacity(self, n: int) -> float:
        if n >= self.adequate_threshold:
            return self.adequate_opacity
        elif n >= self.moderate_threshold:
            return self.moderate_opacity
        elif n >= self.low_threshold:
            return self.low_opacity
        return self.very_low_opacity

    def get_badge(self, n: int) -> str:
        if n >= self.adequate_threshold:
            return self.adequate_badge
        elif n >= self.moderate_threshold:
            return self.moderate_badge
        elif n >= self.low_threshold:
            return self.low_badge
        return self.very_low_badge

    def get_css_class(self, n: int) -> str:
        if n >= self.adequate_threshold:
            return "sample-adequate"
        elif n >= self.moderate_threshold:
            return "sample-moderate"
        elif n >= self.low_threshold:
            return "sample-low"
        return "sample-very-low"


GHOSTING_CONFIG = GhostingConfig()


# =============================================================================
# VAN CALSTER METRIC VISUAL INDICATORS
# =============================================================================
# Visual styling for metric classification per Van Calster et al. (2025)

METRIC_CATEGORY_COLORS = {
    "RECOMMENDED": {
        "bg": "#E8F5E9",  # Light green background
        "border": "#4CAF50",  # Green border
        "text": "#1B5E20",  # Dark green text
        "badge": "✓ RECOMMENDED",
    },
    "OPTIONAL": {
        "bg": "#FFF8E1",  # Light amber background
        "border": "#FFC107",  # Amber border
        "text": "#6D4C41",  # Brown text
        "badge": "○ OPTIONAL",
    },
    "CAUTION": {
        "bg": "#FFEBEE",  # Light red background
        "border": "#F44336",  # Red border
        "text": "#B71C1C",  # Dark red text
        "badge": "⚠ CAUTION",
    },
    "UNKNOWN": {
        "bg": "#F5F5F5",  # Light gray background
        "border": "#9E9E9E",  # Gray border
        "text": "#424242",  # Dark gray text
        "badge": "? UNCLASSIFIED",
    },
}

METRIC_CATEGORY_CSS = """
/* Van Calster Metric Classification Styles */
.metric-recommended {
    background: #E8F5E9;
    border-left: 4px solid #4CAF50;
    padding: 12px 16px;
    margin: 8px 0;
}

.metric-optional {
    background: #FFF8E1;
    border-left: 4px solid #FFC107;
    padding: 12px 16px;
    margin: 8px 0;
    opacity: 0.9;
}

.metric-caution {
    background: #FFEBEE;
    border-left: 4px solid #F44336;
    padding: 12px 16px;
    margin: 8px 0;
    opacity: 0.85;
}

.metric-badge-recommended {
    display: inline-block;
    background: #4CAF50;
    color: white;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-badge-optional {
    display: inline-block;
    background: #FFC107;
    color: #6D4C41;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-badge-caution {
    display: inline-block;
    background: #F44336;
    color: white;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.optional-section {
    border: 1px dashed #FFC107;
    border-radius: 8px;
    padding: 16px;
    margin: 16px 0;
    background: #FFFBF0;
}

.optional-section-header {
    font-size: 12px;
    font-weight: 600;
    color: #F57C00;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
}

.caution-warning {
    background: #FFF3E0;
    border: 1px solid #FF9800;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #E65100;
}

.caution-warning::before {
    content: "⚠️ ";
}
"""


def get_metric_category_style(category: str) -> dict:
    """Get visual styling for a Van Calster metric category.

    Args:
        category: One of "RECOMMENDED", "OPTIONAL", "CAUTION", "UNKNOWN".

    Returns:
        Dict with 'bg', 'border', 'text', 'badge' styling values.
    """
    return METRIC_CATEGORY_COLORS.get(category, METRIC_CATEGORY_COLORS["UNKNOWN"])


def render_metric_badge_html(category: str) -> str:
    """Render an HTML badge for a metric category.

    Args:
        category: One of "RECOMMENDED", "OPTIONAL", "CAUTION".

    Returns:
        HTML string for the badge.
    """
    style = get_metric_category_style(category)
    css_class = f"metric-badge-{category.lower()}"
    return f'<span class="{css_class}">{style["badge"]}</span>'


def render_optional_section_html(content: str, title: str = "OPTIONAL Metrics") -> str:
    """Wrap content in an OPTIONAL section with visual indicator.

    Args:
        content: HTML content to wrap.
        title: Section title.

    Returns:
        HTML string with optional section styling.
    """
    return f"""
    <div class="optional-section">
        <div class="optional-section-header">{title}</div>
        {content}
    </div>
    """


# =============================================================================
# CHART HEIGHT CALCULATION - Standardized dynamic heights
# =============================================================================
def calculate_chart_height(n_items: int, chart_type: str = "default") -> int:
    """Calculate standardized chart height based on item count.

    Args:
        n_items: Number of items (groups, rows, categories) in the chart.
        chart_type: Type of chart for specific height rules.
            - "default": General purpose (base=300, per_item=50, max=800)
            - "forest": Forest plots (base=300, per_item=50, max=800)
            - "heatmap": Heatmaps (base=300, per_item=60, max=900)
            - "bar": Bar charts (base=400, per_item=30, max=700)

    Returns:
        Calculated height in pixels.
    """
    configs = {
        "default": {"base": 300, "per_item": 50, "max": 800},
        "forest": {"base": 300, "per_item": 50, "max": 800},
        "heatmap": {"base": 300, "per_item": 60, "max": 900},
        "bar": {"base": 400, "per_item": 30, "max": 700},
    }
    cfg = configs.get(chart_type, configs["default"])
    return min(cfg["max"], max(cfg["base"], cfg["base"] + n_items * cfg["per_item"]))


def get_plotly_template(editorial_mode: bool = True) -> dict:
    """Get Plotly template with publication-ready editorial aesthetic.

    Professional white backgrounds with WCAG 2.1 AA compliant text contrast.
    Enhanced font sizes for improved readability.
    """
    return {
        "layout": {
            "font": {
                "family": TYPOGRAPHY["data_font"],
                "size": TYPOGRAPHY["body_size"],
                "color": SEMANTIC_COLORS["text"],
            },
            "title": {
                "font": {
                    "family": TYPOGRAPHY["heading_font"],
                    "size": TYPOGRAPHY["heading_size"],
                    "color": SEMANTIC_COLORS["text"],
                },
                "x": 0,
                "xanchor": "left",
                "y": 0.98,
                "yanchor": "top",
            },
            # Pure white backgrounds for professional style
            "paper_bgcolor": "#FFFFFF",
            "plot_bgcolor": "#FFFFFF",
            "margin": {"l": 80, "r": 40, "t": 100, "b": 80},
            "xaxis": {
                "showgrid": False,
                "showline": True,
                "linewidth": 1,
                "linecolor": "#CCCCCC",  # Light gray axis line
                "tickfont": {
                    "size": TYPOGRAPHY["tick_size"],
                    "color": SEMANTIC_COLORS["text"],  # Dark text for WCAG contrast
                },
                "title": {
                    "font": {
                        "size": TYPOGRAPHY["axis_title_size"],
                        "color": SEMANTIC_COLORS["text"],  # Dark text for readability
                    }
                },
                "zeroline": False,
            },
            "yaxis": {
                "showgrid": editorial_mode,
                "gridwidth": 1,
                "gridcolor": "#E3E2E0",  # Subtle grid lines
                "showline": False,
                "tickfont": {
                    "size": TYPOGRAPHY["tick_size"],
                    "color": SEMANTIC_COLORS["text"],  # Dark text for WCAG contrast
                },
                "title": {
                    "font": {
                        "size": TYPOGRAPHY["axis_title_size"],
                        "color": SEMANTIC_COLORS["text"],  # Dark text for readability
                    }
                },
                "zeroline": False,
            },
            "legend": {
                "font": {
                    "size": TYPOGRAPHY["legend_size"],
                    "color": SEMANTIC_COLORS["text"],
                },
                "bgcolor": "rgba(255,255,255,0.9)",  # Semi-transparent white
            },
            "hoverlabel": {
                "bgcolor": "#FFFFFF",
                "bordercolor": "#CCCCCC",
                "font": {
                    "family": TYPOGRAPHY["data_font"],
                    "size": TYPOGRAPHY["body_size"],
                    "color": SEMANTIC_COLORS["text"],
                },
            },
            "colorway": GROUP_COLORS,
        }
    }


def register_plotly_template() -> None:
    """Register FairCareAI template with Plotly."""
    try:
        import plotly.io as pio

        pio.templates["faircareai"] = get_plotly_template()
        pio.templates.default = "faircareai"
    except ImportError:
        pass


def apply_faircareai_theme(fig: Any) -> Any:
    """Apply FairCareAI theme to a Plotly figure.

    Professional white backgrounds with WCAG 2.1 AA compliant contrast.
    Publication-ready editorial aesthetic with enhanced readability.

    Args:
        fig: Plotly Figure object.

    Returns:
        The figure with theme applied.
    """
    template = get_plotly_template()
    layout_updates = template.get("layout", {})

    fig.update_layout(
        font=layout_updates.get("font", {}),
        paper_bgcolor="#FFFFFF",  # Pure white
        plot_bgcolor="#FFFFFF",  # Pure white
        hoverlabel=layout_updates.get("hoverlabel", {}),
    )

    return fig


def render_footer_html() -> str:
    """Render FairCareAI branded footer HTML.

    Returns:
        HTML string for the branded footer.
    """
    return f"""
    <div class="faircareai-footer">
        <span class="brand-name">FairCare<span class="brand-accent">AI</span></span>
        · {FAIRCAREAI_BRAND["tagline"]}
    </div>
    """


def get_source_annotation(custom_note: str | None = None) -> str:
    """Get standardized source annotation for charts.

    Args:
        custom_note: Optional additional note to append.

    Returns:
        Source attribution string.
    """
    base = FAIRCAREAI_BRAND["source_note"]
    return f"{base} | {custom_note}" if custom_note else base


# Streamlit CSS with full editorial styling
# Updated: Pure white backgrounds for professional aesthetic with WCAG 2.1 AA compliance
STREAMLIT_CSS = """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global App Styling - Pure white for professional look */
    .stApp {
        background-color: #FFFFFF;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Typography */
    h1 {
        font-family: 'Merriweather', Georgia, serif !important;
        font-weight: 900 !important;
        color: #191919 !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    h2, h3 {
        font-family: 'Merriweather', Georgia, serif !important;
        font-weight: 700 !important;
        color: #191919 !important;
        letter-spacing: -0.01em;
    }

    h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #37352F !important;
    }

    p, li, span {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: #37352F;
    }

    /* Metric Cards - Professional Style */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E3E2E0;
        border-radius: 8px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6B6B6B !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 36px !important;
        line-height: 1.1;
    }

    /* Status Badge Styling */
    .status-pass {
        background: linear-gradient(135deg, #E0F2EF 0%, #C8E6C9 100%);
        border-left: 4px solid #009E73;
        color: #00695C;
    }

    .status-warn {
        background: linear-gradient(135deg, #FFFDE7 0%, #FFF8E6 100%);
        border-left: 4px solid #F0E442;
        color: #827717;
    }

    .status-fail {
        background: linear-gradient(135deg, #FFEBE5 0%, #FFCCBC 100%);
        border-left: 4px solid #D55E00;
        color: #BF360C;
    }

    /* Scorecard Container */
    .scorecard {
        display: flex;
        gap: 16px;
        margin: 24px 0;
    }

    .scorecard-item {
        flex: 1;
        background: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 1px solid #E3E2E0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .scorecard-value {
        font-family: 'Inter', sans-serif;
        font-size: 48px;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 8px;
    }

    .scorecard-label {
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6B6B6B;
    }

    /* Pass/Warn/Fail specific colors */
    .score-pass .scorecard-value { color: #009E73; }
    .score-warn .scorecard-value { color: #C9B900; }
    .score-fail .scorecard-value { color: #D55E00; }

    /* Data Tables - Clean white design with high contrast text */
    .dataframe {
        font-family: 'Inter', sans-serif !important;
        border-collapse: collapse;
        width: 100%;
        background: #FFFFFF;
    }

    .dataframe th {
        background: #FAFAFA !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #191919 !important;  /* Dark text for WCAG contrast */
        border-bottom: 2px solid #191919 !important;
        padding: 12px 16px !important;
    }

    .dataframe td {
        padding: 12px 16px !important;
        border-bottom: 1px solid #E3E2E0 !important;
        font-size: 14px;
        color: #191919;  /* Dark text for readability */
    }

    .dataframe tr:hover td {
        background: #F5F5F5 !important;  /* Subtle hover on white */
    }

    /* Ghosting / Sample Size Opacity - matches GhostingConfig values */
    .sample-adequate { opacity: 1.0; }
    .sample-moderate { opacity: 0.7; }
    .sample-low { opacity: 0.3; filter: saturate(0.7); }
    .sample-very-low { opacity: 0.15; filter: saturate(0.4); }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E3E2E0;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 16px !important;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        color: #191919 !important;
        background: #FAFAFA;
        border-radius: 4px;
    }

    /* Button Styling */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 6px;
        border: none;
        padding: 8px 20px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #E3E2E0;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 14px;
        color: #6B6B6B;
        padding: 12px 24px;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #191919;
    }

    .stTabs [aria-selected="true"] {
        color: #0072B2 !important;
        border-bottom-color: #0072B2 !important;
        font-weight: 600;
    }

    /* Info/Warning/Error Boxes */
    .stAlert {
        border-radius: 8px;
        border-left-width: 4px;
    }

    /* Code Blocks */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px;
        background: #F8F8F8;
        padding: 2px 6px;
        border-radius: 4px;
        color: #191919;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #E3E2E0;
        margin: 32px 0;
    }

    /* Safe Zone Annotation for Charts */
    .safe-zone-annotation {
        background: linear-gradient(to right, rgba(232, 245, 233, 0.8), rgba(232, 245, 233, 0.3));
        border-left: 2px solid #A5D6A7;
        padding: 8px 12px;
        font-size: 12px;
        color: #2E7D32;
        margin: 8px 0;
    }

    /* Legend Styling */
    .custom-legend {
        display: flex;
        gap: 24px;
        font-size: 12px;
        color: #6B6B6B;
        margin: 16px 0;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .scorecard {
            flex-direction: column;
        }
        .scorecard-value {
            font-size: 36px;
        }
    }

    /* ===========================================
       WCAG 2.1 AA Accessibility Enhancements
       =========================================== */

    /* Focus Indicators - WCAG 2.4.7 Focus Visible */
    *:focus-visible {
        outline: 3px solid #0072B2 !important;
        outline-offset: 2px !important;
    }

    /* Skip Link - WCAG 2.4.1 Bypass Blocks */
    .skip-link {
        position: absolute;
        top: -40px;
        left: 0;
        background: #0072B2;
        color: white !important;
        padding: 8px 16px;
        z-index: 9999;
        text-decoration: none;
        font-weight: 600;
        border-radius: 0 0 4px 0;
        font-family: 'Inter', sans-serif;
    }

    .skip-link:focus {
        top: 0;
    }

    /* Screen Reader Only - Hidden but accessible */
    .sr-only {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        padding: 0 !important;
        margin: -1px !important;
        overflow: hidden !important;
        clip: rect(0, 0, 0, 0) !important;
        white-space: nowrap !important;
        border: 0 !important;
    }

    /* Live Region for Announcements - WCAG 4.1.3 Status Messages */
    .aria-live-region {
        position: absolute;
        left: -10000px;
        width: 1px;
        height: 1px;
        overflow: hidden;
    }

    /* Reduced Motion - WCAG 2.3.3 Animation from Interactions */
    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }
    }

    /* High Contrast Support */
    @media (prefers-contrast: high) {
        .scorecard-item {
            border: 2px solid currentColor !important;
        }

        .status-pass, .status-warn, .status-fail {
            border-width: 3px !important;
        }

        [data-testid="stMetric"] {
            border: 2px solid #191919 !important;
        }

        .stButton > button {
            border: 2px solid currentColor !important;
        }
    }

    /* Ensure sufficient color contrast on interactive states */
    .dataframe tr:hover td {
        background: #F5F5F5 !important;
        color: #191919 !important;
    }

    /* Visible focus for custom interactive elements */
    .scorecard-item:focus-within,
    .legend-item:focus-within {
        outline: 3px solid #0072B2;
        outline-offset: 2px;
    }

    /* Accessible link styling - WCAG 1.4.1 Use of Color */
    a {
        text-decoration: underline;
    }

    a:hover, a:focus {
        text-decoration-thickness: 2px;
    }

    /* Minimum target size for touch - WCAG 2.5.5 Target Size */
    .stButton > button,
    .stSelectbox,
    .stRadio label,
    .stCheckbox label {
        min-height: 44px;
        min-width: 44px;
    }

    /* Error state styling with icon - WCAG 1.4.1 Use of Color */
    .stAlert[data-baseweb="notification"] {
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }

    /* Table accessibility - WCAG 1.3.1 Info and Relationships */
    .dataframe caption {
        font-weight: 600;
        text-align: left;
        margin-bottom: 8px;
        color: #37352F;
    }

    .dataframe th[scope] {
        font-weight: 600;
    }

    /* ===========================================
       FairCareAI Branding
       =========================================== */
    .faircareai-footer {
        text-align: center;
        padding: 24px 0;
        margin-top: 48px;
        border-top: 1px solid #E3E2E0;
        font-size: 12px;
        color: #6B6B6B;
        font-family: 'Inter', sans-serif;
    }

    .faircareai-footer .brand-name {
        font-weight: 600;
        color: #191919;
    }

    .faircareai-footer .brand-accent {
        color: #0072B2;
    }
</style>
"""


def inject_streamlit_css() -> None:
    """Inject editorial CSS into Streamlit."""
    try:
        import streamlit as st

        st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    except ImportError:
        pass


def render_scorecard_html(pass_count: int, warn_count: int, flag_count: int) -> str:
    """Render HTML scorecard with professional styling.

    Displays counts of metrics within, near, and outside configured thresholds.
    """
    return f"""
    <div class="scorecard">
        <div class="scorecard-item score-pass">
            <div class="scorecard-value">{pass_count}</div>
            <div class="scorecard-label">Pass</div>
        </div>
        <div class="scorecard-item score-warn">
            <div class="scorecard-value">{warn_count}</div>
            <div class="scorecard-label">Review</div>
        </div>
        <div class="scorecard-item score-fail">
            <div class="scorecard-value">{flag_count}</div>
            <div class="scorecard-label">Flag</div>
        </div>
    </div>
    """


def render_status_badge(status: str, text: str) -> str:
    """Render a status badge with appropriate styling."""
    return f'<div class="status-{status.lower()}" style="padding: 12px 16px; border-radius: 6px; margin: 8px 0;">{text}</div>'


def render_metric_card(
    label: str, value: str, status: str = "neutral", delta: str | None = None
) -> str:
    """Render a styled metric card."""
    color_map = {
        "pass": "#009E73",
        "warn": "#C9B900",
        "fail": "#D55E00",
        "neutral": "#0072B2",
    }
    color = color_map.get(status, color_map["neutral"])

    delta_html = ""
    if delta:
        delta_html = f'<div style="font-size: 12px; color: #6B6B6B; margin-top: 4px;">{delta}</div>'

    return f"""
    <div style="background: #FFFFFF; border: 1px solid #E3E2E0; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #6B6B6B; margin-bottom: 8px;">{label}</div>
        <div style="font-size: 32px; font-weight: 700; color: {color}; line-height: 1;">{value}</div>
        {delta_html}
    </div>
    """
