"""
FairCareAI Altair Static Visualization Components

Static charts for PDF/PPTX export with NYT/D3 aesthetic.
"""

import altair as alt
import polars as pl

from .themes import GHOSTING_CONFIG, SEMANTIC_COLORS, TYPOGRAPHY, GhostingConfig


def register_altair_theme() -> None:
    """Register FairCareAI JAMA-style theme with Altair.

    JAMA Scientific Publication Standards:
    - Large, clear axis labels (22pt titles, 18pt ticks)
    - Readable legend text (18pt)
    - Prominent titles (36pt)
    """

    def theme():
        return {
            "config": {
                "background": SEMANTIC_COLORS["background"],
                # Title configuration - JAMA style large, clear
                "title": {
                    "font": TYPOGRAPHY["heading_font"],
                    "fontSize": TYPOGRAPHY["headline_size"],
                    "fontWeight": TYPOGRAPHY["heading_weight"],
                    "color": SEMANTIC_COLORS["text"],
                },
                # Axis configuration - CRITICAL for scientific figures
                "axis": {
                    "grid": False,
                    "labelFont": TYPOGRAPHY["data_font"],
                    "labelFontSize": TYPOGRAPHY["tick_size"],  # 18pt tick labels
                    "labelColor": SEMANTIC_COLORS["text"],
                    "titleFont": TYPOGRAPHY["data_font"],
                    "titleFontSize": TYPOGRAPHY["axis_title_size"],  # 22pt axis titles
                    "titleFontWeight": TYPOGRAPHY["label_weight"],
                    "titleColor": SEMANTIC_COLORS["text"],
                },
                # Legend configuration - readable
                "legend": {
                    "labelFont": TYPOGRAPHY["data_font"],
                    "labelFontSize": TYPOGRAPHY["legend_size"],  # 18pt
                    "titleFont": TYPOGRAPHY["data_font"],
                    "titleFontSize": TYPOGRAPHY["legend_size"],
                    "titleFontWeight": TYPOGRAPHY["subheading_weight"],
                },
                # Header (facet titles) configuration
                "header": {
                    "labelFont": TYPOGRAPHY["data_font"],
                    "labelFontSize": TYPOGRAPHY["label_size"],
                    "titleFont": TYPOGRAPHY["heading_font"],
                    "titleFontSize": TYPOGRAPHY["subheading_size"],
                },
                # Mark configuration
                "point": {"size": 100},
                "circle": {"size": 120},
                "text": {
                    "font": TYPOGRAPHY["data_font"],
                    "fontSize": TYPOGRAPHY["annotation_size"],
                },
            }
        }

    alt.themes.register("faircareai", theme)
    alt.themes.enable("faircareai")


register_altair_theme()


def create_forest_plot_static(
    metrics_df: pl.DataFrame,
    metric: str = "tpr",
    title: str | None = None,
    enable_ghosting: bool = True,
    ghosting_config: GhostingConfig | None = None,
) -> alt.LayerChart:
    """Create static forest plot for PDF/PPTX export."""
    ghost_cfg = ghosting_config or GHOSTING_CONFIG

    df = metrics_df.filter(pl.col("group") != "_overall")
    df = df.with_columns(
        [
            pl.col("n")
            .map_elements(ghost_cfg.get_opacity, return_dtype=pl.Float64)
            .alias("opacity"),
        ]
    )

    pdf = df.to_pandas()
    title = title or f"{metric.upper()} by Group"

    points = (
        alt.Chart(pdf)
        .mark_circle(size=120)
        .encode(
            y=alt.Y("group:N", title=None),
            x=alt.X(f"{metric}:Q", title=metric.upper(), axis=alt.Axis(format=".0%")),
            opacity=alt.Opacity("opacity:Q", legend=None) if enable_ghosting else alt.value(1.0),
        )
    )

    return points.properties(title=title, width=500, height=max(200, len(pdf) * 40))


def create_icon_array(
    affected: int,
    total: int = 100,
    title: str = "Impact",
) -> str:
    """Create SVG icon array for impact visualization."""
    icons = []
    for i in range(total):
        color = SEMANTIC_COLORS["fail"] if i < affected else "#E0E0E0"
        row, col = divmod(i, 10)
        icons.append(f'<circle cx="{col * 25 + 12}" cy="{row * 25 + 40}" r="8" fill="{color}"/>')

    return f'''<svg width="260" height="300">
        <text x="5" y="20" font-family="{TYPOGRAPHY["heading_font"]}" font-weight="bold">{title}</text>
        <text x="5" y="35" font-size="12" font-family="{TYPOGRAPHY["data_font"]}">{affected} of {total} affected</text>
        {"".join(icons)}
    </svg>'''
