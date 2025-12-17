"""
Tests for FairCareAI visualization themes module.

Tests cover:
- Color constants and palettes
- GhostingConfig dataclass
- calculate_chart_height function
- get_plotly_template function
- register_plotly_template function
- apply_faircareai_theme function
- render_footer_html function
- get_source_annotation function
- HTML rendering functions
"""

import pytest

from faircareai.visualization.themes import (
    COLORSCALES,
    EDITORIAL_COLORS,
    FAIRCAREAI_BRAND,
    FAIRCAREAI_COLORS,
    GHOSTING_CONFIG,
    GROUP_COLORS,
    LEGEND_POSITIONS,
    METHODOLOGY_CITATION,
    OKABE_ITO,
    SEMANTIC_COLORS,
    STREAMLIT_CSS,
    SUBPLOT_SPACING,
    TYPOGRAPHY,
    GhostingConfig,
    apply_faircareai_theme,
    calculate_chart_height,
    get_plotly_template,
    get_source_annotation,
    inject_streamlit_css,
    register_plotly_template,
    render_footer_html,
    render_metric_card,
    render_scorecard_html,
    render_status_badge,
)


class TestColorConstants:
    """Tests for color constants."""

    def test_faircareai_colors_has_primary(self) -> None:
        """Test that FAIRCAREAI_COLORS has primary."""
        assert "primary" in FAIRCAREAI_COLORS
        assert FAIRCAREAI_COLORS["primary"].startswith("#")

    def test_faircareai_colors_has_background(self) -> None:
        """Test that FAIRCAREAI_COLORS has background."""
        assert "background" in FAIRCAREAI_COLORS

    def test_okabe_ito_has_all_colors(self) -> None:
        """Test that OKABE_ITO palette has all expected colors."""
        expected = ["orange", "sky_blue", "bluish_green", "yellow", "blue", "vermillion"]
        for color in expected:
            assert color in OKABE_ITO

    def test_semantic_colors_has_pass_warn_fail(self) -> None:
        """Test that SEMANTIC_COLORS has status colors."""
        assert "pass" in SEMANTIC_COLORS
        assert "warn" in SEMANTIC_COLORS
        assert "fail" in SEMANTIC_COLORS

    def test_group_colors_is_list(self) -> None:
        """Test that GROUP_COLORS is a list."""
        assert isinstance(GROUP_COLORS, list)
        assert len(GROUP_COLORS) > 0

    def test_editorial_colors_has_newsprint(self) -> None:
        """Test that EDITORIAL_COLORS has newsprint."""
        assert "newsprint" in EDITORIAL_COLORS


class TestColorscales:
    """Tests for COLORSCALES constant."""

    def test_has_diverging_disparity(self) -> None:
        """Test that diverging disparity colorscale exists."""
        assert "diverging_disparity" in COLORSCALES

    def test_has_sequential_status(self) -> None:
        """Test that sequential status colorscale exists."""
        assert "sequential_status" in COLORSCALES

    def test_has_gauge_steps(self) -> None:
        """Test that gauge steps exist."""
        assert "gauge_steps" in COLORSCALES


class TestLegendPositions:
    """Tests for LEGEND_POSITIONS constant."""

    def test_has_top_horizontal(self) -> None:
        """Test that top horizontal position exists."""
        assert "top_horizontal" in LEGEND_POSITIONS

    def test_has_bottom_horizontal(self) -> None:
        """Test that bottom horizontal position exists."""
        assert "bottom_horizontal" in LEGEND_POSITIONS

    def test_position_has_required_keys(self) -> None:
        """Test that position configs have required keys."""
        for pos_name, pos_config in LEGEND_POSITIONS.items():
            assert "y" in pos_config or "yanchor" in pos_config


class TestSubplotSpacing:
    """Tests for SUBPLOT_SPACING constant."""

    def test_has_default(self) -> None:
        """Test that default spacing exists."""
        assert "default" in SUBPLOT_SPACING

    def test_has_tight(self) -> None:
        """Test that tight spacing exists."""
        assert "tight" in SUBPLOT_SPACING

    def test_has_wide(self) -> None:
        """Test that wide spacing exists."""
        assert "wide" in SUBPLOT_SPACING


class TestTypography:
    """Tests for TYPOGRAPHY constant."""

    def test_has_heading_font(self) -> None:
        """Test that heading font is defined."""
        assert "heading_font" in TYPOGRAPHY

    def test_has_data_font(self) -> None:
        """Test that data font is defined."""
        assert "data_font" in TYPOGRAPHY

    def test_has_heading_size(self) -> None:
        """Test that heading size is defined."""
        assert "heading_size" in TYPOGRAPHY
        assert isinstance(TYPOGRAPHY["heading_size"], int)

    def test_has_axis_sizes(self) -> None:
        """Test that axis sizes are defined."""
        assert "axis_title_size" in TYPOGRAPHY
        assert "tick_size" in TYPOGRAPHY


class TestGhostingConfig:
    """Tests for GhostingConfig dataclass."""

    def test_default_instance(self) -> None:
        """Test default GhostingConfig instance."""
        config = GhostingConfig()
        assert config.adequate_threshold == 50
        assert config.moderate_threshold == 30
        assert config.low_threshold == 10

    def test_get_opacity_adequate(self) -> None:
        """Test opacity for adequate sample size."""
        config = GhostingConfig()
        assert config.get_opacity(100) == 1.0

    def test_get_opacity_moderate(self) -> None:
        """Test opacity for moderate sample size."""
        config = GhostingConfig()
        assert config.get_opacity(40) == 0.7

    def test_get_opacity_low(self) -> None:
        """Test opacity for low sample size."""
        config = GhostingConfig()
        assert config.get_opacity(15) == 0.3

    def test_get_opacity_very_low(self) -> None:
        """Test opacity for very low sample size."""
        config = GhostingConfig()
        assert config.get_opacity(5) == 0.15

    def test_get_badge_adequate(self) -> None:
        """Test badge for adequate sample size."""
        config = GhostingConfig()
        assert config.get_badge(100) == ""

    def test_get_badge_moderate(self) -> None:
        """Test badge for moderate sample size."""
        config = GhostingConfig()
        assert "Limited" in config.get_badge(40)

    def test_get_badge_low(self) -> None:
        """Test badge for low sample size."""
        config = GhostingConfig()
        assert "caution" in config.get_badge(15)

    def test_get_badge_very_low(self) -> None:
        """Test badge for very low sample size."""
        config = GhostingConfig()
        assert "Insufficient" in config.get_badge(5)

    def test_get_css_class_adequate(self) -> None:
        """Test CSS class for adequate sample size."""
        config = GhostingConfig()
        assert config.get_css_class(100) == "sample-adequate"

    def test_get_css_class_moderate(self) -> None:
        """Test CSS class for moderate sample size."""
        config = GhostingConfig()
        assert config.get_css_class(40) == "sample-moderate"

    def test_get_css_class_low(self) -> None:
        """Test CSS class for low sample size."""
        config = GhostingConfig()
        assert config.get_css_class(15) == "sample-low"

    def test_get_css_class_very_low(self) -> None:
        """Test CSS class for very low sample size."""
        config = GhostingConfig()
        assert config.get_css_class(5) == "sample-very-low"

    def test_global_config_instance(self) -> None:
        """Test that GHOSTING_CONFIG is a valid instance."""
        assert isinstance(GHOSTING_CONFIG, GhostingConfig)


class TestCalculateChartHeight:
    """Tests for calculate_chart_height function."""

    def test_default_type_base_height(self) -> None:
        """Test default chart type base height."""
        height = calculate_chart_height(0)
        assert height >= 300

    def test_height_increases_with_items(self) -> None:
        """Test that height increases with more items."""
        height_5 = calculate_chart_height(5)
        height_10 = calculate_chart_height(10)
        assert height_10 > height_5

    def test_max_height_enforced(self) -> None:
        """Test that maximum height is enforced."""
        height = calculate_chart_height(100)
        assert height <= 800

    def test_forest_chart_type(self) -> None:
        """Test forest chart type."""
        height = calculate_chart_height(5, chart_type="forest")
        assert height >= 300

    def test_heatmap_chart_type(self) -> None:
        """Test heatmap chart type."""
        height = calculate_chart_height(5, chart_type="heatmap")
        assert height >= 300

    def test_bar_chart_type(self) -> None:
        """Test bar chart type."""
        height = calculate_chart_height(5, chart_type="bar")
        assert height >= 400

    def test_unknown_type_uses_default(self) -> None:
        """Test that unknown type uses default config."""
        height_default = calculate_chart_height(5, chart_type="default")
        height_unknown = calculate_chart_height(5, chart_type="unknown_type")
        assert height_default == height_unknown


class TestGetPlotlyTemplate:
    """Tests for get_plotly_template function."""

    def test_returns_dict(self) -> None:
        """Test that function returns a dictionary."""
        template = get_plotly_template()
        assert isinstance(template, dict)

    def test_has_layout(self) -> None:
        """Test that template has layout."""
        template = get_plotly_template()
        assert "layout" in template

    def test_layout_has_font(self) -> None:
        """Test that layout has font settings."""
        template = get_plotly_template()
        assert "font" in template["layout"]

    def test_layout_has_title(self) -> None:
        """Test that layout has title settings."""
        template = get_plotly_template()
        assert "title" in template["layout"]

    def test_layout_has_xaxis(self) -> None:
        """Test that layout has xaxis settings."""
        template = get_plotly_template()
        assert "xaxis" in template["layout"]

    def test_layout_has_yaxis(self) -> None:
        """Test that layout has yaxis settings."""
        template = get_plotly_template()
        assert "yaxis" in template["layout"]

    def test_editorial_mode_affects_grid(self) -> None:
        """Test that editorial mode affects grid visibility."""
        template_editorial = get_plotly_template(editorial_mode=True)
        template_plain = get_plotly_template(editorial_mode=False)
        assert template_editorial["layout"]["yaxis"]["showgrid"] is True
        assert template_plain["layout"]["yaxis"]["showgrid"] is False


class TestRegisterPlotlyTemplate:
    """Tests for register_plotly_template function."""

    def test_no_error_without_plotly(self) -> None:
        """Test that function doesn't error without plotly."""
        # This should not raise even if plotly is not available
        register_plotly_template()


class TestApplyFaircareaiTheme:
    """Tests for apply_faircareai_theme function."""

    def test_returns_figure(self) -> None:
        """Test that function returns the figure."""
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

        fig = go.Figure()
        result = apply_faircareai_theme(fig)
        assert result is fig

    def test_updates_paper_bgcolor(self) -> None:
        """Test that paper background color is updated."""
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

        fig = go.Figure()
        apply_faircareai_theme(fig)
        assert fig.layout.paper_bgcolor == "#FFFFFF"


class TestRenderFooterHtml:
    """Tests for render_footer_html function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = render_footer_html()
        assert isinstance(result, str)

    def test_contains_brand_name(self) -> None:
        """Test that footer contains brand name."""
        result = render_footer_html()
        assert "FairCare" in result

    def test_contains_tagline(self) -> None:
        """Test that footer contains tagline."""
        result = render_footer_html()
        assert FAIRCAREAI_BRAND["tagline"] in result


class TestGetSourceAnnotation:
    """Tests for get_source_annotation function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = get_source_annotation()
        assert isinstance(result, str)

    def test_contains_source_note(self) -> None:
        """Test that result contains source note."""
        result = get_source_annotation()
        assert FAIRCAREAI_BRAND["source_note"] in result

    def test_with_custom_note(self) -> None:
        """Test with custom note."""
        result = get_source_annotation("Custom note")
        assert "Custom note" in result
        assert FAIRCAREAI_BRAND["source_note"] in result


class TestInjectStreamlitCss:
    """Tests for inject_streamlit_css function."""

    def test_no_error_without_streamlit(self) -> None:
        """Test that function doesn't error without streamlit."""
        # This should not raise even if streamlit is not available
        inject_streamlit_css()


class TestRenderScorecardHtml:
    """Tests for render_scorecard_html function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = render_scorecard_html(5, 3, 2)
        assert isinstance(result, str)

    def test_contains_pass_count(self) -> None:
        """Test that result contains pass count."""
        result = render_scorecard_html(5, 3, 2)
        assert ">5<" in result

    def test_contains_warn_count(self) -> None:
        """Test that result contains warn count."""
        result = render_scorecard_html(5, 3, 2)
        assert ">3<" in result

    def test_contains_flag_count(self) -> None:
        """Test that result contains flag count."""
        result = render_scorecard_html(5, 3, 2)
        assert ">2<" in result

    def test_contains_scorecard_class(self) -> None:
        """Test that result contains scorecard CSS class."""
        result = render_scorecard_html(1, 2, 3)
        assert 'class="scorecard"' in result


class TestRenderStatusBadge:
    """Tests for render_status_badge function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = render_status_badge("pass", "Test passed")
        assert isinstance(result, str)

    def test_contains_status_class(self) -> None:
        """Test that result contains status class."""
        result = render_status_badge("fail", "Test failed")
        assert "status-fail" in result

    def test_contains_text(self) -> None:
        """Test that result contains the text."""
        result = render_status_badge("warn", "Warning message")
        assert "Warning message" in result


class TestRenderMetricCard:
    """Tests for render_metric_card function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = render_metric_card("Test Label", "42%")
        assert isinstance(result, str)

    def test_contains_label(self) -> None:
        """Test that result contains the label."""
        result = render_metric_card("AUROC", "0.85")
        assert "AUROC" in result

    def test_contains_value(self) -> None:
        """Test that result contains the value."""
        result = render_metric_card("Score", "95%")
        assert "95%" in result

    def test_pass_status_color(self) -> None:
        """Test pass status uses correct color."""
        result = render_metric_card("Test", "100%", status="pass")
        assert "#009E73" in result

    def test_fail_status_color(self) -> None:
        """Test fail status uses correct color."""
        result = render_metric_card("Test", "50%", status="fail")
        assert "#D55E00" in result

    def test_with_delta(self) -> None:
        """Test with delta value."""
        result = render_metric_card("Test", "100%", delta="+5%")
        assert "+5%" in result


class TestMethodologyCitation:
    """Tests for methodology citation constant."""

    def test_citation_is_string(self) -> None:
        """Test that citation is a string."""
        assert isinstance(METHODOLOGY_CITATION, str)

    def test_citation_not_empty(self) -> None:
        """Test that citation is not empty."""
        assert len(METHODOLOGY_CITATION) > 0


class TestStreamlitCss:
    """Tests for STREAMLIT_CSS constant."""

    def test_is_string(self) -> None:
        """Test that STREAMLIT_CSS is a string."""
        assert isinstance(STREAMLIT_CSS, str)

    def test_contains_style_tag(self) -> None:
        """Test that it contains style tag."""
        assert "<style>" in STREAMLIT_CSS

    def test_contains_font_import(self) -> None:
        """Test that it contains font import."""
        assert "@import" in STREAMLIT_CSS
