"""
Tests for FairCareAI visualization export utilities.

Tests cover:
- Plotly figure export (PNG, PDF, SVG, HTML, JSON)
- Altair chart export
- Bundle export to multiple formats
- Error handling for invalid formats and missing dependencies
- Recommended export settings
"""

import json
from pathlib import Path
from unittest.mock import patch

import altair as alt
import plotly.graph_objects as go
import pytest

from faircareai.visualization.exporters import (
    ExportFormat,
    FigureExportError,
    export_altair_chart,
    export_figure_bundle,
    export_plotly_figure,
    get_recommended_export_settings,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_plotly_figure() -> go.Figure:
    """Create a simple Plotly figure for testing."""
    return go.Figure(
        data=go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode="lines+markers"),
        layout=go.Layout(title="Test Figure"),
    )


@pytest.fixture
def simple_altair_chart() -> alt.Chart:
    """Create a simple Altair chart for testing."""
    import pandas as pd

    data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return alt.Chart(data).mark_point().encode(x="x:Q", y="y:Q")


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    return output_dir


# =============================================================================
# Tests: export_plotly_figure
# =============================================================================


class TestExportPlotlyFigure:
    """Tests for Plotly figure export function."""

    def test_export_html(self, simple_plotly_figure: go.Figure, temp_output_dir: Path) -> None:
        """Test exporting Plotly figure to HTML."""
        output_path = temp_output_dir / "figure.html"
        result = export_plotly_figure(simple_plotly_figure, output_path)

        assert result == output_path
        assert output_path.exists()
        content = output_path.read_text()
        assert "<html>" in content or "plotly" in content.lower()

    def test_export_json(self, simple_plotly_figure: go.Figure, temp_output_dir: Path) -> None:
        """Test exporting Plotly figure to JSON."""
        output_path = temp_output_dir / "figure.json"
        result = export_plotly_figure(simple_plotly_figure, output_path)

        assert result == output_path
        assert output_path.exists()
        # Verify valid JSON
        data = json.loads(output_path.read_text())
        assert "data" in data
        assert "layout" in data

    def test_export_png_with_kaleido(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test exporting Plotly figure to PNG (requires kaleido)."""
        output_path = temp_output_dir / "figure.png"

        try:
            result = export_plotly_figure(
                simple_plotly_figure, output_path, width=800, height=600, scale=1.0
            )
            assert result == output_path
            assert output_path.exists()
            # Check PNG magic bytes
            with open(output_path, "rb") as f:
                header = f.read(8)
                assert header[:4] == b"\x89PNG"
        except FigureExportError as e:
            if "kaleido" in str(e).lower():
                pytest.skip("kaleido not installed")
            raise

    def test_export_svg_with_kaleido(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test exporting Plotly figure to SVG (requires kaleido)."""
        output_path = temp_output_dir / "figure.svg"

        try:
            result = export_plotly_figure(simple_plotly_figure, output_path)
            assert result == output_path
            assert output_path.exists()
            content = output_path.read_text()
            assert "<svg" in content
        except FigureExportError as e:
            if "kaleido" in str(e).lower():
                pytest.skip("kaleido not installed")
            raise

    def test_export_pdf_with_kaleido(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test exporting Plotly figure to PDF (requires kaleido)."""
        output_path = temp_output_dir / "figure.pdf"

        try:
            result = export_plotly_figure(simple_plotly_figure, output_path)
            assert result == output_path
            assert output_path.exists()
            # Check PDF magic bytes
            with open(output_path, "rb") as f:
                header = f.read(4)
                assert header == b"%PDF"
        except FigureExportError as e:
            if "kaleido" in str(e).lower():
                pytest.skip("kaleido not installed")
            raise

    def test_format_inference_from_extension(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test that format is correctly inferred from file extension."""
        output_path = temp_output_dir / "figure.html"
        result = export_plotly_figure(simple_plotly_figure, output_path)
        assert result.suffix == ".html"

    def test_explicit_format_override(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test that explicit format overrides file extension."""
        # Path says .txt but we force JSON format
        output_path = temp_output_dir / "figure.txt"
        result = export_plotly_figure(simple_plotly_figure, output_path, format="json")
        assert result == output_path
        assert output_path.exists()
        # Should be valid JSON despite .txt extension
        json.loads(output_path.read_text())

    def test_invalid_format_raises_error(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test that invalid format raises FigureExportError."""
        output_path = temp_output_dir / "figure.xyz"
        with pytest.raises(FigureExportError) as exc_info:
            export_plotly_figure(simple_plotly_figure, output_path, format="xyz")

        assert exc_info.value.format == "xyz"
        assert "Unsupported format" in exc_info.value.reason

    def test_no_format_and_no_extension_raises_error(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test that missing format and extension raises error."""
        output_path = temp_output_dir / "figure"
        with pytest.raises(FigureExportError) as exc_info:
            export_plotly_figure(simple_plotly_figure, output_path)

        assert exc_info.value.format == "unknown"
        assert "could not infer" in exc_info.value.reason.lower()

    def test_creates_parent_directories(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test that export creates parent directories if needed."""
        output_path = temp_output_dir / "nested" / "deep" / "figure.html"
        assert not output_path.parent.exists()

        result = export_plotly_figure(simple_plotly_figure, output_path)
        assert result == output_path
        assert output_path.exists()

    def test_custom_dimensions(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test export with custom width and height."""
        output_path = temp_output_dir / "figure.html"
        # HTML export doesn't use dimensions directly, but the function should accept them
        result = export_plotly_figure(
            simple_plotly_figure, output_path, width=1920, height=1080, scale=3.0
        )
        assert result == output_path


# =============================================================================
# Tests: export_altair_chart
# =============================================================================


class TestExportAltairChart:
    """Tests for Altair chart export function."""

    def test_export_html(self, simple_altair_chart: alt.Chart, temp_output_dir: Path) -> None:
        """Test exporting Altair chart to HTML."""
        output_path = temp_output_dir / "chart.html"
        result = export_altair_chart(simple_altair_chart, output_path)

        assert result == output_path
        assert output_path.exists()
        content = output_path.read_text()
        assert "vega" in content.lower() or "html" in content.lower()

    def test_export_json(self, simple_altair_chart: alt.Chart, temp_output_dir: Path) -> None:
        """Test exporting Altair chart to Vega-Lite JSON."""
        output_path = temp_output_dir / "chart.json"
        result = export_altair_chart(simple_altair_chart, output_path)

        assert result == output_path
        assert output_path.exists()
        # Verify valid Vega-Lite JSON
        data = json.loads(output_path.read_text())
        assert "$schema" in data or "mark" in data

    def test_export_png_with_vl_convert(
        self, simple_altair_chart: alt.Chart, temp_output_dir: Path
    ) -> None:
        """Test exporting Altair chart to PNG (requires vl-convert)."""
        output_path = temp_output_dir / "chart.png"

        try:
            result = export_altair_chart(simple_altair_chart, output_path)
            assert result == output_path
            assert output_path.exists()
            # Check PNG magic bytes
            with open(output_path, "rb") as f:
                header = f.read(8)
                assert header[:4] == b"\x89PNG"
        except FigureExportError as e:
            if "vl-convert" in str(e).lower():
                pytest.skip("vl-convert-python not installed")
            raise

    def test_export_svg_with_vl_convert(
        self, simple_altair_chart: alt.Chart, temp_output_dir: Path
    ) -> None:
        """Test exporting Altair chart to SVG (requires vl-convert)."""
        output_path = temp_output_dir / "chart.svg"

        try:
            result = export_altair_chart(simple_altair_chart, output_path)
            assert result == output_path
            assert output_path.exists()
            content = output_path.read_text()
            assert "<svg" in content
        except FigureExportError as e:
            if "vl-convert" in str(e).lower():
                pytest.skip("vl-convert-python not installed")
            raise

    def test_invalid_format_raises_error(
        self, simple_altair_chart: alt.Chart, temp_output_dir: Path
    ) -> None:
        """Test that invalid format raises FigureExportError."""
        output_path = temp_output_dir / "chart.xyz"
        with pytest.raises(FigureExportError) as exc_info:
            export_altair_chart(simple_altair_chart, output_path, format="xyz")

        assert exc_info.value.format == "xyz"
        assert "Unsupported format" in exc_info.value.reason


# =============================================================================
# Tests: export_figure_bundle
# =============================================================================


class TestExportFigureBundle:
    """Tests for bundle export function."""

    def test_bundle_export_html_and_json(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test bundle export to HTML and JSON (no external deps)."""
        base_path = temp_output_dir / "bundle"
        formats: list[ExportFormat] = ["html", "json"]

        result = export_figure_bundle(simple_plotly_figure, base_path, formats=formats)

        assert "html" in result
        assert "json" in result
        assert result["html"].exists()
        assert result["json"].exists()
        assert result["html"].suffix == ".html"
        assert result["json"].suffix == ".json"

    def test_bundle_export_default_formats(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test bundle export with default formats (png, svg, pdf)."""
        base_path = temp_output_dir / "bundle"

        try:
            result = export_figure_bundle(simple_plotly_figure, base_path)
            # Default is png, svg, pdf - all require kaleido
            assert len(result) > 0
        except FigureExportError as e:
            if "kaleido" in str(e).lower() or "All exports failed" in str(e):
                pytest.skip("kaleido not installed")
            raise

    def test_bundle_partial_failure(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test that bundle export continues after partial failure."""
        base_path = temp_output_dir / "bundle"
        # Mix of valid (html, json) and potentially failing (png) formats
        formats: list[ExportFormat] = ["html", "json", "png"]

        # This should at least succeed for html and json
        result = export_figure_bundle(simple_plotly_figure, base_path, formats=formats)

        assert "html" in result
        assert "json" in result

    def test_bundle_all_fail_raises_error(
        self, simple_plotly_figure: go.Figure, temp_output_dir: Path
    ) -> None:
        """Test that bundle export raises error if all exports fail."""
        base_path = temp_output_dir / "bundle"

        # Mock to force all exports to fail
        with patch("faircareai.visualization.exporters.export_plotly_figure") as mock_export:
            mock_export.side_effect = FigureExportError(format="png", reason="Mock failure")

            with pytest.raises(FigureExportError) as exc_info:
                export_figure_bundle(simple_plotly_figure, base_path, formats=["png", "svg"])

            assert "All exports failed" in exc_info.value.reason

    def test_bundle_with_altair_chart(
        self, simple_altair_chart: alt.Chart, temp_output_dir: Path
    ) -> None:
        """Test bundle export with Altair chart."""
        base_path = temp_output_dir / "altair_bundle"
        formats: list[ExportFormat] = ["html", "json"]

        result = export_figure_bundle(simple_altair_chart, base_path, formats=formats)

        assert "html" in result
        assert "json" in result


# =============================================================================
# Tests: get_recommended_export_settings
# =============================================================================


class TestGetRecommendedExportSettings:
    """Tests for recommended export settings function."""

    def test_web_settings(self) -> None:
        """Test recommended settings for web purpose."""
        settings = get_recommended_export_settings("web")

        assert settings["width"] == 800
        assert settings["height"] == 500
        assert settings["scale"] == 1.0
        assert settings["format"] == "png"

    def test_print_settings(self) -> None:
        """Test recommended settings for print purpose."""
        settings = get_recommended_export_settings("print")

        assert settings["width"] == 1600
        assert settings["height"] == 1000
        assert settings["scale"] == 3.0
        assert settings["format"] == "png"

    def test_presentation_settings(self) -> None:
        """Test recommended settings for presentation purpose."""
        settings = get_recommended_export_settings("presentation")

        assert settings["width"] == 1200
        assert settings["height"] == 800
        assert settings["scale"] == 2.0
        assert settings["format"] == "png"

    def test_journal_settings(self) -> None:
        """Test recommended settings for journal purpose."""
        settings = get_recommended_export_settings("journal")

        assert settings["width"] == 1200
        assert settings["height"] == 800
        assert settings["format"] == "pdf"

    def test_unknown_purpose_returns_presentation(self) -> None:
        """Test that unknown purpose falls back to presentation settings."""
        settings = get_recommended_export_settings("unknown")  # type: ignore
        presentation = get_recommended_export_settings("presentation")

        assert settings == presentation


# =============================================================================
# Tests: FigureExportError
# =============================================================================


class TestFigureExportError:
    """Tests for the FigureExportError exception class."""

    def test_error_attributes(self) -> None:
        """Test that error has correct attributes."""
        error = FigureExportError(
            format="png", reason="kaleido not installed", path="/tmp/test.png"
        )

        assert error.format == "png"
        assert error.reason == "kaleido not installed"
        assert error.path == "/tmp/test.png"

    def test_error_message_format(self) -> None:
        """Test that error message is correctly formatted."""
        error = FigureExportError(format="svg", reason="Test failure", path="/tmp/test")

        assert "svg" in str(error)
        assert "Test failure" in str(error)
        assert "/tmp/test" in str(error)

    def test_error_without_path(self) -> None:
        """Test error message without path."""
        error = FigureExportError(format="pdf", reason="Generic error")

        assert "pdf" in str(error)
        assert "Generic error" in str(error)

    def test_error_inherits_from_faircareai_error(self) -> None:
        """Test that FigureExportError inherits from FairCareAIError."""
        from faircareai.core.exceptions import FairCareAIError

        error = FigureExportError(format="png", reason="test")
        assert isinstance(error, FairCareAIError)
