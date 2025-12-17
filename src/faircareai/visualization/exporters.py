"""
FairCareAI Figure Export Utilities.

Provides unified export functions for Plotly and Altair figures
with consistent error handling, format support, and publication-ready defaults.

Supported formats:
- PNG: High-resolution raster (default 2x scale for retina)
- PDF: Vector format for publications
- SVG: Scalable vector format
- HTML: Interactive web format
- JSON: Plotly JSON specification

Example:
    >>> from faircareai.visualization import create_forest_plot
    >>> from faircareai.visualization.exporters import export_plotly_figure
    >>> fig = create_forest_plot(metrics_df)
    >>> export_plotly_figure(fig, "forest_plot.png", width=1200, height=800)
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from faircareai.core.exceptions import FairCareAIError
from faircareai.core.logging import get_logger

if TYPE_CHECKING:
    import altair as alt
    import plotly.graph_objects as go

logger = get_logger(__name__)

# Type alias for supported export formats
ExportFormat = Literal["png", "pdf", "svg", "html", "json"]

# Formats requiring kaleido engine
RASTER_FORMATS = {"png", "jpeg", "webp"}
VECTOR_FORMATS = {"pdf", "svg"}
KALEIDO_FORMATS = RASTER_FORMATS | VECTOR_FORMATS


class FigureExportError(FairCareAIError):
    """Raised when figure export fails.

    Attributes:
        format: The export format that was attempted.
        reason: Human-readable explanation of the failure.
        path: The target path for the export (if applicable).
    """

    def __init__(
        self,
        format: str,
        reason: str,
        path: str | Path | None = None,
    ) -> None:
        self.format = format
        self.reason = reason
        self.path = path
        message = f"Failed to export figure as {format}: {reason}"
        if path:
            message += f" (target: {path})"
        super().__init__(message)


def export_plotly_figure(
    fig: "go.Figure",
    path: str | Path,
    format: ExportFormat | None = None,
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0,
) -> Path:
    """Export Plotly figure to a static file.

    Supports PNG, PDF, SVG, HTML, and JSON formats. For raster and vector
    formats (PNG, PDF, SVG), requires kaleido to be installed.

    Args:
        fig: Plotly Figure object to export.
        path: Output file path. Format is inferred from extension if not specified.
        format: Export format. If None, inferred from path extension.
        width: Figure width in pixels (for static formats).
        height: Figure height in pixels (for static formats).
        scale: Resolution scale factor. 2.0 = retina/HiDPI quality.

    Returns:
        Path to the exported file.

    Raises:
        FigureExportError: If export fails due to missing dependencies,
            invalid format, or I/O errors.

    Example:
        >>> fig = go.Figure(data=go.Scatter(x=[1,2,3], y=[4,5,6]))
        >>> export_plotly_figure(fig, "chart.png")
    """

    path = Path(path)

    # Infer format from extension if not specified
    if format is None:
        format = path.suffix.lstrip(".").lower()
        if not format:
            raise FigureExportError(
                format="unknown",
                reason="No format specified and could not infer from path extension",
                path=path,
            )

    # Validate format
    valid_formats = {"png", "pdf", "svg", "html", "json", "jpeg", "webp"}
    if format not in valid_formats:
        raise FigureExportError(
            format=format,
            reason=f"Unsupported format. Valid formats: {', '.join(sorted(valid_formats))}",
            path=path,
        )

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == "html":
            fig.write_html(
                str(path),
                include_plotlyjs=True,
                full_html=True,
            )
            logger.info("Exported interactive HTML: %s", path)

        elif format == "json":
            fig.write_json(str(path))
            logger.info("Exported Plotly JSON: %s", path)

        elif format in KALEIDO_FORMATS:
            # Check kaleido availability
            try:
                import kaleido  # noqa: F401
            except ImportError as e:
                raise FigureExportError(
                    format=format,
                    reason=(
                        "kaleido package required for static image export. "
                        "Install with: pip install kaleido"
                    ),
                    path=path,
                ) from e

            fig.write_image(
                str(path),
                format=format,
                width=width,
                height=height,
                scale=scale,
            )
            logger.info(
                "Exported %s image (%dx%d @ %.1fx scale): %s",
                format.upper(),
                width,
                height,
                scale,
                path,
            )

        return path

    except Exception as e:
        if isinstance(e, FigureExportError):
            raise
        raise FigureExportError(
            format=format,
            reason=str(e),
            path=path,
        ) from e


def export_altair_chart(
    chart: "alt.Chart | alt.LayerChart | alt.HConcatChart | alt.VConcatChart",
    path: str | Path,
    format: ExportFormat | None = None,
    scale_factor: float = 2.0,
) -> Path:
    """Export Altair chart to a static file.

    Uses vl-convert-python for high-quality export without requiring
    a browser or Node.js.

    Args:
        chart: Altair Chart object to export.
        path: Output file path. Format is inferred from extension if not specified.
        format: Export format. If None, inferred from path extension.
        scale_factor: Resolution scale factor for raster formats.

    Returns:
        Path to the exported file.

    Raises:
        FigureExportError: If export fails due to missing dependencies,
            invalid format, or I/O errors.

    Example:
        >>> import altair as alt
        >>> chart = alt.Chart(...).mark_bar()
        >>> export_altair_chart(chart, "chart.svg")
    """

    path = Path(path)

    # Infer format from extension if not specified
    if format is None:
        format = path.suffix.lstrip(".").lower()
        if not format:
            raise FigureExportError(
                format="unknown",
                reason="No format specified and could not infer from path extension",
                path=path,
            )

    # Validate format
    valid_formats = {"png", "pdf", "svg", "html", "json"}
    if format not in valid_formats:
        raise FigureExportError(
            format=format,
            reason=f"Unsupported format for Altair. Valid formats: {', '.join(sorted(valid_formats))}",
            path=path,
        )

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == "html":
            chart.save(str(path), format="html")
            logger.info("Exported Altair HTML: %s", path)

        elif format == "json":
            chart.save(str(path), format="json")
            logger.info("Exported Vega-Lite JSON: %s", path)

        elif format in {"png", "pdf", "svg"}:
            # Check vl-convert availability
            try:
                import vl_convert as vlc  # noqa: F401
            except ImportError as e:
                raise FigureExportError(
                    format=format,
                    reason=(
                        "vl-convert-python package required for static Altair export. "
                        "Install with: pip install vl-convert-python"
                    ),
                    path=path,
                ) from e

            if format == "png":
                chart.save(str(path), format="png", scale_factor=scale_factor)
            else:
                chart.save(str(path), format=format)

            logger.info("Exported Altair %s: %s", format.upper(), path)

        return path

    except Exception as e:
        if isinstance(e, FigureExportError):
            raise
        raise FigureExportError(
            format=format,
            reason=str(e),
            path=path,
        ) from e


def export_figure_bundle(
    fig: "go.Figure | alt.Chart",
    base_path: str | Path,
    formats: list[ExportFormat] | None = None,
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0,
) -> dict[str, Path]:
    """Export a figure to multiple formats at once.

    Useful for generating publication assets in multiple formats
    from a single figure.

    Args:
        fig: Plotly Figure or Altair Chart to export.
        base_path: Base path without extension. Each format will append its extension.
        formats: List of formats to export. Defaults to ["png", "svg", "pdf"].
        width: Figure width in pixels (Plotly only).
        height: Figure height in pixels (Plotly only).
        scale: Resolution scale factor.

    Returns:
        Dictionary mapping format names to exported file paths.

    Raises:
        FigureExportError: If any export fails.

    Example:
        >>> paths = export_figure_bundle(fig, "output/chart", formats=["png", "svg"])
        >>> print(paths)
    """
    import plotly.graph_objects as go

    if formats is None:
        formats = ["png", "svg", "pdf"]

    base_path = Path(base_path)
    results: dict[str, Path] = {}
    errors: list[str] = []

    for fmt in formats:
        export_path = base_path.with_suffix(f".{fmt}")
        try:
            if isinstance(fig, go.Figure):
                export_plotly_figure(
                    fig,
                    export_path,
                    format=fmt,
                    width=width,
                    height=height,
                    scale=scale,
                )
            else:
                # Assume Altair chart
                export_altair_chart(fig, export_path, format=fmt, scale_factor=scale)
            results[fmt] = export_path
        except FigureExportError as e:
            errors.append(f"{fmt}: {e.reason}")
            logger.warning("Failed to export %s: %s", fmt, e.reason)

    if errors and not results:
        raise FigureExportError(
            format="bundle",
            reason=f"All exports failed: {'; '.join(errors)}",
            path=base_path,
        )

    if errors:
        logger.warning(
            "Bundle export partially failed (%d/%d succeeded): %s",
            len(results),
            len(formats),
            "; ".join(errors),
        )

    return results


def get_recommended_export_settings(
    purpose: Literal["web", "print", "presentation", "journal"],
) -> dict:
    """Get recommended export settings for common use cases.

    Args:
        purpose: The intended use of the exported figure.
            - "web": Optimized for web display (smaller files)
            - "print": High-resolution for printing
            - "presentation": Balanced for slides
            - "journal": Publication-ready vector format

    Returns:
        Dictionary with recommended width, height, scale, and format.

    Example:
        >>> settings = get_recommended_export_settings("journal")
        >>> export_plotly_figure(fig, "figure1.pdf", **settings)
    """
    settings = {
        "web": {
            "width": 800,
            "height": 500,
            "scale": 1.0,
            "format": "png",
        },
        "print": {
            "width": 1600,
            "height": 1000,
            "scale": 3.0,
            "format": "png",
        },
        "presentation": {
            "width": 1200,
            "height": 800,
            "scale": 2.0,
            "format": "png",
        },
        "journal": {
            "width": 1200,
            "height": 800,
            "scale": 1.0,  # Vector, scale doesn't matter
            "format": "pdf",
        },
    }
    return settings.get(purpose, settings["presentation"])
