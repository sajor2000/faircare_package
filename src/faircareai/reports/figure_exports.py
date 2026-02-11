"""Figure export utilities for FairCareAI reports."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any

import plotly.io as pio

from faircareai.core.config import OutputPersona


def _slugify(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return clean or "figure"


def render_png_bytes(
    fig: Any, scale: int = 2, width: int | None = None, height: int | None = None
) -> bytes:
    """Render a Plotly figure to PNG bytes via Kaleido."""
    try:
        return pio.to_image(fig, format="png", scale=scale, width=width, height=height)
    except ValueError as err:
        raise ImportError(
            "PNG export requires the kaleido engine. Install with: "
            'pip install "faircareai[export]"'
        ) from err


def collect_governance_figures(results: Any) -> dict[str, Any]:
    """Collect governance persona figures for export."""
    from faircareai.visualization.governance_dashboard import (
        create_governance_overall_figures,
        create_governance_subgroup_figures,
    )

    figures: dict[str, Any] = {}
    figures["Executive Summary"] = results.plot_executive_summary()
    figures["Go/No-Go Scorecard"] = results.plot_go_nogo_scorecard()

    overall = create_governance_overall_figures(results)
    for title, fig in overall.items():
        if title == "_explanations":
            continue
        figures[f"Overall - {title}"] = fig

    subgroup_figs = create_governance_subgroup_figures(results)
    for attr, fig_map in subgroup_figs.items():
        for title, fig in fig_map.items():
            figures[f"{attr} - {title}"] = fig

    return figures


def collect_data_scientist_figures(results: Any, include_optional: bool = False) -> dict[str, Any]:
    """Collect data scientist persona figures for export."""
    from faircareai.visualization.performance_charts import (
        plot_calibration_curve,
        plot_decision_curve,
        plot_discrimination_curves,
        plot_threshold_analysis,
    )
    from faircareai.visualization.governance_dashboard import create_fairness_dashboard

    figures: dict[str, Any] = {}
    figures["Discrimination Curves"] = plot_discrimination_curves(
        results, include_optional=include_optional, persona=OutputPersona.DATA_SCIENTIST
    )
    figures["Calibration Curve"] = plot_calibration_curve(
        results, include_optional=include_optional, persona=OutputPersona.DATA_SCIENTIST
    )
    figures["Decision Curve"] = plot_decision_curve(results)
    figures["Threshold Analysis"] = plot_threshold_analysis(
        results,
        selected_threshold=results.overall_performance.get("primary_threshold", 0.5),
    )
    figures["Fairness Dashboard"] = create_fairness_dashboard(results)

    # Optional: Van Calster dashboard when raw audit data is available
    if getattr(results, "_audit", None) is not None:
        try:
            from faircareai.metrics.vancalster import compute_vancalster_metrics
            from faircareai.visualization.vancalster_plots import create_vancalster_dashboard

            audit = results._audit
            vancalster = compute_vancalster_metrics(
                df=audit.df,
                y_prob_col=audit.pred_col,
                y_true_col=audit.target_col,
                group_col=audit.sensitive_attributes[0].column
                if audit.sensitive_attributes
                else None,
            )
            figures["Van Calster Dashboard"] = create_vancalster_dashboard(vancalster)
        except Exception:
            pass

    return figures


def collect_figures(
    results: Any,
    persona: OutputPersona,
    include_optional: bool = False,
) -> dict[str, Any]:
    """Collect figures for the selected persona."""
    if persona == OutputPersona.GOVERNANCE:
        return collect_governance_figures(results)
    return collect_data_scientist_figures(results, include_optional=include_optional)


def export_png_bundle(
    results: Any,
    output_path: str | Path,
    persona: OutputPersona = OutputPersona.GOVERNANCE,
    include_optional: bool = False,
    scale: int = 2,
) -> Path:
    """Export figures to a directory or zip bundle of PNGs."""
    output_path = Path(output_path)
    figures = collect_figures(results, persona=persona, include_optional=include_optional)

    if output_path.suffix.lower() == ".zip":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, fig in figures.items():
                filename = f"{_slugify(name)}.png"
                png_bytes = render_png_bytes(fig, scale=scale)
                zf.writestr(filename, png_bytes)
        return output_path

    # Treat as directory
    output_path.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        filename = output_path / f"{_slugify(name)}.png"
        png_bytes = render_png_bytes(fig, scale=scale)
        filename.write_bytes(png_bytes)

    return output_path
