"""
FairCareAI Command-Line Interface

Provides command-line access to FairCareAI functionality including:
- Launching the interactive dashboard
- Running audits from configuration files
- Generating reports

Metrics computed per Van Calster et al. (2025) methodology.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from faircareai.core.exceptions import (
    ConfigurationError,
    DataValidationError,
    MetricComputationError,
)

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(message="%(prog)s %(version)s")
def main() -> None:
    """FairCareAI - Healthcare AI Fairness Auditing.

    A command-line tool for auditing ML models for fairness in clinical contexts.
    Designed for data scientists to present fairness analysis results
    to governance stakeholders.

    \b
    Metrics computed per Van Calster et al. (2025) methodology.
    Healthcare organizations interpret results based on their context.
    """
    pass


@main.command()
@click.option(
    "--port",
    "-p",
    default=8501,
    help="Port to run the dashboard on (default: 8501)",
)
@click.option(
    "--host",
    "-h",
    default="localhost",
    help="Host to bind to (default: localhost)",
)
def dashboard(port: int, host: str) -> None:
    """Launch the interactive FairCareAI dashboard.

    Opens a Streamlit web application for interactive fairness analysis,
    data exploration, and report generation.

    \b
    Example:
        faircareai dashboard
        faircareai dashboard --port 8080
    """
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        console.print(
            "[red]Error:[/red] Streamlit is required for the dashboard. "
            "Install with: pip install streamlit"
        )
        sys.exit(1)

    # Get the path to the dashboard app
    from faircareai.dashboard import app as dashboard_app

    app_path = Path(dashboard_app.__file__).resolve()

    console.print(
        Panel.fit(
            f"[bold blue]FairCareAI Dashboard[/bold blue]\nStarting on http://{host}:{port}",
            border_style="blue",
        )
    )

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]
    stcli.main()


@main.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--pred-col",
    "-p",
    required=True,
    help="Column name for model predictions/probabilities",
)
@click.option(
    "--target-col",
    "-t",
    required=True,
    help="Column name for actual outcomes (0/1)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (supports .html, .pdf, .pptx, .json, .md)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(
        [
            "html",
            "pdf",
            "pptx",
            "json",
            "png",
            "model-card",
            "chai-model-card",
            "chai-model-card-json",
            "raic-checklist",
            "repro-bundle",
        ],
        case_sensitive=False,
    ),
    help=(
        "Output format (html, pdf, pptx, json, png, model-card, chai-model-card, "
        "chai-model-card-json, raic-checklist, repro-bundle). "
        "Overrides file suffix if provided."
    ),
)
@click.option(
    "--persona",
    type=click.Choice(["data_scientist", "governance"], case_sensitive=False),
    default="data_scientist",
    show_default=True,
    help="Output persona for reports.",
)
@click.option(
    "--include-optional",
    is_flag=True,
    default=False,
    help="Include OPTIONAL metrics in data scientist outputs.",
)
@click.option(
    "--threshold",
    default=0.5,
    help="Decision threshold for binary classification (default: 0.5)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for bootstrap resampling",
)
@click.option(
    "--model-name",
    default="Unnamed Model",
    help="Name of the model being audited",
)
@click.option(
    "--attributes",
    "-a",
    multiple=True,
    help="Sensitive attributes to analyze (can specify multiple)",
)
def audit(
    data_path: Path,
    pred_col: str,
    target_col: str,
    output: Path | None,
    output_format: str | None,
    threshold: float,
    seed: int | None,
    model_name: str,
    attributes: tuple[str, ...],
    persona: str,
    include_optional: bool,
) -> None:
    """Run a fairness audit on model predictions.

    \b
    DATA_PATH: Path to predictions file (parquet or csv)

    \b
    Example:
        faircareai audit predictions.parquet -p risk_score -t outcome -o report.html
        faircareai audit data.csv -p prob -t label -a race -a sex

    \b
    Methodology: Van Calster et al. (2025), CHAI RAIC Checkpoint 1.
    """
    from faircareai import FairCareAudit, FairnessConfig
    from faircareai.core.config import FairnessMetric

    # Expand and resolve paths for cross-platform compatibility
    data_path = data_path.expanduser().resolve()
    if output:
        output = output.expanduser().resolve()

    console.print(
        Panel.fit(
            f"[bold blue]FairCareAI Audit[/bold blue]\nData: {data_path}\nModel: {model_name}",
            border_style="blue",
        )
    )

    # Initialize audit
    console.print("\n[bold]Loading data...[/bold]")
    try:
        audit_obj = FairCareAudit(
            data=data_path,
            pred_col=pred_col,
            target_col=target_col,
            threshold=threshold,
        )
        console.print(f"  Loaded {len(audit_obj.df):,} records")
    except (DataValidationError, ConfigurationError, MetricComputationError) as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("Unexpected error loading data")
        sys.exit(1)

    # Handle attributes
    if attributes:
        console.print(f"\n[bold]Using specified attributes:[/bold] {', '.join(attributes)}")
        for attr in attributes:
            try:
                audit_obj.add_sensitive_attribute(attr)
            except ValueError as e:
                console.print(f"[yellow]Warning:[/yellow] {e}")
    else:
        console.print("\n[bold]Detecting sensitive attributes...[/bold]")
        suggestions = audit_obj.suggest_attributes(display=False)
        if suggestions:
            console.print(f"  Found {len(suggestions)} suggested attributes")
            # Auto-accept first 3 suggestions for CLI
            indices: list[int | str] = list(range(1, min(4, len(suggestions) + 1)))
            audit_obj.accept_suggested_attributes(indices)
            console.print(f"  Accepted: {[s['suggested_name'] for s in suggestions[:3]]}")
        else:
            console.print("[yellow]No sensitive attributes detected. Add with -a flag.[/yellow]")
            sys.exit(1)

    # Configure audit
    console.print("\n[bold]Configuring audit...[/bold]")
    audit_obj.config = FairnessConfig(
        model_name=model_name,
        primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
        fairness_justification=(
            "Default CLI audit using equalized odds. Review and adjust based on clinical context."
        ),
    )

    # Run audit
    console.print("\n[bold]Running fairness audit...[/bold]")
    try:
        results = audit_obj.run(random_seed=seed)
    except (DataValidationError, ConfigurationError, MetricComputationError) as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("Unexpected error running audit")
        sys.exit(1)

    # Display summary
    rec = results.governance_recommendation
    status_color = {
        "READY": "green",
        "CONDITIONAL": "yellow",
        "REVIEW": "red",
    }.get(rec["status"], "white")

    console.print(
        Panel.fit(
            f"[bold]Audit Complete[/bold]\n\n"
            f"Status: [{status_color}]{rec['status']}[/{status_color}]\n"
            f"Advisory: {rec['advisory']}\n\n"
            f"Checks: {rec['n_pass']} pass, {rec['n_warnings']} warn, {rec['n_errors']} error\n\n"
            f"[dim]{rec['disclaimer']}[/dim]",
            border_style=status_color,
        )
    )

    # Export if output specified or format explicitly requested
    if output or output_format:
        fmt = output_format.lower() if output_format else None
        output_path = Path(output) if output else None

        if fmt == "model-card":
            fmt_ext = "md"
        elif fmt == "chai-model-card":
            fmt_ext = "xml"
        elif fmt in {"chai-model-card-json", "raic-checklist"}:
            fmt_ext = "json"
        elif fmt == "png":
            fmt_ext = "zip"
        elif fmt == "repro-bundle":
            fmt_ext = "json"
        else:
            fmt_ext = fmt

        # Default output path if format provided without output
        if output_path is None and fmt:
            output_path = Path(f"faircareai_report.{fmt_ext}")

        # Normalize suffix to match format when provided
        if fmt and output_path is not None:
            if fmt == "png" and output_path.exists() and output_path.is_dir():
                fmt_ext = None
            if fmt_ext is not None and output_path.suffix.lower() != f".{fmt_ext}":
                output_path = output_path.with_suffix(f".{fmt_ext}")

        # Infer format from suffix if not explicitly provided
        if fmt is None and output_path is not None:
            inferred = output_path.suffix.lstrip(".").lower()
            if inferred in {"md", "markdown"}:
                fmt = "model-card"
            elif inferred == "xml":
                fmt = "chai-model-card"
            else:
                fmt = inferred

        console.print(f"\n[bold]Exporting to {output_path}...[/bold]")

        try:
            from faircareai.core.config import OutputPersona

            persona_enum = (
                OutputPersona.GOVERNANCE
                if persona.lower() == OutputPersona.GOVERNANCE.value
                else OutputPersona.DATA_SCIENTIST
            )

            if fmt == "html":
                results.to_html(
                    str(output_path),
                    persona=persona_enum,
                    include_optional=include_optional,
                )
            elif fmt == "pdf":
                results.to_pdf(
                    str(output_path),
                    persona=persona_enum,
                    include_optional=include_optional,
                )
            elif fmt == "pptx":
                results.to_pptx(str(output_path), persona=persona_enum)
            elif fmt == "json":
                results.to_json(str(output_path))
            elif fmt == "png":
                results.to_png(
                    str(output_path),
                    persona=persona_enum,
                    include_optional=include_optional,
                )
            elif fmt == "model-card":
                results.to_model_card(str(output_path))
            elif fmt == "chai-model-card":
                results.to_chai_model_card(str(output_path))
            elif fmt == "chai-model-card-json":
                results.to_chai_model_card_json(str(output_path))
            elif fmt == "raic-checklist":
                results.to_raic_checkpoint_1(str(output_path))
            elif fmt == "repro-bundle":
                results.to_reproducibility_bundle(str(output_path))
            else:
                console.print(
                    f"[yellow]Unknown format {fmt}, defaulting to HTML[/yellow]"
                )
                results.to_html(str(output_path.with_suffix(".html")), persona=persona_enum)

            console.print(f"  [green]Saved to {output_path}[/green]")
        except (DataValidationError, ConfigurationError, MetricComputationError) as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        except FileNotFoundError as e:
            console.print(f"[red]File not found: {e}[/red]")
            sys.exit(1)
        except (OSError, PermissionError) as e:
            console.print(f"[red]File error: {e}[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            logger.exception("Unexpected error exporting results")
            sys.exit(1)


@main.command()
def version() -> None:
    """Display version and system information."""
    from faircareai import __version__

    console.print(
        Panel.fit(
            f"[bold blue]FairCareAI[/bold blue] v{__version__}\n\n"
            "Healthcare AI Fairness Auditing\n"
            "CHAI-aligned governance framework\n\n"
            "[dim]Package SUGGESTS, humans DECIDE[/dim]",
            border_style="blue",
        )
    )


@main.command()
def info() -> None:
    """Display information about fairness metrics and use cases."""
    info_text = """
[bold]Fairness Metrics[/bold]

[cyan]Demographic Parity[/cyan]
  Equal selection rates across groups.
  Best for: Resource allocation

[cyan]Equalized Odds[/cyan]
  Equal TPR and FPR across groups.
  Best for: Intervention triggers

[cyan]Equal Opportunity[/cyan]
  Equal TPR (sensitivity) across groups.
  Best for: Screening programs

[cyan]Predictive Parity[/cyan]
  Equal PPV across groups.
  Best for: Risk communication

[cyan]Calibration[/cyan]
  Equal calibration curves across groups.
  Best for: Shared decision-making

[bold]Impossibility Theorem[/bold]

Per Chouldechova (2017) and Kleinberg et al. (2017), it is
mathematically impossible to satisfy all fairness metrics
simultaneously when base rates differ between groups.

[bold yellow]The choice of fairness metric is a value judgment
that humans must make based on clinical context.[/bold yellow]
"""
    console.print(
        Panel(
            info_text,
            title="[bold]FairCareAI Fairness Guide[/bold]",
            border_style="blue",
        )
    )


if __name__ == "__main__":
    main()
