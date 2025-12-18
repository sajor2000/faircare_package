"""
FairCareAI Command-Line Interface

Provides command-line access to FairCareAI functionality including:
- Launching the interactive dashboard
- Running audits from configuration files
- Generating reports

Metrics computed per Van Calster et al. (2025) methodology.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


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
    help="Output file path (supports .html, .pdf, .pptx)",
)
@click.option(
    "--threshold",
    default=0.5,
    help="Decision threshold for binary classification (default: 0.5)",
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
    threshold: float,
    model_name: str,
    attributes: tuple[str, ...],
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
    except Exception as e:
        console.print(f"[red]Error loading data:[/red] {e}")
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
        results = audit_obj.run()
    except Exception as e:
        console.print(f"[red]Error running audit:[/red] {e}")
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

    # Export if output specified
    if output:
        output_path = Path(output)
        console.print(f"\n[bold]Exporting to {output_path}...[/bold]")

        try:
            if output_path.suffix == ".html":
                results.to_html(str(output_path))
            elif output_path.suffix == ".pdf":
                results.to_pdf(str(output_path))
            elif output_path.suffix == ".pptx":
                results.to_pptx(str(output_path))
            else:
                console.print(
                    f"[yellow]Unknown format {output_path.suffix}, defaulting to HTML[/yellow]"
                )
                results.to_html(str(output_path.with_suffix(".html")))

            console.print(f"  [green]Saved to {output_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error exporting:[/red] {e}")
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
