#!/usr/bin/env python3
"""
FairCareAI ICU Mortality Demo

End-to-end demonstration of the fairness auditing pipeline
using synthetic ICU mortality prediction data.

Usage:
    python examples/icu_mortality_demo.py

Or with the package installed:
    cd /path/to/faircareai
    pip install -e .
    python examples/icu_mortality_demo.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from faircareai import FairCareAudit, FairnessConfig
from faircareai.core.config import FairnessMetric, UseCaseType
from faircareai.data.synthetic import generate_icu_mortality_data, get_data_summary


def main():
    """Run the complete fairness audit demo."""
    console = Console()

    # Header
    console.print(Panel.fit(
        "[bold blue]FairCareAI ICU Mortality Demo[/bold blue]\n"
        "Synthetic data fairness auditing demonstration",
        border_style="blue",
    ))
    console.print()

    # Step 1: Generate synthetic data
    console.print("[bold]Step 1:[/bold] Generating synthetic ICU mortality data...")

    df = generate_icu_mortality_data(
        n_samples=2000,
        seed=42,
        disparity_strength=0.08,  # 8% TPR gap
    )

    summary = get_data_summary(df)
    console.print(f"  Generated {summary['n_samples']:,} patient records")
    console.print(f"  Mortality rate: {summary['mortality_rate']:.1%}")
    console.print(f"  Prediction rate: {summary['prediction_rate']:.1%}")
    console.print()

    # Show demographic distribution
    console.print("[bold]Demographic Distribution:[/bold]")

    for attr in ["race_ethnicity", "insurance", "language"]:
        table = Table(title=attr.replace("_", " ").title(), show_header=True)
        table.add_column("Group")
        table.add_column("N", justify="right")

        for item in summary["demographics"][attr]:
            table.add_row(str(item[attr]), str(item["count"]))

        console.print(table)
        console.print()

    # Step 2: Initialize and configure audit
    console.print("[bold]Step 2:[/bold] Configuring fairness audit...")

    audit = FairCareAudit(
        data=df,
        pred_col="prediction",
        target_col="mortality",
        threshold=0.5,
    )

    # Accept suggested attributes
    suggestions = audit.suggest_attributes(display=False)
    if suggestions:
        console.print(f"  Detected {len(suggestions)} sensitive attributes")
        audit.accept_suggested_attributes([1, 2, 3])
        console.print(f"  Configured: race_ethnicity, insurance, language")

    # Configure
    audit.config = FairnessConfig(
        model_name="ICU Mortality Prediction Model v1.0",
        model_version="1.0.0",
        intended_use="Trigger early intervention for high-risk ICU patients",
        intended_population="Adult patients in medical/surgical ICU",
        primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
        fairness_justification=(
            "Model triggers intervention. Equalized odds ensures "
            "equal TPR/FPR across demographic groups."
        ),
        use_case_type=UseCaseType.INTERVENTION_TRIGGER,
    )

    console.print()

    # Step 3: Run audit
    console.print("[bold]Step 3:[/bold] Running fairness audit...")
    console.print("  Computing performance metrics...")
    console.print("  Computing fairness metrics...")
    console.print("  Generating flags...")

    result = audit.run(
        bootstrap_ci=True,
        n_bootstrap=500,  # Reduced for demo speed
    )

    console.print("  Done!")
    console.print()

    # Step 4: Display results
    console.print("[bold]Step 4:[/bold] Audit Results")
    console.print()

    # Governance recommendation
    rec = result.governance_recommendation
    status_color = {
        "READY": "green",
        "CONDITIONAL": "yellow",
        "REVIEW_REQUIRED": "red",
    }.get(rec["status"], "white")

    console.print(Panel.fit(
        f"[bold]Governance Recommendation[/bold]\n\n"
        f"Status: [{status_color}]{rec['status']}[/{status_color}]\n"
        f"Advisory: {rec['advisory']}\n\n"
        f"Checks: {rec['n_pass']} pass, {rec['n_warnings']} warn, {rec['n_errors']} error",
        border_style=status_color,
    ))
    console.print()

    # Show flags if any
    if result.flags:
        console.print("[bold]Flags Identified:[/bold]")
        for flag in result.flags[:5]:  # Show first 5
            severity_color = "red" if flag["severity"] == "error" else "yellow"
            console.print(f"  [{severity_color}][{flag['severity'].upper()}][/{severity_color}] {flag['message']}")
        if len(result.flags) > 5:
            console.print(f"  ... and {len(result.flags) - 5} more")
        console.print()

    # Show subgroup metrics table
    console.print("[bold]Subgroup Performance (Race/Ethnicity):[/bold]")

    if "race_ethnicity" in result.subgroup_performance:
        metrics_table = Table(show_header=True)
        metrics_table.add_column("Group")
        metrics_table.add_column("N", justify="right")
        metrics_table.add_column("TPR", justify="right")
        metrics_table.add_column("FPR", justify="right")

        subgroup_data = result.subgroup_performance["race_ethnicity"]
        groups_data = subgroup_data.get("groups", subgroup_data)

        for group, metrics in groups_data.items():
            if group in ("reference", "attribute", "threshold"):
                continue
            if isinstance(metrics, dict):
                n = metrics.get("n", "?")
                tpr = metrics.get("tpr", {})
                fpr = metrics.get("fpr", {})

                tpr_val = tpr.get("value", "N/A") if isinstance(tpr, dict) else tpr
                fpr_val = fpr.get("value", "N/A") if isinstance(fpr, dict) else fpr

                tpr_str = f"{tpr_val:.3f}" if isinstance(tpr_val, float) else str(tpr_val)
                fpr_str = f"{fpr_val:.3f}" if isinstance(fpr_val, float) else str(fpr_val)

                metrics_table.add_row(str(group), str(n), tpr_str, fpr_str)

        console.print(metrics_table)
    console.print()

    # Show fairness metrics
    console.print("[bold]Fairness Metrics (Race/Ethnicity):[/bold]")

    if "race_ethnicity" in result.fairness_metrics:
        fairness_data = result.fairness_metrics["race_ethnicity"]

        dp_ratios = fairness_data.get("demographic_parity_ratio", {})
        eo_diffs = fairness_data.get("equalized_odds_diff", {})

        if dp_ratios:
            console.print("  Demographic Parity Ratios:")
            for group, ratio in dp_ratios.items():
                if ratio is not None:
                    status = "[green]OK[/green]" if 0.8 <= ratio <= 1.25 else "[yellow]FLAG[/yellow]"
                    console.print(f"    {group}: {ratio:.3f} {status}")

        if eo_diffs:
            console.print("  Equalized Odds Differences (TPR):")
            for group, diff in eo_diffs.items():
                if diff is not None:
                    status = "[green]OK[/green]" if abs(diff) <= 0.1 else "[yellow]FLAG[/yellow]"
                    console.print(f"    {group}: {diff:+.3f} {status}")

    console.print()

    # Step 5: Export options
    console.print("[bold]Step 5:[/bold] Export Options")
    console.print()
    console.print("To export reports, use:")
    console.print("  [cyan]result.to_html('fairness_report.html')[/cyan]")
    console.print("  [cyan]result.to_pdf('fairness_report.pdf')[/cyan]")
    console.print("  [cyan]result.to_pptx('governance_deck.pptx')[/cyan]")
    console.print()

    # Step 6: Dashboard
    console.print("[bold]Step 6:[/bold] Interactive Dashboard")
    console.print()
    console.print("To launch the interactive dashboard:")
    console.print("  [cyan]import faircareai[/cyan]")
    console.print("  [cyan]faircareai.launch()[/cyan]")
    console.print()
    console.print("Or from command line: [cyan]faircareai dashboard[/cyan]")
    console.print()

    # Final summary
    n_warnings = rec["n_warnings"]
    n_errors = rec["n_errors"]

    if n_errors > 0:
        final_status = "[red]REVIEW REQUIRED[/red]"
        border = "red"
    elif n_warnings > 0:
        final_status = "[yellow]CONDITIONAL[/yellow]"
        border = "yellow"
    else:
        final_status = "[green]READY[/green]"
        border = "green"

    console.print(Panel.fit(
        f"[bold]Audit Complete![/bold]\n\n"
        f"Status: {final_status}\n\n"
        f"[dim]Remember: All FairCareAI outputs are ADVISORY.\n"
        f"Final deployment decisions rest with clinical stakeholders.[/dim]",
        border_style=border,
    ))


if __name__ == "__main__":
    main()
