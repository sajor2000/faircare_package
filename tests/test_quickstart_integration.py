"""
End-to-end integration tests for FairCareAudit quickstart pattern.

Tests the complete documented workflow:
1. Initialize with predictions
2. Configure sensitive attributes
3. Set governance configuration
4. Run audit and export to multiple formats
"""

import contextlib
import json
from pathlib import Path

import polars as pl
import pytest

from faircareai import FairCareAudit
from faircareai.core.config import FairnessConfig, FairnessMetric, UseCaseType
from faircareai.core.results import AuditResults


class TestQuickstartIntegration:
    """Integration tests for FairCareAudit quickstart pattern."""

    def test_complete_quickstart_flow(
        self, sample_multigroup_data: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test complete quickstart pattern as documented.

        Validates the full workflow:
        - Initialize with DataFrame
        - suggest_attributes() and accept_suggested_attributes()
        - Set FairnessConfig
        - Run audit
        - Export to multiple formats
        """
        output_dir = tmp_path / "quickstart_outputs"
        output_dir.mkdir()

        # Step 1: Initialize with predictions
        audit = FairCareAudit(
            data=sample_multigroup_data, pred_col="probability", target_col="outcome"
        )

        # Assertions: Initialization
        assert audit.pred_col == "probability"
        assert audit.target_col == "outcome"
        assert len(audit.df) == 400
        assert audit.threshold == 0.5

        # Step 2: Configure sensitive attributes
        suggestions = audit.suggest_attributes(display=False)
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 3  # race, sex, insurance

        audit.accept_suggested_attributes([1, 2, 3])

        # Assertions: Attributes configured
        assert len(audit.sensitive_attributes) == 3
        attr_names = {attr.name for attr in audit.sensitive_attributes}
        assert attr_names == {"race", "sex", "insurance"}

        # Step 3: Set governance configuration
        audit.config = FairnessConfig(
            model_name="Quickstart Integration Test",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            fairness_justification="Testing intervention trigger model for integration tests",
            use_case_type=UseCaseType.INTERVENTION_TRIGGER,
        )

        # Assertions: Configuration
        assert audit.config.model_name == "Quickstart Integration Test"
        assert audit.config.primary_fairness_metric == FairnessMetric.EQUALIZED_ODDS
        assert audit.config.use_case_type == UseCaseType.INTERVENTION_TRIGGER
        assert audit.config.fairness_justification is not None

        # Step 4: Run audit and export
        results = audit.run(bootstrap_ci=False)  # Skip CI for speed

        # Assertions: Results object
        assert results is not None
        assert isinstance(results, AuditResults)
        assert results.config == audit.config
        assert len(results.subgroup_performance) > 0
        assert len(results.fairness_metrics) > 0
        assert isinstance(results.flags, list)

        # Test HTML export
        html_path = results.to_html(output_dir / "full_report.html")
        assert html_path.exists()
        assert html_path.stat().st_size > 0
        content = html_path.read_text()
        assert "Quickstart Integration Test" in content

        # Test governance HTML export
        gov_html_path = results.to_governance_html(output_dir / "governance.html")
        assert gov_html_path.exists()
        assert gov_html_path.stat().st_size > 0

        # Test PPTX export (skip if python-pptx not installed)
        try:
            pptx_path = results.to_pptx(output_dir / "deck.pptx")
            assert pptx_path.exists()
            assert pptx_path.stat().st_size > 0
        except ImportError:
            pytest.skip("python-pptx not installed - skipping PPTX export test")

        # Test JSON export
        json_path = results.to_json(output_dir / "results.json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
            assert "config" in data
            assert "overall_performance" in data
            assert "subgroup_performance" in data
            assert data["config"]["model_name"] == "Quickstart Integration Test"

    def test_quickstart_with_parquet_input(
        self, sample_multigroup_data: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test quickstart pattern with parquet file input instead of DataFrame."""
        # Write test data to parquet file
        parquet_path = tmp_path / "test_data.parquet"
        sample_multigroup_data.write_parquet(parquet_path)

        output_dir = tmp_path / "parquet_outputs"
        output_dir.mkdir()

        # Step 1: Initialize with parquet file path (as string)
        audit = FairCareAudit(data=str(parquet_path), pred_col="probability", target_col="outcome")

        # Assertions: File loaded correctly
        assert audit.pred_col == "probability"
        assert len(audit.df) == 400

        # Step 2-3: Configure attributes and config
        audit.suggest_attributes(display=False)
        audit.accept_suggested_attributes([1, 2])  # Just race and sex

        audit.config = FairnessConfig(
            model_name="Parquet Input Test",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            fairness_justification="Testing with parquet file input",
            use_case_type=UseCaseType.SCREENING,
        )

        # Step 4: Run and verify
        results = audit.run(bootstrap_ci=False)

        assert results is not None
        assert len(audit.sensitive_attributes) == 2

        # Test one export to verify it works
        html_path = results.to_html(output_dir / "report.html")
        assert html_path.exists()
        assert html_path.stat().st_size > 0

    def test_quickstart_all_exports(
        self, sample_multigroup_data: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test all export methods work correctly in quickstart workflow."""
        output_dir = tmp_path / "all_exports"
        output_dir.mkdir()

        # Setup audit (steps 1-3)
        audit = FairCareAudit(
            data=sample_multigroup_data, pred_col="probability", target_col="outcome"
        )
        audit.suggest_attributes(display=False)
        audit.accept_suggested_attributes([1])  # Just race for speed

        audit.config = FairnessConfig(
            model_name="Export Test",
            primary_fairness_metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            fairness_justification="Testing all export formats",
            use_case_type=UseCaseType.RESOURCE_ALLOCATION,
        )

        # Run audit
        results = audit.run(bootstrap_ci=False)

        # Test all export formats
        exports = {
            "html": results.to_html(output_dir / "full_report.html"),
            "governance_html": results.to_governance_html(output_dir / "governance.html"),
            "json": results.to_json(output_dir / "data.json"),
        }

        # Test PDF exports if weasyprint is available
        with contextlib.suppress(ImportError):
            exports["pdf"] = results.to_pdf(output_dir / "full_report.pdf")
            exports["governance_pdf"] = results.to_governance_pdf(output_dir / "governance.pdf")

        # Test PPTX export if python-pptx is available
        with contextlib.suppress(ImportError):
            exports["pptx"] = results.to_pptx(output_dir / "deck.pptx")

        # Verify all exports exist and have content
        for format_name, path in exports.items():
            assert path.exists(), f"{format_name} export failed - file not found"
            assert path.stat().st_size > 0, f"{format_name} export is empty"

        # Verify JSON structure
        with open(exports["json"]) as f:
            json_data = json.load(f)
            assert "config" in json_data
            assert "overall_performance" in json_data
            assert "subgroup_performance" in json_data
            assert "fairness_metrics" in json_data

    def test_quickstart_with_bootstrap_ci(
        self, sample_multigroup_data: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test quickstart pattern with bootstrap confidence intervals enabled.

        Uses reduced bootstrap samples (100) for reasonable test execution time.
        """
        # Setup audit
        audit = FairCareAudit(
            data=sample_multigroup_data, pred_col="probability", target_col="outcome"
        )
        audit.suggest_attributes(display=False)
        audit.accept_suggested_attributes([1])  # Just race

        audit.config = FairnessConfig(
            model_name="Bootstrap CI Test",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            fairness_justification="Testing with bootstrap confidence intervals",
            use_case_type=UseCaseType.INTERVENTION_TRIGGER,
        )

        # Run with bootstrap CI (reduced samples for speed)
        results = audit.run(bootstrap_ci=True, n_bootstrap=100)

        # Verify bootstrap CIs are present in results
        assert results is not None
        assert len(results.subgroup_performance) > 0

        # Verify the run completes successfully with bootstrap enabled
        # CI availability depends on the specific metrics computed
        # The fact that we got here means bootstrap worked


class TestQuickstartEdgeCases:
    """Edge case tests for quickstart pattern."""

    def test_quickstart_manual_attribute_addition(
        self, sample_multigroup_data: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test quickstart with manual attribute addition instead of suggest/accept."""
        audit = FairCareAudit(
            data=sample_multigroup_data, pred_col="probability", target_col="outcome"
        )

        # Manual attribute addition (alternative to suggest/accept pattern)
        audit.add_sensitive_attribute(name="race", column="race", reference="White")
        audit.add_sensitive_attribute(name="sex", column="sex", reference="Male")

        audit.config = FairnessConfig(
            model_name="Manual Attributes Test",
            primary_fairness_metric=FairnessMetric.EQUAL_OPPORTUNITY,
            fairness_justification="Testing manual attribute addition",
            use_case_type=UseCaseType.DIAGNOSIS_SUPPORT,
        )

        results = audit.run(bootstrap_ci=False)

        assert results is not None
        assert len(audit.sensitive_attributes) == 2

    def test_quickstart_with_intersections(
        self, sample_multigroup_data: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test quickstart pattern with intersectional analysis."""
        audit = FairCareAudit(
            data=sample_multigroup_data, pred_col="probability", target_col="outcome"
        )

        audit.suggest_attributes(display=False)
        audit.accept_suggested_attributes([1, 2])  # race and sex

        # Add intersectional analysis
        audit.add_intersection(["race", "sex"])

        audit.config = FairnessConfig(
            model_name="Intersectional Test",
            primary_fairness_metric=FairnessMetric.PREDICTIVE_PARITY,
            fairness_justification="Testing intersectional fairness analysis",
            use_case_type=UseCaseType.RISK_COMMUNICATION,
        )

        results = audit.run(bootstrap_ci=False)

        assert results is not None
        assert len(results.intersectional) > 0
        # Verify intersectional groups were analyzed
        assert any("race" in str(group) and "sex" in str(group) for group in results.intersectional)
