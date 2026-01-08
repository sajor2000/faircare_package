"""
Integration test for threshold display in figures and reports.

Verifies that custom thresholds (e.g., 0.414) appear correctly in:
- Report text
- Chart titles (confusion matrix, classification metrics, etc.)
- Figure data structures
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from faircareai import FairCareAudit
from faircareai.core.config import FairnessConfig, FairnessMetric


class TestThresholdInFigures:
    """Test that custom thresholds appear correctly in all visualizations."""

    @pytest.fixture
    def sample_data_with_predictions(self) -> pl.DataFrame:
        """Create sample data for threshold testing."""
        import numpy as np

        np.random.seed(42)
        n = 500

        # Create predictions and outcomes
        risk_scores = np.random.beta(2, 5, n)
        outcomes = (np.random.random(n) < risk_scores).astype(int)
        race = np.random.choice(["White", "Black", "Hispanic"], n)

        return pl.DataFrame(
            {
                "risk_score": risk_scores,
                "outcome": outcomes,
                "race": race,
            }
        )

    @pytest.fixture
    def basic_config(self) -> FairnessConfig:
        """Create basic FairnessConfig with required fields."""
        return FairnessConfig(
            model_name="Test Threshold Model",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            fairness_justification="Integration test for threshold display verification",
        )

    def test_threshold_in_audit_results(
        self, sample_data_with_predictions: pl.DataFrame, basic_config: FairnessConfig
    ) -> None:
        """Test that custom threshold is stored in AuditResults."""
        custom_threshold = 0.414

        audit = FairCareAudit(
            data=sample_data_with_predictions,
            pred_col="risk_score",
            target_col="outcome",
            threshold=custom_threshold,
            config=basic_config,
        )
        audit.add_sensitive_attribute("race")

        results = audit.run()

        # Verify threshold is stored in AuditResults
        assert results.threshold == custom_threshold

    def test_threshold_in_performance_metrics(
        self, sample_data_with_predictions: pl.DataFrame, basic_config: FairnessConfig
    ) -> None:
        """Test that threshold appears in performance metric dictionaries."""
        custom_threshold = 0.414

        audit = FairCareAudit(
            data=sample_data_with_predictions,
            pred_col="risk_score",
            target_col="outcome",
            threshold=custom_threshold,
            config=basic_config,
        )
        audit.add_sensitive_attribute("race")

        results = audit.run()

        # Check overall performance
        perf = results.overall_performance
        assert perf is not None
        assert "primary_threshold" in perf
        assert perf["primary_threshold"] == custom_threshold

        # Check classification_at_threshold
        cls = perf.get("classification_at_threshold", {})
        assert "threshold" in cls
        assert cls["threshold"] == custom_threshold

        # Check confusion_matrix
        cm = perf.get("confusion_matrix", {})
        assert "threshold" in cm
        assert cm["threshold"] == custom_threshold

    def test_threshold_in_confusion_matrix_visualization(
        self, sample_data_with_predictions: pl.DataFrame, basic_config: FairnessConfig
    ) -> None:
        """Test that threshold appears in confusion matrix chart title."""
        from faircareai.visualization.performance_charts import plot_confusion_matrix

        custom_threshold = 0.414

        audit = FairCareAudit(
            data=sample_data_with_predictions,
            pred_col="risk_score",
            target_col="outcome",
            threshold=custom_threshold,
            config=basic_config,
        )
        audit.add_sensitive_attribute("race")

        results = audit.run()

        # Generate confusion matrix figure
        fig = plot_confusion_matrix(results)

        # Check that threshold appears in title
        assert fig.layout.title is not None
        title_text = fig.layout.title.text
        assert title_text is not None

        # Should contain the custom threshold, not 0.5
        assert "0.41" in title_text or "0.414" in title_text
        assert "0.5" not in title_text or "0.50" not in title_text

    def test_threshold_in_governance_figures(
        self, sample_data_with_predictions: pl.DataFrame, basic_config: FairnessConfig
    ) -> None:
        """Test that threshold appears in governance dashboard figures."""
        from faircareai.visualization.governance_dashboard import (
            create_governance_overall_figures,
        )

        custom_threshold = 0.414

        audit = FairCareAudit(
            data=sample_data_with_predictions,
            pred_col="risk_score",
            target_col="outcome",
            threshold=custom_threshold,
            config=basic_config,
        )
        audit.add_sensitive_attribute("race")

        results = audit.run()

        # Generate governance figures
        figures = create_governance_overall_figures(results)

        # Check classification metrics figure
        if "Classification Metrics" in figures or any(
            "Classification" in k for k in figures.keys()
        ):
            for title, fig in figures.items():
                if "Classification" in title and fig is not None:
                    # Check title contains custom threshold
                    if fig.layout.title is not None:
                        title_text = fig.layout.title.text
                        if title_text and "Threshold" in title_text:
                            # Should contain 0.41, not 0.5
                            assert "0.41" in title_text or "0.414" in title_text
                            # Make sure it's not showing default
                            if "0.5" in title_text:
                                # Allow 0.5X but not exactly 0.50 or 0.5
                                assert "0.50" not in title_text

    def test_threshold_in_audit_summary(
        self, sample_data_with_predictions: pl.DataFrame, basic_config: FairnessConfig
    ) -> None:
        """Test that threshold flows through to AuditSummary."""
        custom_threshold = 0.414

        audit = FairCareAudit(
            data=sample_data_with_predictions,
            pred_col="risk_score",
            target_col="outcome",
            threshold=custom_threshold,
            config=basic_config,
        )
        audit.add_sensitive_attribute("race")

        results = audit.run()

        # Convert to AuditSummary (used for report generation)
        summary = results._to_audit_summary()

        # Verify threshold is preserved
        assert summary.threshold == custom_threshold

    @pytest.mark.skip(reason="Requires Playwright - run manually if needed")
    def test_threshold_in_pdf_report(
        self, sample_data_with_predictions: pl.DataFrame, basic_config: FairnessConfig
    ) -> None:
        """Test that threshold appears correctly in generated PDF report."""
        custom_threshold = 0.414

        audit = FairCareAudit(
            data=sample_data_with_predictions,
            pred_col="risk_score",
            target_col="outcome",
            threshold=custom_threshold,
            config=basic_config,
        )
        audit.add_sensitive_attribute("race")

        results = audit.run()

        # Generate PDF report
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test_report.pdf"

            # This will fail if Playwright not installed, hence the skip
            try:
                results.to_pdf(pdf_path, persona="governance")
                assert pdf_path.exists()
            except ImportError:
                pytest.skip("Playwright not installed")

    def test_threshold_in_html_report(
        self, sample_data_with_predictions: pl.DataFrame, basic_config: FairnessConfig
    ) -> None:
        """Test that threshold appears correctly in HTML report content."""
        custom_threshold = 0.414

        audit = FairCareAudit(
            data=sample_data_with_predictions,
            pred_col="risk_score",
            target_col="outcome",
            threshold=custom_threshold,
            config=basic_config,
        )
        audit.add_sensitive_attribute("race")

        results = audit.run()

        # Generate HTML report
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "test_report.html"
            results.to_html(html_path, persona="governance")

            assert html_path.exists()

            # Read HTML and check for threshold
            html_content = html_path.read_text()

            # Should contain custom threshold in text
            assert "0.414" in html_content or "0.41" in html_content or "41%" in html_content

            # Should not prominently display default 0.5 as THE threshold
            # (0.5 might appear in other contexts, but not as the main threshold)
            # We'll check that 41.4% or 0.414 appears near "Threshold"
            import re

            # Look for patterns like "Threshold: 0.414" or "Threshold = 0.41"
            threshold_pattern = r"[Tt]hreshold[:\s=]+0\.4[01234]"
            matches = re.findall(threshold_pattern, html_content)
            assert len(matches) > 0, "Custom threshold should appear in HTML report"
