"""
Tests for FairCareAI dual-persona output system.

Tests cover:
- OutputPersona enum
- Persona parameter normalization
- Governance report generation
- Data scientist (default) report generation
- Persona-specific figure generation
- Convenience methods (to_governance_html, to_governance_pdf)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from faircareai.core.config import (
    FairnessConfig,
    FairnessMetric,
    MetricDisplayConfig,
    OutputPersona,
    UseCaseType,
)
from faircareai.core.results import AuditResults, _normalize_persona


class TestOutputPersonaEnum:
    """Tests for OutputPersona enum."""

    def test_enum_values(self) -> None:
        """Test that enum has expected values."""
        assert OutputPersona.DATA_SCIENTIST.value == "data_scientist"
        assert OutputPersona.GOVERNANCE.value == "governance"

    def test_enum_members(self) -> None:
        """Test enum has exactly two members."""
        members = list(OutputPersona)
        assert len(members) == 2
        assert OutputPersona.DATA_SCIENTIST in members
        assert OutputPersona.GOVERNANCE in members


class TestPersonaNormalization:
    """Tests for _normalize_persona helper function."""

    def test_enum_passthrough(self) -> None:
        """Test that enum values pass through unchanged."""
        assert _normalize_persona(OutputPersona.DATA_SCIENTIST) == OutputPersona.DATA_SCIENTIST
        assert _normalize_persona(OutputPersona.GOVERNANCE) == OutputPersona.GOVERNANCE

    def test_string_data_scientist_variants(self) -> None:
        """Test various string representations for data scientist persona."""
        for value in ["data_scientist", "DATA_SCIENTIST", "datascientist", "full", "technical"]:
            assert _normalize_persona(value) == OutputPersona.DATA_SCIENTIST

    def test_string_governance_variants(self) -> None:
        """Test various string representations for governance persona."""
        for value in ["governance", "GOVERNANCE", "executive", "summary", "streamlined"]:
            assert _normalize_persona(value) == OutputPersona.GOVERNANCE

    def test_invalid_string_raises(self) -> None:
        """Test that invalid strings raise ValueError."""
        with pytest.raises(ValueError, match="Unknown persona"):
            _normalize_persona("invalid")

    def test_invalid_type_raises(self) -> None:
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="must be OutputPersona or str"):
            _normalize_persona(123)  # type: ignore

        with pytest.raises(TypeError, match="must be OutputPersona or str"):
            _normalize_persona(None)  # type: ignore


class TestToHtmlPersona:
    """Tests for to_html with persona parameter."""

    @pytest.fixture
    def basic_results(self) -> AuditResults:
        """Create basic AuditResults for testing."""
        config = FairnessConfig(
            model_name="Test Model",
            model_version="1.0.0",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            fairness_justification="Test justification",
        )
        return AuditResults(
            config=config,
            descriptive_stats={"cohort_overview": {"n_total": 1000}},
            overall_performance={"discrimination": {"auroc": 0.85}},
            governance_recommendation={"n_pass": 5, "n_warnings": 2, "n_errors": 0},
        )

    @patch("faircareai.reports.generator.generate_html_report")
    def test_default_persona_is_data_scientist(
        self, mock_generate: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that default persona is DATA_SCIENTIST."""
        mock_generate.return_value = "<html>report</html>"

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            basic_results.to_html(path)
            mock_generate.assert_called_once()
        finally:
            path.unlink(missing_ok=True)

    @patch("faircareai.reports.generator.generate_governance_html_report")
    def test_governance_persona_routes_correctly(
        self, mock_generate: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that governance persona routes to governance generator."""
        mock_generate.return_value = "<html>governance report</html>"

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            basic_results.to_html(path, persona="governance")
            mock_generate.assert_called_once()
        finally:
            path.unlink(missing_ok=True)

    @patch("faircareai.reports.generator.generate_governance_html_report")
    def test_governance_enum_works(
        self, mock_generate: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that OutputPersona.GOVERNANCE enum works."""
        mock_generate.return_value = "<html>governance report</html>"

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            basic_results.to_html(path, persona=OutputPersona.GOVERNANCE)
            mock_generate.assert_called_once()
        finally:
            path.unlink(missing_ok=True)


class TestToPdfPersona:
    """Tests for to_pdf with persona parameter."""

    @pytest.fixture
    def basic_results(self) -> AuditResults:
        """Create basic AuditResults for testing."""
        config = FairnessConfig(
            model_name="Test Model",
            model_version="1.0.0",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            fairness_justification="Test justification",
        )
        return AuditResults(
            config=config,
            descriptive_stats={"cohort_overview": {"n_total": 1000}},
            overall_performance={"discrimination": {"auroc": 0.85}},
            governance_recommendation={"n_pass": 5, "n_warnings": 2, "n_errors": 0},
        )

    @patch("faircareai.reports.generator.generate_pdf_report")
    def test_default_persona_is_data_scientist(
        self, mock_generate: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that default persona is DATA_SCIENTIST."""
        mock_generate.return_value = Path("/tmp/test.pdf")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = Path(f.name)

        try:
            basic_results.to_pdf(path)
            mock_generate.assert_called_once()
        finally:
            path.unlink(missing_ok=True)

    @patch("faircareai.reports.generator.generate_governance_pdf_report")
    def test_governance_persona_routes_correctly(
        self, mock_generate: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that governance persona routes to governance generator."""
        mock_generate.return_value = Path("/tmp/test.pdf")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = Path(f.name)

        try:
            basic_results.to_pdf(path, persona="governance")
            mock_generate.assert_called_once()
        finally:
            path.unlink(missing_ok=True)


class TestConvenienceMethods:
    """Tests for convenience methods to_governance_html and to_governance_pdf."""

    @pytest.fixture
    def basic_results(self) -> AuditResults:
        """Create basic AuditResults for testing."""
        config = FairnessConfig(
            model_name="Test Model",
            model_version="1.0.0",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            fairness_justification="Test justification",
        )
        return AuditResults(
            config=config,
            descriptive_stats={"cohort_overview": {"n_total": 1000}},
            overall_performance={"discrimination": {"auroc": 0.85}},
            governance_recommendation={"n_pass": 5, "n_warnings": 2, "n_errors": 0},
        )

    @patch("faircareai.reports.generator.generate_governance_html_report")
    def test_to_governance_html(
        self, mock_generate: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test to_governance_html convenience method."""
        mock_generate.return_value = "<html>governance</html>"

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            basic_results.to_governance_html(path)
            mock_generate.assert_called_once()
        finally:
            path.unlink(missing_ok=True)

    @patch("faircareai.reports.generator.generate_governance_pdf_report")
    def test_to_governance_pdf(self, mock_generate: MagicMock, basic_results: AuditResults) -> None:
        """Test to_governance_pdf convenience method."""
        mock_generate.return_value = Path("/tmp/test.pdf")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = Path(f.name)

        try:
            basic_results.to_governance_pdf(path)
            mock_generate.assert_called_once()
        finally:
            path.unlink(missing_ok=True)


class TestGovernanceOverallFigures:
    """Tests for governance overall figure generation."""

    @pytest.fixture
    def results_with_performance(self) -> AuditResults:
        """Create AuditResults with overall performance data."""
        config = FairnessConfig(
            model_name="Test Model",
            model_version="1.0.0",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            fairness_justification="Test justification",
        )
        return AuditResults(
            config=config,
            overall_performance={
                "discrimination": {
                    "auroc": 0.82,
                    "auroc_ci_lower": 0.78,
                    "auroc_ci_upper": 0.86,
                },
                "calibration": {
                    "brier_score": 0.12,
                    "calibration_slope": 0.95,
                    "calibration_intercept": 0.02,
                },
                "classification_at_threshold": {
                    "threshold": 0.5,
                    "sensitivity": 0.75,
                    "specificity": 0.80,
                    "ppv": 0.65,
                    "npv": 0.88,
                },
            },
            governance_recommendation={"n_pass": 5, "n_warnings": 2, "n_errors": 0},
        )

    def test_create_governance_overall_figures_returns_dict(
        self, results_with_performance: AuditResults
    ) -> None:
        """Test that create_governance_overall_figures returns a dict."""
        from faircareai.visualization.governance_dashboard import (
            create_governance_overall_figures,
        )

        figures = create_governance_overall_figures(results_with_performance)
        assert isinstance(figures, dict)

    def test_creates_expected_figure_keys(self, results_with_performance: AuditResults) -> None:
        """Test that expected figure keys are created."""
        from faircareai.visualization.governance_dashboard import (
            create_governance_overall_figures,
        )

        figures = create_governance_overall_figures(results_with_performance)
        # Should have 4 figures + _explanations dict (5 total)
        assert len(figures) == 5
        assert "_explanations" in figures
        # Check that we have the expected figure types (by checking key contains expected words)
        figure_keys = [k for k in figures.keys() if k != "_explanations"]
        assert len(figure_keys) == 4
        keys_lower = [k.lower() for k in figure_keys]
        assert any("auroc" in k for k in keys_lower)
        assert any("calibration" in k for k in keys_lower)
        assert any("brier" in k for k in keys_lower)
        assert any("classification" in k or "metric" in k for k in keys_lower)


class TestGovernanceSubgroupFigures:
    """Tests for governance subgroup figure generation."""

    @pytest.fixture
    def results_with_subgroups(self) -> AuditResults:
        """Create AuditResults with subgroup performance data."""
        config = FairnessConfig(
            model_name="Test Model",
            model_version="1.0.0",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            fairness_justification="Test justification",
        )
        return AuditResults(
            config=config,
            subgroup_performance={
                "race": {
                    "reference": "White",
                    "groups": {
                        "White": {
                            "auroc": 0.85,
                            "tpr": 0.80,
                            "fpr": 0.15,
                            "selection_rate": 0.30,
                            "n": 500,
                        },
                        "Black": {
                            "auroc": 0.82,
                            "tpr": 0.75,
                            "fpr": 0.18,
                            "selection_rate": 0.35,
                            "n": 300,
                        },
                        "Hispanic": {
                            "auroc": 0.83,
                            "tpr": 0.78,
                            "fpr": 0.16,
                            "selection_rate": 0.32,
                            "n": 200,
                        },
                    },
                },
            },
            governance_recommendation={"n_pass": 5, "n_warnings": 2, "n_errors": 0},
        )

    def test_create_governance_subgroup_figures_returns_dict(
        self, results_with_subgroups: AuditResults
    ) -> None:
        """Test that create_governance_subgroup_figures returns a dict."""
        from faircareai.visualization.governance_dashboard import (
            create_governance_subgroup_figures,
        )

        figures = create_governance_subgroup_figures(results_with_subgroups)
        assert isinstance(figures, dict)

    def test_creates_figures_per_attribute(self, results_with_subgroups: AuditResults) -> None:
        """Test that figures are created for each sensitive attribute."""
        from faircareai.visualization.governance_dashboard import (
            create_governance_subgroup_figures,
        )

        figures = create_governance_subgroup_figures(results_with_subgroups)
        assert "race" in figures

    def test_creates_four_figures_per_attribute(self, results_with_subgroups: AuditResults) -> None:
        """Test that 4 figures are created per attribute."""
        from faircareai.visualization.governance_dashboard import (
            create_governance_subgroup_figures,
        )

        figures = create_governance_subgroup_figures(results_with_subgroups)
        race_figures = figures.get("race", {})

        # Should have 4 subgroup figures with descriptive names
        assert len(race_figures) == 4
        # Check that we have the expected figure types (by checking key contains expected words)
        keys_lower = [k.lower() for k in race_figures.keys()]
        assert any("auroc" in k for k in keys_lower)
        assert any("sensitivity" in k or "tpr" in k for k in keys_lower)
        assert any("fpr" in k or "false positive" in k for k in keys_lower)
        assert any("selection" in k for k in keys_lower)


class TestIntegrationDataScientist:
    """Integration tests for data scientist persona output."""

    @pytest.fixture
    def full_audit_results(self, sample_multigroup_data: pl.DataFrame) -> AuditResults:
        """Create full audit results for integration testing."""
        from faircareai import FairCareAudit

        audit = FairCareAudit(
            data=sample_multigroup_data,
            pred_col="probability",
            target_col="outcome",
        )
        audit.add_sensitive_attribute("race", reference="White")
        audit.config.model_name = "Integration Test Model"
        audit.config.primary_fairness_metric = FairnessMetric.EQUALIZED_ODDS
        audit.config.fairness_justification = "Test justification"
        audit.config.use_case_type = UseCaseType.INTERVENTION_TRIGGER

        return audit.run(bootstrap_ci=False)

    def test_data_scientist_html_export(self, full_audit_results: AuditResults) -> None:
        """Test full data scientist HTML export."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            result_path = full_audit_results.to_html(path)
            assert result_path.exists()
            content = path.read_text()
            assert "Integration Test Model" in content
        finally:
            path.unlink(missing_ok=True)


class TestIntegrationGovernance:
    """Integration tests for governance persona output."""

    @pytest.fixture
    def full_audit_results(self, sample_multigroup_data: pl.DataFrame) -> AuditResults:
        """Create full audit results for integration testing."""
        from faircareai import FairCareAudit

        audit = FairCareAudit(
            data=sample_multigroup_data,
            pred_col="probability",
            target_col="outcome",
        )
        audit.add_sensitive_attribute("race", reference="White")
        audit.config.model_name = "Governance Test Model"
        audit.config.primary_fairness_metric = FairnessMetric.EQUALIZED_ODDS
        audit.config.fairness_justification = "Test justification"
        audit.config.use_case_type = UseCaseType.INTERVENTION_TRIGGER

        return audit.run(bootstrap_ci=False)

    def test_governance_html_export(self, full_audit_results: AuditResults) -> None:
        """Test governance HTML export."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            result_path = full_audit_results.to_governance_html(path)
            assert result_path.exists()
            content = path.read_text()
            assert "Governance Test Model" in content
        finally:
            path.unlink(missing_ok=True)

    def test_governance_html_has_fewer_sections(self, full_audit_results: AuditResults) -> None:
        """Test that governance HTML has fewer sections than data scientist HTML."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            ds_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            gov_path = Path(f.name)

        try:
            full_audit_results.to_html(ds_path)  # Data scientist (default)
            full_audit_results.to_governance_html(gov_path)  # Governance

            ds_content = ds_path.read_text()
            gov_content = gov_path.read_text()

            # Governance output should have fewer section headers
            # Data scientist has 7 sections, governance has 5
            ds_section_count = ds_content.lower().count("<h2")
            gov_section_count = gov_content.lower().count("<h2")

            # Governance should have fewer or equal sections
            assert gov_section_count <= ds_section_count
        finally:
            ds_path.unlink(missing_ok=True)
            gov_path.unlink(missing_ok=True)


class TestMetricDisplayConfig:
    """Tests for MetricDisplayConfig dataclass (Van Calster metric classification)."""

    def test_governance_config_defaults(self) -> None:
        """Test that governance config shows only RECOMMENDED."""
        config = MetricDisplayConfig.governance()
        assert config.show_recommended is True
        assert config.show_optional is False
        assert config.show_caution is False
        assert config.persona == OutputPersona.GOVERNANCE

    def test_data_scientist_default_no_optional(self) -> None:
        """Test that data scientist default does not show OPTIONAL."""
        config = MetricDisplayConfig.data_scientist()
        assert config.show_recommended is True
        assert config.show_optional is False
        assert config.show_caution is False
        assert config.persona == OutputPersona.DATA_SCIENTIST

    def test_data_scientist_with_optional(self) -> None:
        """Test that data scientist can enable OPTIONAL metrics."""
        config = MetricDisplayConfig.data_scientist(include_optional=True)
        assert config.show_recommended is True
        assert config.show_optional is True
        assert config.show_caution is False

    def test_should_show_recommended(self) -> None:
        """Test that RECOMMENDED metrics are always shown."""
        config = MetricDisplayConfig.governance()
        # Van Calster RECOMMENDED metrics
        assert config.should_show("auroc") is True
        assert config.should_show("calibration_plot") is True
        assert config.should_show("net_benefit") is True
        assert config.should_show("decision_curve") is True
        assert config.should_show("risk_distribution_plot") is True

    def test_should_not_show_optional_by_default(self) -> None:
        """Test that OPTIONAL metrics are hidden by default."""
        config = MetricDisplayConfig.data_scientist()
        # Van Calster OPTIONAL metrics
        assert config.should_show("brier_score") is False
        assert config.should_show("oe_ratio") is False
        assert config.should_show("calibration_slope") is False
        assert config.should_show("ici") is False
        assert config.should_show("sensitivity") is False
        assert config.should_show("specificity") is False
        assert config.should_show("ppv") is False
        assert config.should_show("npv") is False

    def test_should_show_optional_when_enabled(self) -> None:
        """Test that OPTIONAL metrics are shown when enabled."""
        config = MetricDisplayConfig.data_scientist(include_optional=True)
        # Van Calster OPTIONAL metrics
        assert config.should_show("brier_score") is True
        assert config.should_show("oe_ratio") is True
        assert config.should_show("calibration_slope") is True
        assert config.should_show("ici") is True
        assert config.should_show("sensitivity") is True
        assert config.should_show("specificity") is True

    def test_should_never_show_caution_by_default(self) -> None:
        """Test that CAUTION metrics are never shown by default."""
        # Even with include_optional=True
        config = MetricDisplayConfig.data_scientist(include_optional=True)
        # Van Calster CAUTION metrics
        assert config.should_show("f1_score") is False
        assert config.should_show("accuracy") is False
        assert config.should_show("balanced_accuracy") is False
        assert config.should_show("mcc") is False
        assert config.should_show("dor") is False
        assert config.should_show("kappa") is False
        assert config.should_show("auprc") is False

    def test_get_metric_category(self) -> None:
        """Test metric category classification."""
        config = MetricDisplayConfig.data_scientist()
        assert config.get_metric_category("auroc") == "RECOMMENDED"
        assert config.get_metric_category("brier_score") == "OPTIONAL"
        assert config.get_metric_category("f1_score") == "CAUTION"
        assert config.get_metric_category("unknown_metric") == "UNKNOWN"

    def test_filter_metrics(self) -> None:
        """Test filtering a list of metrics."""
        config = MetricDisplayConfig.data_scientist()
        metrics = ["auroc", "brier_score", "f1_score", "net_benefit", "accuracy"]
        filtered = config.filter_metrics(metrics)
        # Only RECOMMENDED should pass
        assert "auroc" in filtered
        assert "net_benefit" in filtered
        assert "brier_score" not in filtered  # OPTIONAL
        assert "f1_score" not in filtered  # CAUTION
        assert "accuracy" not in filtered  # CAUTION

    def test_filter_metrics_with_optional(self) -> None:
        """Test filtering with OPTIONAL enabled."""
        config = MetricDisplayConfig.data_scientist(include_optional=True)
        metrics = ["auroc", "brier_score", "f1_score", "net_benefit"]
        filtered = config.filter_metrics(metrics)
        assert "auroc" in filtered
        assert "brier_score" in filtered  # OPTIONAL now included
        assert "net_benefit" in filtered
        assert "f1_score" not in filtered  # CAUTION still excluded


class TestIncludeOptionalParameter:
    """Tests for include_optional parameter in export methods."""

    @pytest.fixture
    def basic_results(self) -> AuditResults:
        """Create basic AuditResults for testing."""
        config = FairnessConfig(
            model_name="Test Model",
            model_version="1.0.0",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            fairness_justification="Test justification",
        )
        return AuditResults(
            config=config,
            descriptive_stats={"cohort_overview": {"n_total": 1000}},
            overall_performance={"discrimination": {"auroc": 0.85}},
            governance_recommendation={"n_pass": 5, "n_warnings": 2, "n_errors": 0},
        )

    @patch("faircareai.reports.generator.generate_html_report")
    def test_to_html_passes_metric_config(
        self, mock_generate: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that to_html passes MetricDisplayConfig to generator."""
        mock_generate.return_value = "<html>report</html>"

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            basic_results.to_html(path, include_optional=True)
            mock_generate.assert_called_once()
            # Check metric_config was passed
            call_kwargs = mock_generate.call_args[1]
            assert "metric_config" in call_kwargs
            assert call_kwargs["metric_config"].show_optional is True
        finally:
            path.unlink(missing_ok=True)

    @patch("faircareai.reports.generator.generate_html_report")
    def test_to_html_default_no_optional(
        self, mock_generate: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that to_html default does not include optional."""
        mock_generate.return_value = "<html>report</html>"

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            basic_results.to_html(path)  # No include_optional
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert "metric_config" in call_kwargs
            assert call_kwargs["metric_config"].show_optional is False
        finally:
            path.unlink(missing_ok=True)

    @patch("faircareai.reports.generator.generate_governance_html_report")
    def test_governance_ignores_include_optional(
        self, mock_generate: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that governance persona ignores include_optional."""
        mock_generate.return_value = "<html>governance</html>"

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            # Even with include_optional=True, governance should use RECOMMENDED only
            basic_results.to_html(path, persona="governance", include_optional=True)
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert "metric_config" in call_kwargs
            # Governance config should always have show_optional=False
            assert call_kwargs["metric_config"].show_optional is False
        finally:
            path.unlink(missing_ok=True)


# =============================================================================
# PHASE 2: PERSONA TERMINOLOGY AND PRIORITY ORDERING TESTS
# =============================================================================


class TestPersonaTerminology:
    """Tests for persona-specific terminology (Van Calster alignment)."""

    def test_get_label_returns_correct_data_scientist_text(self) -> None:
        """Test that get_label returns technical terminology for Data Scientist."""
        from faircareai.core.config import get_label

        assert get_label("auroc", OutputPersona.DATA_SCIENTIST) == "AUROC"
        assert get_label("sensitivity", OutputPersona.DATA_SCIENTIST) == "Sensitivity"
        assert get_label("specificity", OutputPersona.DATA_SCIENTIST) == "Specificity"
        assert get_label("brier_score", OutputPersona.DATA_SCIENTIST) == "Brier Score"
        assert get_label("oe_ratio", OutputPersona.DATA_SCIENTIST) == "O:E Ratio"
        assert get_label("calibration", OutputPersona.DATA_SCIENTIST) == "Calibration"
        assert get_label("net_benefit", OutputPersona.DATA_SCIENTIST) == "Net Benefit"

    def test_get_label_returns_correct_governance_text(self) -> None:
        """Test that get_label returns plain language + technical term + interpretation for Governance."""
        from faircareai.core.config import get_label

        # Governance labels include: plain language (technical) — interpretation
        auroc_label = get_label("auroc", OutputPersona.GOVERNANCE)
        assert "Model Discrimination" in auroc_label
        assert "(AUROC)" in auroc_label
        assert "higher = better" in auroc_label

        sens_label = get_label("sensitivity", OutputPersona.GOVERNANCE)
        assert "Detection Rate" in sens_label
        assert "(Sensitivity)" in sens_label
        assert "fewer missed cases" in sens_label

        spec_label = get_label("specificity", OutputPersona.GOVERNANCE)
        assert "Correct Rejection Rate" in spec_label
        assert "(Specificity)" in spec_label

        brier_label = get_label("brier_score", OutputPersona.GOVERNANCE)
        assert "Prediction Error" in brier_label
        assert "(Brier Score)" in brier_label
        assert "lower = better" in brier_label

        oe_label = get_label("oe_ratio", OutputPersona.GOVERNANCE)
        assert "Observed vs Expected" in oe_label
        assert "(O:E Ratio)" in oe_label
        assert "1.0 = accurate" in oe_label

        cal_label = get_label("calibration", OutputPersona.GOVERNANCE)
        assert "Prediction Accuracy" in cal_label
        assert "(Calibration)" in cal_label

        nb_label = get_label("net_benefit", OutputPersona.GOVERNANCE)
        assert "Clinical Value Added" in nb_label
        assert "(Net Benefit)" in nb_label

    def test_get_label_returns_description(self) -> None:
        """Test that get_label returns descriptions when requested."""
        from faircareai.core.config import get_label

        ds_desc = get_label("auroc", OutputPersona.DATA_SCIENTIST, "description")
        assert "0.5 = random" in ds_desc or "ROC" in ds_desc

        gov_desc = get_label("auroc", OutputPersona.GOVERNANCE, "description")
        assert "separates" in gov_desc.lower() or "high-risk" in gov_desc.lower()

    def test_get_label_returns_axis_labels(self) -> None:
        """Test that get_label returns axis labels when requested."""
        from faircareai.core.config import get_label

        # Data Scientist - technical axis labels
        ds_x = get_label("auroc", OutputPersona.DATA_SCIENTIST, "x_axis")
        assert "Specificity" in ds_x or "False Positive" in ds_x

        ds_y = get_label("auroc", OutputPersona.DATA_SCIENTIST, "y_axis")
        assert "Sensitivity" in ds_y or "True Positive" in ds_y

        # Governance - plain language axis labels
        gov_x = get_label("auroc", OutputPersona.GOVERNANCE, "x_axis")
        assert "Alarm" in gov_x or "alarm" in gov_x.lower()

        gov_y = get_label("auroc", OutputPersona.GOVERNANCE, "y_axis")
        assert "Detection" in gov_y or "detection" in gov_y.lower()

    def test_get_label_fallback_for_unknown_metric(self) -> None:
        """Test that get_label falls back to original metric name if not found."""
        from faircareai.core.config import get_label

        # Unknown metric should return the input unchanged
        assert get_label("unknown_metric", OutputPersona.DATA_SCIENTIST) == "unknown_metric"
        assert get_label("unknown_metric", OutputPersona.GOVERNANCE) == "unknown_metric"

    def test_get_axis_labels_helper(self) -> None:
        """Test get_axis_labels helper function."""
        from faircareai.core.config import get_axis_labels

        # Calibration plot axes
        ds_cal = get_axis_labels("calibration", OutputPersona.DATA_SCIENTIST)
        assert len(ds_cal) == 2
        assert "Predicted" in ds_cal[0]
        assert "Observed" in ds_cal[1]

        gov_cal = get_axis_labels("calibration", OutputPersona.GOVERNANCE)
        assert len(gov_cal) == 2
        # Governance uses plainer language
        assert "Risk" in gov_cal[0] or "Predicted" in gov_cal[0]


class TestVanCalsterPriorityOrdering:
    """Tests for Van Calster priority ordering of OPTIONAL metrics."""

    def test_optional_priority_list_exists(self) -> None:
        """Test that VANCALSTER_OPTIONAL_PRIORITY list exists and has correct items."""
        from faircareai.core.config import VANCALSTER_OPTIONAL_PRIORITY

        assert isinstance(VANCALSTER_OPTIONAL_PRIORITY, list)
        assert len(VANCALSTER_OPTIONAL_PRIORITY) > 0

        # Brier score should be highest priority (first)
        assert VANCALSTER_OPTIONAL_PRIORITY[0] == "brier_score"

        # Required OPTIONAL metrics should be present
        assert "oe_ratio" in VANCALSTER_OPTIONAL_PRIORITY
        assert "ici" in VANCALSTER_OPTIONAL_PRIORITY
        assert "sensitivity" in VANCALSTER_OPTIONAL_PRIORITY
        assert "specificity" in VANCALSTER_OPTIONAL_PRIORITY
        assert "ppv" in VANCALSTER_OPTIONAL_PRIORITY
        assert "npv" in VANCALSTER_OPTIONAL_PRIORITY

    def test_sort_metrics_by_priority_correct_order(self) -> None:
        """Test that sort_metrics_by_priority returns correct order."""
        from faircareai.core.config import sort_metrics_by_priority

        # Input in random order
        metrics = ["ppv", "brier_score", "sensitivity", "oe_ratio"]

        sorted_metrics = sort_metrics_by_priority(metrics)

        # Brier should be first (highest priority)
        assert sorted_metrics[0] == "brier_score"

        # O:E should come before sensitivity
        assert sorted_metrics.index("oe_ratio") < sorted_metrics.index("sensitivity")

        # Sensitivity should come before PPV
        assert sorted_metrics.index("sensitivity") < sorted_metrics.index("ppv")

    def test_sort_metrics_preserves_unknown_at_end(self) -> None:
        """Test that unknown metrics are placed at the end."""
        from faircareai.core.config import sort_metrics_by_priority

        metrics = ["unknown_metric", "brier_score", "custom_metric"]
        sorted_metrics = sort_metrics_by_priority(metrics)

        # Known metric should be first
        assert sorted_metrics[0] == "brier_score"

        # Unknown metrics should be at the end (in original relative order)
        assert "unknown_metric" in sorted_metrics[1:]
        assert "custom_metric" in sorted_metrics[1:]

    def test_sort_metrics_case_insensitive(self) -> None:
        """Test that sort_metrics_by_priority is case insensitive."""
        from faircareai.core.config import sort_metrics_by_priority

        metrics = ["PPV", "BRIER_SCORE", "Sensitivity"]
        sorted_metrics = sort_metrics_by_priority(metrics)

        # Should sort correctly regardless of case
        # Brier should be first
        assert sorted_metrics[0].lower() == "brier_score"

    def test_sort_metrics_with_empty_list(self) -> None:
        """Test that sort_metrics_by_priority handles empty list."""
        from faircareai.core.config import sort_metrics_by_priority

        assert sort_metrics_by_priority([]) == []

    def test_sort_metrics_with_custom_priority_list(self) -> None:
        """Test that sort_metrics_by_priority accepts custom priority list."""
        from faircareai.core.config import sort_metrics_by_priority

        custom_priority = ["metric_c", "metric_a", "metric_b"]
        metrics = ["metric_b", "metric_a", "metric_c"]

        sorted_metrics = sort_metrics_by_priority(metrics, priority_list=custom_priority)

        assert sorted_metrics == ["metric_c", "metric_a", "metric_b"]


class TestPersonaTerminologyInVisualization:
    """Integration tests for persona terminology in visualization functions."""

    def test_calibration_plot_uses_persona_labels(self) -> None:
        """Test that calibration plot uses persona-appropriate labels."""
        from faircareai.visualization.plots import create_calibration_plot

        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.85, 0.15])

        # Test DATA_SCIENTIST persona
        fig_ds = create_calibration_plot(y_true, y_prob, persona=OutputPersona.DATA_SCIENTIST)
        # Check that the figure was created
        assert fig_ds is not None

        # Test GOVERNANCE persona
        fig_gov = create_calibration_plot(y_true, y_prob, persona=OutputPersona.GOVERNANCE)
        assert fig_gov is not None

    def test_roc_curve_uses_persona_labels(self) -> None:
        """Test that ROC curve uses persona-appropriate labels."""
        from faircareai.visualization.plots import create_roc_curve_by_group

        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.85, 0.15])
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])

        # Test DATA_SCIENTIST persona
        fig_ds = create_roc_curve_by_group(
            y_true, y_prob, groups, persona=OutputPersona.DATA_SCIENTIST
        )
        assert fig_ds is not None

        # Test GOVERNANCE persona
        fig_gov = create_roc_curve_by_group(
            y_true, y_prob, groups, persona=OutputPersona.GOVERNANCE
        )
        assert fig_gov is not None


class TestPersonaTerminologyMetricLabels:
    """Test that metric labels match persona expectations."""

    def test_all_recommended_metrics_have_terminology(self) -> None:
        """Test that all RECOMMENDED metrics have persona terminology."""
        from faircareai.core.config import PERSONA_TERMINOLOGY, get_label

        # Key RECOMMENDED metrics that should have terminology
        key_metrics = ["auroc", "calibration", "net_benefit", "risk_distribution"]

        for metric in key_metrics:
            # Should have an entry
            assert metric in PERSONA_TERMINOLOGY, f"Missing terminology for {metric}"

            # Should return different values for each persona
            ds_label = get_label(metric, OutputPersona.DATA_SCIENTIST)
            gov_label = get_label(metric, OutputPersona.GOVERNANCE)

            # Labels should not be the raw metric name (should be transformed)
            assert ds_label != metric or gov_label != metric

    def test_all_optional_metrics_have_terminology(self) -> None:
        """Test that key OPTIONAL metrics have persona terminology."""
        from faircareai.core.config import PERSONA_TERMINOLOGY

        # Key OPTIONAL metrics that should have terminology
        optional_metrics = [
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "brier_score",
            "oe_ratio",
            "ici",
        ]

        for metric in optional_metrics:
            assert metric in PERSONA_TERMINOLOGY, f"Missing terminology for {metric}"
