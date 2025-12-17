"""
Tests for FairCareAI core audit module.

Tests cover:
- FairCareAudit class initialization and validation
- Data loading from various formats
- Sensitive attribute management
- Intersectional analysis
- Flag generation
- Full audit pipeline
- Legacy AuditResult dataclass
"""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from faircareai.core.audit import AuditResult, FairAudit, FairCareAudit
from faircareai.core.config import FairnessConfig, FairnessMetric, UseCaseType
from faircareai.core.exceptions import ConfigurationError, DataValidationError


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """Create sample data for testing."""
    np.random.seed(42)
    n = 500
    return pl.DataFrame(
        {
            "y_true": np.random.randint(0, 2, n).tolist(),
            "y_prob": np.clip(np.random.random(n), 0.01, 0.99).tolist(),
            "race": np.random.choice(["White", "Black", "Asian", "Hispanic"], n).tolist(),
            "sex": np.random.choice(["Male", "Female"], n).tolist(),
            "age_group": np.random.choice(["18-40", "40-65", "65+"], n).tolist(),
        }
    )


@pytest.fixture
def configured_audit(sample_data: pl.DataFrame) -> FairCareAudit:
    """Create a configured audit with attributes."""
    config = FairnessConfig(
        model_name="Test Model",
        primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
        fairness_justification="Testing purposes",
        use_case_type=UseCaseType.INTERVENTION_TRIGGER,
    )
    audit = FairCareAudit(
        data=sample_data,
        pred_col="y_prob",
        target_col="y_true",
        config=config,
        threshold=0.5,
    )
    audit.add_sensitive_attribute(name="race", column="race", reference="White")
    return audit


class TestFairCareAuditInit:
    """Tests for FairCareAudit initialization."""

    def test_basic_init(self, sample_data: pl.DataFrame) -> None:
        """Test basic initialization with DataFrame."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        assert audit.pred_col == "y_prob"
        assert audit.target_col == "y_true"
        assert audit.threshold == 0.5

    def test_with_config(self, sample_data: pl.DataFrame) -> None:
        """Test initialization with config."""
        config = FairnessConfig(model_name="Test Model")
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
            config=config,
        )
        assert audit.config.model_name == "Test Model"

    def test_custom_threshold(self, sample_data: pl.DataFrame) -> None:
        """Test initialization with custom threshold."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
            threshold=0.3,
        )
        assert audit.threshold == 0.3


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_load_polars_dataframe(self, sample_data: pl.DataFrame) -> None:
        """Test loading Polars DataFrame."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        assert len(audit.df) == len(sample_data)

    def test_load_pandas_dataframe(self, sample_data: pl.DataFrame) -> None:
        """Test loading pandas DataFrame."""
        pandas_df = sample_data.to_pandas()
        audit = FairCareAudit(
            data=pandas_df,
            pred_col="y_prob",
            target_col="y_true",
        )
        assert len(audit.df) == len(sample_data)

    def test_load_parquet_file(self, sample_data: pl.DataFrame) -> None:
        """Test loading parquet file."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            sample_data.write_parquet(f.name)
            audit = FairCareAudit(
                data=f.name,
                pred_col="y_prob",
                target_col="y_true",
            )
            assert len(audit.df) == len(sample_data)
            Path(f.name).unlink()

    def test_load_csv_file(self, sample_data: pl.DataFrame) -> None:
        """Test loading CSV file."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            sample_data.write_csv(f.name)
            audit = FairCareAudit(
                data=f.name,
                pred_col="y_prob",
                target_col="y_true",
            )
            assert len(audit.df) == len(sample_data)
            Path(f.name).unlink()

    def test_unsupported_file_format(self) -> None:
        """Test error on unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            Path(f.name).write_text('{"test": 1}')
            with pytest.raises(DataValidationError):
                FairCareAudit(
                    data=f.name,
                    pred_col="y_prob",
                    target_col="y_true",
                )
            Path(f.name).unlink()

    def test_unsupported_data_type(self) -> None:
        """Test error on unsupported data type."""
        with pytest.raises(TypeError):
            FairCareAudit(
                data=[1, 2, 3],  # type: ignore
                pred_col="y_prob",
                target_col="y_true",
            )


class TestDataValidation:
    """Tests for data validation."""

    def test_missing_pred_column(self) -> None:
        """Test error when prediction column is missing."""
        df = pl.DataFrame({"y_true": [0, 1], "other": [0.5, 0.8]})
        with pytest.raises(DataValidationError) as exc_info:
            FairCareAudit(data=df, pred_col="y_prob", target_col="y_true")
        assert "Missing required columns" in str(exc_info.value)

    def test_missing_target_column(self) -> None:
        """Test error when target column is missing."""
        df = pl.DataFrame({"y_prob": [0.5, 0.8], "other": [0, 1]})
        with pytest.raises(DataValidationError) as exc_info:
            FairCareAudit(data=df, pred_col="y_prob", target_col="y_true")
        assert "Missing required columns" in str(exc_info.value)

    def test_non_numeric_predictions(self) -> None:
        """Test error when predictions are non-numeric."""
        df = pl.DataFrame({"y_prob": ["a", "b"], "y_true": [0, 1]})
        with pytest.raises(DataValidationError) as exc_info:
            FairCareAudit(data=df, pred_col="y_prob", target_col="y_true")
        assert "must be numeric" in str(exc_info.value)

    def test_non_numeric_targets(self) -> None:
        """Test error when targets are non-numeric."""
        df = pl.DataFrame({"y_prob": [0.5, 0.8], "y_true": ["a", "b"]})
        with pytest.raises(DataValidationError) as exc_info:
            FairCareAudit(data=df, pred_col="y_prob", target_col="y_true")
        assert "must be numeric" in str(exc_info.value)

    def test_null_predictions(self) -> None:
        """Test error when predictions contain nulls."""
        df = pl.DataFrame({"y_prob": [0.5, None], "y_true": [0, 1]})
        with pytest.raises(DataValidationError) as exc_info:
            FairCareAudit(data=df, pred_col="y_prob", target_col="y_true")
        assert "null" in str(exc_info.value).lower()

    def test_null_targets(self) -> None:
        """Test error when targets contain nulls."""
        df = pl.DataFrame({"y_prob": [0.5, 0.8], "y_true": [0, None]})
        with pytest.raises(DataValidationError) as exc_info:
            FairCareAudit(data=df, pred_col="y_prob", target_col="y_true")
        assert "null" in str(exc_info.value).lower()

    def test_predictions_out_of_range(self) -> None:
        """Test error when predictions are outside [0, 1]."""
        df = pl.DataFrame({"y_prob": [-0.1, 1.5], "y_true": [0, 1]})
        with pytest.raises(DataValidationError) as exc_info:
            FairCareAudit(data=df, pred_col="y_prob", target_col="y_true")
        assert "[0, 1]" in str(exc_info.value)

    def test_non_binary_targets(self) -> None:
        """Test error when targets are not binary."""
        df = pl.DataFrame({"y_prob": [0.5, 0.8, 0.6], "y_true": [0, 1, 2]})
        with pytest.raises(DataValidationError) as exc_info:
            FairCareAudit(data=df, pred_col="y_prob", target_col="y_true")
        assert "binary" in str(exc_info.value).lower()


class TestSuggestAttributes:
    """Tests for suggest_attributes method."""

    def test_returns_list(self, sample_data: pl.DataFrame) -> None:
        """Test that method returns a list."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        suggestions = audit.suggest_attributes(display=False)
        assert isinstance(suggestions, list)

    def test_display_option(self, sample_data: pl.DataFrame) -> None:
        """Test display option works without error."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        # Should not raise
        audit.suggest_attributes(display=True)


class TestAcceptSuggestedAttributes:
    """Tests for accept_suggested_attributes method."""

    def test_accept_by_index(self, sample_data: pl.DataFrame) -> None:
        """Test accepting suggestion by index."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        suggestions = audit.suggest_attributes(display=False)
        if suggestions:
            audit.accept_suggested_attributes([1])
            assert len(audit.sensitive_attributes) >= 1

    def test_invalid_index(self, sample_data: pl.DataFrame) -> None:
        """Test error on invalid index."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        with pytest.raises(ConfigurationError):
            audit.accept_suggested_attributes([999])

    def test_accept_by_name(self, sample_data: pl.DataFrame) -> None:
        """Test accepting suggestion by name."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        suggestions = audit.suggest_attributes(display=False)
        if suggestions:
            name = suggestions[0]["suggested_name"]
            audit.accept_suggested_attributes([name])
            assert len(audit.sensitive_attributes) >= 1

    def test_invalid_name(self, sample_data: pl.DataFrame) -> None:
        """Test error on invalid name."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        with pytest.raises(ConfigurationError):
            audit.accept_suggested_attributes(["nonexistent_attr"])


class TestAddSensitiveAttribute:
    """Tests for add_sensitive_attribute method."""

    def test_add_attribute(self, sample_data: pl.DataFrame) -> None:
        """Test adding a sensitive attribute."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        audit.add_sensitive_attribute(name="race", column="race", reference="White")
        assert len(audit.sensitive_attributes) == 1
        assert audit.sensitive_attributes[0].name == "race"

    def test_method_chaining(self, sample_data: pl.DataFrame) -> None:
        """Test that method returns self for chaining."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        result = audit.add_sensitive_attribute(name="race", column="race")
        assert result is audit

    def test_invalid_column(self, sample_data: pl.DataFrame) -> None:
        """Test error on invalid column."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        with pytest.raises(DataValidationError):
            audit.add_sensitive_attribute(name="invalid", column="nonexistent")


class TestAddIntersection:
    """Tests for add_intersection method."""

    def test_add_intersection(self, sample_data: pl.DataFrame) -> None:
        """Test adding intersection."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        audit.add_sensitive_attribute(name="race", column="race")
        audit.add_sensitive_attribute(name="sex", column="sex")
        audit.add_intersection(["race", "sex"])
        assert len(audit.intersections) == 1

    def test_invalid_intersection_attribute(self, sample_data: pl.DataFrame) -> None:
        """Test error on invalid intersection attribute."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        with pytest.raises(ConfigurationError):
            audit.add_intersection(["nonexistent"])


class TestSuggestFairnessMetric:
    """Tests for suggest_fairness_metric method."""

    def test_returns_dict(self, sample_data: pl.DataFrame) -> None:
        """Test that method returns a dict."""
        config = FairnessConfig(
            model_name="Test",
            use_case_type=UseCaseType.INTERVENTION_TRIGGER,
        )
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
            config=config,
        )
        result = audit.suggest_fairness_metric()
        assert isinstance(result, dict)


class TestRun:
    """Tests for run method."""

    def test_run_basic(self, configured_audit: FairCareAudit) -> None:
        """Test basic audit run."""
        results = configured_audit.run(bootstrap_ci=False)
        assert results is not None
        assert results.config is not None

    def test_run_with_bootstrap(self, configured_audit: FairCareAudit) -> None:
        """Test audit run with bootstrap CI."""
        results = configured_audit.run(bootstrap_ci=True, n_bootstrap=100)
        assert results is not None

    def test_run_without_attributes(self, sample_data: pl.DataFrame) -> None:
        """Test error when running without sensitive attributes."""
        config = FairnessConfig(
            model_name="Test",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            fairness_justification="Test",
        )
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
            config=config,
        )
        with pytest.raises(ConfigurationError):
            audit.run()

    def test_run_without_config(self, sample_data: pl.DataFrame) -> None:
        """Test error when running without proper config."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        audit.add_sensitive_attribute(name="race", column="race")
        with pytest.raises(ConfigurationError):
            audit.run()

    def test_run_with_intersections(self, sample_data: pl.DataFrame) -> None:
        """Test audit run with intersections."""
        config = FairnessConfig(
            model_name="Test",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            fairness_justification="Test",
        )
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
            config=config,
        )
        audit.add_sensitive_attribute(name="race", column="race", reference="White")
        audit.add_sensitive_attribute(name="sex", column="sex", reference="Male")
        audit.add_intersection(["race", "sex"])
        results = audit.run(bootstrap_ci=False)
        assert "race x sex" in results.intersectional


class TestGetAttrColumn:
    """Tests for _get_attr_column method."""

    def test_get_existing_attr(self, sample_data: pl.DataFrame) -> None:
        """Test getting column for existing attribute."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        audit.add_sensitive_attribute(name="race", column="race")
        col = audit._get_attr_column("race")
        assert col == "race"

    def test_get_nonexistent_attr(self, sample_data: pl.DataFrame) -> None:
        """Test error for nonexistent attribute."""
        audit = FairCareAudit(
            data=sample_data,
            pred_col="y_prob",
            target_col="y_true",
        )
        with pytest.raises(ConfigurationError):
            audit._get_attr_column("nonexistent")


class TestFlagGeneration:
    """Tests for flag generation methods."""

    def test_check_subgroup_sizes(self, configured_audit: FairCareAudit) -> None:
        """Test subgroup size checking."""
        results = configured_audit.run(bootstrap_ci=False)
        # Should have generated flags
        assert isinstance(results.flags, list)

    def test_check_fairness_violations(self, configured_audit: FairCareAudit) -> None:
        """Test fairness violation checking."""
        results = configured_audit.run(bootstrap_ci=False)
        fairness_flags = [f for f in results.flags if f.get("category") == "fairness"]
        # May or may not have fairness flags depending on random data
        assert isinstance(fairness_flags, list)


class TestGenerateRecommendation:
    """Tests for _generate_recommendation method."""

    def test_recommendation_structure(self, configured_audit: FairCareAudit) -> None:
        """Test recommendation dict structure."""
        results = configured_audit.run(bootstrap_ci=False)
        rec = results.governance_recommendation
        assert "methodology" in rec
        assert "outside_threshold_count" in rec
        assert "near_threshold_count" in rec
        assert "within_threshold_count" in rec


class TestLegacyAuditResult:
    """Tests for legacy AuditResult dataclass."""

    def test_dataclass_construction(self) -> None:
        """Test dataclass construction."""
        result = AuditResult(
            model_name="Test Model",
            audit_date="2025-01-01",
            n_samples=1000,
            threshold=0.5,
            group_col="race",
            metrics_df=pl.DataFrame({"group": ["A"], "tpr": [0.8]}),
            disparities_df=pl.DataFrame({"group": ["A"], "diff": [0.1]}),
            pass_count=5,
            warn_count=2,
            fail_count=1,
            worst_disparity=("Group B", "tpr", -0.15),
        )
        assert result.model_name == "Test Model"
        assert result.n_samples == 1000

    def test_summary_method(self) -> None:
        """Test summary method."""
        result = AuditResult(
            model_name="Test Model",
            audit_date="2025-01-01",
            n_samples=1000,
            threshold=0.5,
            group_col="race",
            metrics_df=pl.DataFrame({"group": ["A"], "tpr": [0.8]}),
            disparities_df=pl.DataFrame({"group": ["A"], "diff": [0.1]}),
            pass_count=5,
            warn_count=2,
            fail_count=1,
            worst_disparity=("Group B", "tpr", -0.15),
        )
        summary = result.summary()
        assert "Test Model" in summary
        assert "1,000" in summary
        assert "Group B" in summary

    def test_summary_no_disparity(self) -> None:
        """Test summary method without worst disparity."""
        result = AuditResult(
            model_name="Test Model",
            audit_date="2025-01-01",
            n_samples=1000,
            threshold=0.5,
            group_col="race",
            metrics_df=pl.DataFrame({"group": ["A"], "tpr": [0.8]}),
            disparities_df=pl.DataFrame({"group": ["A"], "diff": [0.1]}),
            pass_count=10,
            warn_count=0,
            fail_count=0,
            worst_disparity=None,
        )
        summary = result.summary()
        assert "Test Model" in summary


class TestFairAuditAlias:
    """Tests for FairAudit backward compatibility alias."""

    def test_alias_exists(self) -> None:
        """Test that FairAudit alias exists."""
        assert FairAudit is FairCareAudit
