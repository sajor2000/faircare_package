"""
Tests for FairCareAI fairness decision tree module.

Tests cover:
- Fairness metric options by use case
- Metric descriptions
- Impossibility theorem documentation
- Trade-off comparison
- Decision tree text formatting
"""

from faircareai.core.config import FairnessMetric, UseCaseType
from faircareai.fairness.decision_tree import (
    DECISION_TREE,
    IMPOSSIBILITY_THEOREM,
    compare_metrics_tradeoffs,
    format_decision_tree_text,
    get_fairness_metric_options,
    get_impossibility_warning,
    get_metric_description,
    recommend_fairness_metric,
)


class TestGetFairnessMetricOptions:
    """Tests for get_fairness_metric_options function."""

    def test_intervention_trigger_returns_equalized_odds(self) -> None:
        """Test that INTERVENTION_TRIGGER recommends EQUALIZED_ODDS."""
        result = get_fairness_metric_options(UseCaseType.INTERVENTION_TRIGGER)
        assert result["primary_option"] == FairnessMetric.EQUALIZED_ODDS.value
        assert "use_case" in result
        assert result["use_case"] == UseCaseType.INTERVENTION_TRIGGER.value

    def test_risk_communication_returns_calibration(self) -> None:
        """Test that RISK_COMMUNICATION recommends CALIBRATION."""
        result = get_fairness_metric_options(UseCaseType.RISK_COMMUNICATION)
        assert result["primary_option"] == FairnessMetric.CALIBRATION.value

    def test_resource_allocation_returns_demographic_parity(self) -> None:
        """Test that RESOURCE_ALLOCATION recommends DEMOGRAPHIC_PARITY."""
        result = get_fairness_metric_options(UseCaseType.RESOURCE_ALLOCATION)
        assert result["primary_option"] == FairnessMetric.DEMOGRAPHIC_PARITY.value

    def test_screening_returns_equal_opportunity(self) -> None:
        """Test that SCREENING recommends EQUAL_OPPORTUNITY."""
        result = get_fairness_metric_options(UseCaseType.SCREENING)
        assert result["primary_option"] == FairnessMetric.EQUAL_OPPORTUNITY.value

    def test_diagnosis_support_returns_predictive_parity(self) -> None:
        """Test that DIAGNOSIS_SUPPORT recommends PREDICTIVE_PARITY."""
        result = get_fairness_metric_options(UseCaseType.DIAGNOSIS_SUPPORT)
        assert result["primary_option"] == FairnessMetric.PREDICTIVE_PARITY.value

    def test_none_use_case_returns_error(self) -> None:
        """Test that None use case returns error dict."""
        result = get_fairness_metric_options(None)
        assert "error" in result
        assert "available_use_cases" in result

    def test_includes_impossibility_note(self) -> None:
        """Test that result includes impossibility note."""
        result = get_fairness_metric_options(UseCaseType.SCREENING)
        assert "impossibility_note" in result
        assert "trade-offs" in result["impossibility_note"].lower()

    def test_includes_rationale(self) -> None:
        """Test that result includes rationale."""
        result = get_fairness_metric_options(UseCaseType.SCREENING)
        assert "rationale" in result
        assert len(result["rationale"]) > 0

    def test_includes_alternatives(self) -> None:
        """Test that result includes alternatives."""
        result = get_fairness_metric_options(UseCaseType.INTERVENTION_TRIGGER)
        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)

    def test_includes_clinical_examples(self) -> None:
        """Test that result includes clinical examples."""
        result = get_fairness_metric_options(UseCaseType.RISK_COMMUNICATION)
        assert "clinical_examples" in result
        assert isinstance(result["clinical_examples"], list)
        assert len(result["clinical_examples"]) > 0

    def test_includes_tradeoff_notes(self) -> None:
        """Test that result includes tradeoff notes."""
        result = get_fairness_metric_options(UseCaseType.INTERVENTION_TRIGGER)
        assert "tradeoff_notes" in result
        assert isinstance(result["tradeoff_notes"], list)

    def test_resource_allocation_has_special_considerations(self) -> None:
        """Test that RESOURCE_ALLOCATION includes special considerations."""
        result = get_fairness_metric_options(UseCaseType.RESOURCE_ALLOCATION)
        assert "special_considerations" in result

    def test_available_use_cases_list(self) -> None:
        """Test that None returns list of available use cases."""
        result = get_fairness_metric_options(None)
        assert "available_use_cases" in result
        use_cases = result["available_use_cases"]
        assert len(use_cases) == len(UseCaseType)

    def test_backward_compatibility_alias(self) -> None:
        """Test that recommend_fairness_metric is alias for get_fairness_metric_options."""
        assert recommend_fairness_metric is get_fairness_metric_options


class TestGetMetricDescription:
    """Tests for get_metric_description function."""

    def test_demographic_parity_description(self) -> None:
        """Test DEMOGRAPHIC_PARITY has complete description."""
        result = get_metric_description(FairnessMetric.DEMOGRAPHIC_PARITY)
        assert result["name"] == "Demographic Parity"
        assert "definition" in result
        assert "intuition" in result
        assert "when_appropriate" in result
        assert "when_inappropriate" in result
        assert "measurement" in result

    def test_equalized_odds_description(self) -> None:
        """Test EQUALIZED_ODDS has complete description."""
        result = get_metric_description(FairnessMetric.EQUALIZED_ODDS)
        assert result["name"] == "Equalized Odds"
        assert "TPR" in result["definition"] and "FPR" in result["definition"]

    def test_equal_opportunity_description(self) -> None:
        """Test EQUAL_OPPORTUNITY has complete description."""
        result = get_metric_description(FairnessMetric.EQUAL_OPPORTUNITY)
        assert result["name"] == "Equal Opportunity"
        assert "TPR" in result["definition"]

    def test_calibration_description(self) -> None:
        """Test CALIBRATION has complete description."""
        result = get_metric_description(FairnessMetric.CALIBRATION)
        assert result["name"] == "Calibration"
        assert "probabilities" in result["definition"].lower()

    def test_predictive_parity_description(self) -> None:
        """Test PREDICTIVE_PARITY has complete description."""
        result = get_metric_description(FairnessMetric.PREDICTIVE_PARITY)
        assert result["name"] == "Predictive Parity"
        assert "PPV" in result["definition"]

    def test_also_known_as_field(self) -> None:
        """Test descriptions include also_known_as aliases."""
        result = get_metric_description(FairnessMetric.DEMOGRAPHIC_PARITY)
        assert "also_known_as" in result
        assert isinstance(result["also_known_as"], list)

    def test_when_appropriate_is_list(self) -> None:
        """Test when_appropriate is a list."""
        result = get_metric_description(FairnessMetric.CALIBRATION)
        assert isinstance(result["when_appropriate"], list)
        assert len(result["when_appropriate"]) > 0


class TestCompareMetricsTradeoffs:
    """Tests for compare_metrics_tradeoffs function."""

    def test_known_pair_demographic_equalized(self) -> None:
        """Test trade-off between DEMOGRAPHIC_PARITY and EQUALIZED_ODDS."""
        result = compare_metrics_tradeoffs(
            FairnessMetric.DEMOGRAPHIC_PARITY, FairnessMetric.EQUALIZED_ODDS
        )
        assert "tension" in result
        assert "choose_first_when" in result
        assert "choose_second_when" in result

    def test_known_pair_equal_opportunity_predictive(self) -> None:
        """Test trade-off between EQUAL_OPPORTUNITY and PREDICTIVE_PARITY."""
        result = compare_metrics_tradeoffs(
            FairnessMetric.EQUAL_OPPORTUNITY, FairnessMetric.PREDICTIVE_PARITY
        )
        assert "tension" in result

    def test_known_pair_calibration_demographic(self) -> None:
        """Test trade-off between CALIBRATION and DEMOGRAPHIC_PARITY."""
        result = compare_metrics_tradeoffs(
            FairnessMetric.CALIBRATION, FairnessMetric.DEMOGRAPHIC_PARITY
        )
        assert "tension" in result

    def test_reversed_pair_returns_same_result(self) -> None:
        """Test that reversed pairs return the same information."""
        result1 = compare_metrics_tradeoffs(
            FairnessMetric.DEMOGRAPHIC_PARITY, FairnessMetric.EQUALIZED_ODDS
        )
        result2 = compare_metrics_tradeoffs(
            FairnessMetric.EQUALIZED_ODDS, FairnessMetric.DEMOGRAPHIC_PARITY
        )
        # Should get the same trade-off info
        assert result1 == result2

    def test_unknown_pair_returns_note(self) -> None:
        """Test that unknown pairs return a note."""
        result = compare_metrics_tradeoffs(
            FairnessMetric.CALIBRATION, FairnessMetric.EQUALIZED_ODDS
        )
        assert "note" in result
        assert "impossibility" in result["note"].lower()


class TestGetImpossibilityWarning:
    """Tests for get_impossibility_warning function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = get_impossibility_warning()
        assert isinstance(result, str)

    def test_contains_key_concepts(self) -> None:
        """Test that warning contains key mathematical concepts."""
        result = get_impossibility_warning()
        assert "base rates" in result.lower() or "prevalence" in result.lower()
        assert "mathematically impossible" in result.lower() or "impossible" in result.lower()

    def test_contains_references(self) -> None:
        """Test that warning contains academic references."""
        result = get_impossibility_warning()
        assert "Chouldechova" in result or "Kleinberg" in result

    def test_matches_module_constant(self) -> None:
        """Test that function returns the module constant."""
        result = get_impossibility_warning()
        assert result == IMPOSSIBILITY_THEOREM


class TestFormatDecisionTreeText:
    """Tests for format_decision_tree_text function."""

    def test_returns_multiline_string(self) -> None:
        """Test that function returns a multiline string."""
        result = format_decision_tree_text()
        assert isinstance(result, str)
        assert "\n" in result

    def test_contains_all_use_cases(self) -> None:
        """Test that output contains all use cases."""
        result = format_decision_tree_text()
        for use_case in UseCaseType:
            # Convert enum value to expected format in output
            expected = use_case.value.upper().replace("_", " ")
            assert expected in result

    def test_contains_section_headers(self) -> None:
        """Test that output contains section headers."""
        result = format_decision_tree_text()
        assert "USE CASE:" in result
        assert "PRIMARY OPTION:" in result

    def test_contains_examples(self) -> None:
        """Test that output contains clinical examples."""
        result = format_decision_tree_text()
        assert "EXAMPLES:" in result

    def test_contains_alternatives(self) -> None:
        """Test that output contains alternatives."""
        result = format_decision_tree_text()
        assert "ALTERNATIVES:" in result

    def test_mentions_impossibility_warning(self) -> None:
        """Test that output mentions impossibility warning."""
        result = format_decision_tree_text()
        assert "impossibility" in result.lower()


class TestDecisionTreeConfiguration:
    """Tests for DECISION_TREE configuration."""

    def test_all_use_cases_have_entries(self) -> None:
        """Test that all UseCaseType values have entries in DECISION_TREE."""
        for use_case in UseCaseType:
            assert use_case in DECISION_TREE

    def test_all_entries_have_recommended(self) -> None:
        """Test that all entries have recommended metric."""
        for use_case, config in DECISION_TREE.items():
            assert "recommended" in config
            assert isinstance(config["recommended"], FairnessMetric)

    def test_all_entries_have_rationale(self) -> None:
        """Test that all entries have rationale."""
        for use_case, config in DECISION_TREE.items():
            assert "rationale" in config
            assert len(config["rationale"]) > 0

    def test_all_entries_have_clinical_examples(self) -> None:
        """Test that all entries have clinical examples."""
        for use_case, config in DECISION_TREE.items():
            assert "clinical_examples" in config
            assert isinstance(config["clinical_examples"], list)
            assert len(config["clinical_examples"]) > 0

    def test_alternatives_are_fairness_metrics(self) -> None:
        """Test that all alternatives are FairnessMetric values."""
        for use_case, config in DECISION_TREE.items():
            for alt in config.get("alternatives", []):
                assert isinstance(alt, FairnessMetric)

    def test_contraindicated_are_fairness_metrics(self) -> None:
        """Test that all contraindicated are FairnessMetric values."""
        for use_case, config in DECISION_TREE.items():
            for contra in config.get("contraindicated", []):
                assert isinstance(contra, FairnessMetric)
