"""
FairCareAI - Main Orchestration Class

Coordinates the fairness auditing pipeline from data to reports.

This package computes and presents fairness metrics per Van Calster et al. (2025)
methodology. Healthcare organizations interpret results based on their clinical
context, organizational values, and governance frameworks.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl

from faircareai.core.config import (
    FairnessConfig,
    SensitiveAttribute,
)
from faircareai.core.exceptions import (
    ConfigurationError,
    DataValidationError,
)
from faircareai.core.logging import get_logger
from faircareai.core.results import AuditResults
from faircareai.data.sensitive_attrs import (
    display_suggestions,
    get_reference_group,
    suggest_sensitive_attributes,
    validate_attribute,
)

logger = get_logger(__name__)


class FairCareAudit:
    """
    Main class for conducting fairness audits on ML models.

    Computes fairness metrics per Van Calster et al. (2025) methodology.
    Designed for data scientists to present fairness analysis results
    to governance stakeholders.

    Data Requirements:
        Your data must contain:

        1. **Prediction Column** (pred_col): Model-generated probabilities
           - Values must be in range [0.0, 1.0]
           - Example: risk_score, predicted_prob, mortality_risk

        2. **Target Column** (target_col): Actual binary outcomes
           - Values must be exactly 0 or 1
           - Example: readmit_30d, mortality, los_gt_7

        3. **Sensitive Attribute Columns**: Demographics for fairness analysis
           - Categorical columns like race, sex, age_group, insurance
           - Auto-detected or manually specified via add_sensitive_attribute()

    Output Personas:
        FairCareAudit supports two output personas for different audiences:

        - **Data Scientist** (default): Full technical output with all metrics,
          confidence intervals, detailed tables, and ~15-20 figures.

        - **Governance**: Streamlined 3-5 page output with 8 key figures,
          plain language summaries, and clear decision blocks.

        Select persona when exporting:
        >>> results.to_pdf("full_report.pdf")  # Data scientist (default)
        >>> results.to_pdf("governance.pdf", persona="governance")

    Example:
        >>> from faircareai import FairCareAudit, FairnessConfig
        >>> from faircareai.core.config import FairnessMetric, UseCaseType
        >>>
        >>> # Load your predictions
        >>> audit = FairCareAudit(
        ...     data="predictions.parquet",
        ...     pred_col="risk_score",
        ...     target_col="readmit_30d"
        ... )
        >>>
        >>> # See suggested sensitive attributes
        >>> audit.suggest_attributes()
        >>>
        >>> # Accept suggestions (1-indexed)
        >>> audit.accept_suggested_attributes([1, 2])  # race, sex
        >>>
        >>> # Or manually add sensitive attributes
        >>> audit.add_sensitive_attribute(
        ...     name="race",
        ...     column="patient_race",
        ...     reference="White",
        ...     clinical_justification="Historical disparities in care"
        ... )
        >>>
        >>> # Get fairness metric recommendation
        >>> audit.suggest_fairness_metric()
        >>>
        >>> # Configure and run
        >>> config = FairnessConfig(
        ...     model_name="Readmission Risk v2",
        ...     primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
        ...     fairness_justification="Model triggers care management...",
        ...     use_case_type=UseCaseType.INTERVENTION_TRIGGER
        ... )
        >>> audit.config = config
        >>> results = audit.run()
        >>>
        >>> # Export for data scientist (full technical output)
        >>> results.to_html("full_report.html")
        >>> results.to_pdf("full_report.pdf")
        >>>
        >>> # Export for governance (streamlined 3-5 page output)
        >>> results.to_governance_pdf("governance.pdf")
        >>> results.to_html("governance.html", persona="governance")
    """

    def __init__(
        self,
        data: pl.DataFrame | str | Path,
        pred_col: str,
        target_col: str,
        config: FairnessConfig | None = None,
        threshold: float = 0.5,
    ):
        """
        Initialize a fairness audit.

        Args:
            data: Model predictions data. Accepts:
                - Polars DataFrame (preferred)
                - pandas DataFrame (auto-converted to Polars)
                - Path to .parquet or .csv file

            pred_col: Column name containing model predictions as probabilities.
                Must be numeric values in the range [0.0, 1.0].

                Examples of valid column names:
                - "risk_score" - Generic risk score
                - "predicted_prob" - Predicted probability
                - "readmission_prob" - Readmission probability
                - "mortality_risk" - Mortality risk score

            target_col: Column name containing actual outcomes as binary values.
                Must be numeric values of exactly 0 or 1.

                Examples of valid column names:
                - "readmit_30d" - 30-day readmission outcome
                - "mortality" - Mortality outcome
                - "outcome" - Generic outcome
                - "los_gt_7" - Length of stay > 7 days

            config: FairnessConfig object with audit settings. Can be set later
                via `audit.config = FairnessConfig(...)`. Required fields:
                - model_name: Name of the model being audited
                - primary_fairness_metric: Selected fairness metric (use suggest_fairness_metric())
                - fairness_justification: Rationale for metric selection

            threshold: Decision threshold for converting probabilities to binary
                predictions. Default is 0.5. Adjust based on clinical context.

        Example:
            >>> # From parquet file
            >>> audit = FairCareAudit(
            ...     data="predictions.parquet",
            ...     pred_col="risk_score",
            ...     target_col="readmit_30d"
            ... )

            >>> # From pandas DataFrame
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     "risk_score": [0.2, 0.7, 0.4, 0.9],
            ...     "readmit_30d": [0, 1, 0, 1],
            ...     "race": ["White", "Black", "Hispanic", "White"],
            ...     "sex": ["M", "F", "M", "F"]
            ... })
            >>> audit = FairCareAudit(df, "risk_score", "readmit_30d")

        Note:
            After initialization, use suggest_attributes() to see auto-detected
            sensitive attributes, then accept_suggested_attributes() or
            add_sensitive_attribute() to define which demographics to analyze.
        """
        self.df = self._load_data(data)
        self.pred_col = pred_col
        self.target_col = target_col
        self.threshold = threshold
        self.config = config or FairnessConfig(model_name="Unnamed Model")

        # Store for visualization access
        self.y_true_col = target_col
        self.y_prob_col = pred_col

        self.sensitive_attributes: list[SensitiveAttribute] = []
        self.intersections: list[list[str]] = []

        self._validate_data()

        # Auto-detect suggested attributes
        self._suggestions = suggest_sensitive_attributes(self.df)

    def __getstate__(self) -> dict:
        """Return picklable state (exclude logger for Windows multiprocessing)."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)

    def _load_data(self, data: pl.DataFrame | str | Path | Any) -> pl.DataFrame:
        """Load data from DataFrame or file path.

        Supports:
            - Polars DataFrame (preferred, used internally)
            - pandas DataFrame (auto-converted to Polars)
            - Path to parquet or csv file

        Args:
            data: Input data as DataFrame or file path.

        Returns:
            Polars DataFrame ready for analysis.

        Raises:
            TypeError: If data type is not supported.
            ValueError: If file format is not supported.
        """
        # Check for Polars DataFrame first (most common case)
        if isinstance(data, pl.DataFrame):
            return data

        # Auto-convert pandas DataFrame to Polars
        try:
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                logger.info(
                    "Converting pandas DataFrame to Polars. "
                    "FairCareAI uses Polars internally for performance."
                )
                return pl.from_pandas(data)
        except ImportError:
            pass  # pandas not installed, continue to other checks

        # Handle file paths
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix == ".parquet":
                return pl.read_parquet(path)
            elif path.suffix == ".csv":
                return pl.read_csv(path)
            else:
                raise DataValidationError(
                    f"Unsupported file format: {path.suffix}. Supported formats: .parquet, .csv"
                )

        # Provide helpful error message
        type_name = type(data).__name__
        raise TypeError(
            f"Expected Polars DataFrame, pandas DataFrame, or file path, got {type_name}. "
            f"Ensure your model predictions are in a supported DataFrame format."
        )

    def _validate_data(self) -> None:
        """Validate data for binary classification fairness analysis.

        Ensures compatibility with any binary ML healthcare model producing
        probability scores in [0, 1].

        Raises:
            DataValidationError: If data fails validation checks.
        """
        from faircareai.core.exceptions import DataValidationError

        # 1. Check required columns exist
        required = [self.pred_col, self.target_col]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")

        # 2. Validate numeric types
        if not self.df[self.pred_col].dtype.is_numeric():
            raise DataValidationError(
                f"Prediction column '{self.pred_col}' must be numeric, "
                f"got {self.df[self.pred_col].dtype}. "
                f"Ensure your model outputs probability scores.",
                column=self.pred_col,
            )
        if not self.df[self.target_col].dtype.is_numeric():
            raise DataValidationError(
                f"Target column '{self.target_col}' must be numeric, "
                f"got {self.df[self.target_col].dtype}. "
                f"Binary outcomes should be encoded as 0/1.",
                column=self.target_col,
            )

        # 3. Check for null/NaN values
        pred_nulls = self.df[self.pred_col].null_count()
        if pred_nulls > 0:
            raise DataValidationError(
                f"Predictions contain {pred_nulls} null/NaN values. "
                f"Remove or impute missing predictions before analysis.",
                column=self.pred_col,
            )
        target_nulls = self.df[self.target_col].null_count()
        if target_nulls > 0:
            raise DataValidationError(
                f"Targets contain {target_nulls} null/NaN values. "
                f"Remove rows with missing outcomes before analysis.",
                column=self.target_col,
            )

        # 4. Validate predictions are probabilities [0, 1]
        pred_min_value = self.df[self.pred_col].min()
        pred_max_value = self.df[self.pred_col].max()

        if not isinstance(pred_min_value, (int, float, Decimal)) or not isinstance(
            pred_max_value, (int, float, Decimal)
        ):
            raise DataValidationError(
                f"Prediction column '{self.pred_col}' must contain numeric probability values.",
                column=self.pred_col,
            )

        pred_min = float(pred_min_value)
        pred_max = float(pred_max_value)
        if pred_min < 0 or pred_max > 1:
            raise DataValidationError(
                f"Predictions must be probabilities in [0, 1], "
                f"got range [{pred_min:.4f}, {pred_max:.4f}]. "
                f"Apply sigmoid/softmax if using raw logits.",
                column=self.pred_col,
            )

        # 5. Validate targets are binary (0/1 only)
        target_values = self.df[self.target_col].unique().to_list()
        valid_values = {0, 1}
        invalid = [v for v in target_values if v not in valid_values]
        if invalid:
            raise DataValidationError(
                f"Targets must be binary (0/1), found values: {target_values}. "
                f"FairCareAI supports binary classification only.",
                column=self.target_col,
            )

        # 6. Sample size warning (statistical reliability)
        n = len(self.df)
        if n < 30:
            logger.warning(
                f"Small dataset (n={n}). Fairness metrics may be unreliable. "
                f"Consider collecting more data for robust analysis."
            )

    def suggest_attributes(self, display: bool = True) -> list[dict]:
        """
        Show suggested sensitive attributes based on detected columns.

        FairCareAI scans your data for common healthcare demographic columns
        and suggests which to use for fairness analysis.

        Args:
            display: Print formatted suggestions to console.

        Returns:
            List of suggestion dicts. User must explicitly accept.
        """
        if display:
            logger.info(
                "Suggested sensitive attributes:\n%s", display_suggestions(self._suggestions)
            )
        return self._suggestions

    def accept_suggested_attributes(
        self,
        selections: list[int | str],
        modify: dict | None = None,
    ) -> "FairCareAudit":
        """
        Accept suggested sensitive attributes.

        Args:
            selections: Indices (1-based) or names of suggestions to accept.
            modify: Overrides for accepted suggestions, e.g.
                    {"race": {"reference": "Black"}}

        Returns:
            self: For method chaining.
        """
        modify = modify or {}

        for sel in selections:
            # Find the suggestion
            if isinstance(sel, int):
                if 1 <= sel <= len(self._suggestions):
                    suggestion = self._suggestions[sel - 1]
                else:
                    raise ConfigurationError(
                        "selection", f"Invalid index: {sel}. Valid: 1-{len(self._suggestions)}"
                    )
            else:
                matches = [s for s in self._suggestions if s["suggested_name"] == sel]
                if not matches:
                    raise ConfigurationError("selection", f"No suggestion named: {sel}")
                suggestion = matches[0]

            # Apply any modifications
            overrides = modify.get(suggestion["suggested_name"], {})

            self.add_sensitive_attribute(
                name=suggestion["suggested_name"],
                column=overrides.get("column", suggestion["detected_column"]),
                reference=overrides.get("reference", suggestion["suggested_reference"]),
                categories=overrides.get("categories"),
                clinical_justification=overrides.get(
                    "clinical_justification",
                    suggestion["clinical_justification"],
                ),
            )

            suggestion["accepted"] = True

        return self

    def add_sensitive_attribute(
        self,
        name: str,
        column: str | None = None,
        reference: str | None = None,
        categories: list[str] | None = None,
        clinical_justification: str | None = None,
    ) -> "FairCareAudit":
        """
        Add a sensitive attribute for fairness analysis.

        Sensitive attributes are demographic or social characteristics used to
        assess fairness across patient subgroups. Common examples include race,
        sex, age group, insurance type, and language.

        Args:
            name: Display name for the attribute in reports and visualizations.
                Use a clear, descriptive name (e.g., "race", "sex", "insurance").

            column: Column name in your data containing this attribute.
                Defaults to `name` if not specified. Use this when your column
                name differs from the display name.

            reference: Reference group for disparity calculations. Disparities
                are computed as (group_metric / reference_metric). If not specified,
                defaults to the largest group or standard clinical reference:
                - Race: "White"
                - Sex: "Male"
                - Insurance: "Commercial"

            categories: List of expected category values. If specified, validates
                that all values in the data match expected categories.

            clinical_justification: CHAI-required documentation of why this
                attribute is relevant for fairness analysis in your clinical context.

        Returns:
            self: For method chaining.

        Examples:
            >>> # Race/Ethnicity - common healthcare demographic
            >>> audit.add_sensitive_attribute(
            ...     name="race",
            ...     column="patient_race",  # Your column name
            ...     reference="White",
            ...     clinical_justification="Historical disparities in healthcare access and outcomes"
            ... )

            >>> # Sex/Gender
            >>> audit.add_sensitive_attribute(
            ...     name="sex",
            ...     column="birth_sex",
            ...     reference="Male",
            ...     categories=["Male", "Female"],
            ...     clinical_justification="Biological and social factors affecting disease presentation"
            ... )

            >>> # Age Group
            >>> audit.add_sensitive_attribute(
            ...     name="age_group",
            ...     column="age_category",
            ...     reference="45-64",  # Working-age adults as reference
            ...     categories=["18-44", "45-64", "65-74", "75+"],
            ...     clinical_justification="Age-related differences in treatment response and risk"
            ... )

            >>> # Insurance/Payer Type
            >>> audit.add_sensitive_attribute(
            ...     name="insurance",
            ...     column="payer_type",
            ...     reference="Commercial",
            ...     categories=["Commercial", "Medicare", "Medicaid", "Uninsured"],
            ...     clinical_justification="Insurance status affects access to preventive care"
            ... )

            >>> # Primary Language
            >>> audit.add_sensitive_attribute(
            ...     name="language",
            ...     column="preferred_language",
            ...     reference="English",
            ...     clinical_justification="Language barriers may affect care coordination"
            ... )

            >>> # Custom attribute (e.g., geographic)
            >>> audit.add_sensitive_attribute(
            ...     name="rural_urban",
            ...     column="geographic_type",
            ...     reference="Urban",
            ...     categories=["Urban", "Suburban", "Rural"],
            ...     clinical_justification="Rural patients may face transportation and access barriers"
            ... )

        Note:
            FairCareAI auto-detects common sensitive attribute columns.
            Use suggest_attributes() to see auto-detected columns, then
            accept_suggested_attributes() for quick setup, or use this method
            for full control over attribute configuration.
        """
        col = column or name

        # Validate
        issues = validate_attribute(self.df, name, col, reference, categories)
        if any("not found" in issue for issue in issues):
            raise DataValidationError(f"Attribute validation failed: {issues}", column=col)

        # Warn but don't fail for other issues
        for issue in issues:
            if "not found" not in issue:
                logger.warning("Attribute validation: %s", issue)

        # Determine reference group if not specified
        if reference is None:
            reference = get_reference_group(self.df, col, None)

        attr = SensitiveAttribute(
            name=name,
            column=col,
            reference=reference,
            categories=categories,
            clinical_justification=clinical_justification,
        )

        self.sensitive_attributes.append(attr)
        return self

    def add_intersection(self, attributes: list[str]) -> "FairCareAudit":
        """
        Add intersectional analysis (e.g., race x sex).

        Args:
            attributes: List of attribute names to intersect.

        Returns:
            self: For method chaining.
        """
        # Validate attributes exist
        attr_names = [a.name for a in self.sensitive_attributes]
        missing = [a for a in attributes if a not in attr_names]
        if missing:
            raise ConfigurationError(
                "intersectional_attributes",
                f"Attributes not registered: {missing}. "
                f"First call add_sensitive_attribute() or accept_suggested_attributes().",
            )

        self.intersections.append(attributes)
        return self

    def suggest_fairness_metric(self) -> dict:
        """
        Get fairness metric options based on use case.

        Returns options from healthcare AI fairness literature.
        Your organization selects based on clinical context.

        Returns:
            Dict with metric options and context.
        """
        from faircareai.fairness.decision_tree import get_fairness_metric_options

        return get_fairness_metric_options(self.config.use_case_type)

    def run(
        self,
        bootstrap_ci: bool = True,
        n_bootstrap: int = 1000,
    ) -> AuditResults:
        """
        Execute the fairness audit.

        Computes all fairness and performance metrics based on the configured
        sensitive attributes and fairness criteria. Results can be exported
        in two personas: full technical output for data scientists, or
        streamlined output for governance committees.

        Args:
            bootstrap_ci: Calculate bootstrap confidence intervals for all
                metrics. Recommended for statistical rigor. Default: True.
            n_bootstrap: Number of bootstrap iterations for confidence
                intervals. Higher values increase precision but take longer.
                Default: 1000.

        Returns:
            AuditResults object containing all computed metrics and methods for:
            - Visualization: plot_executive_summary(), plot_forest_plot(), etc.
            - Export: to_html(), to_pdf(), to_pptx()
            - Governance export: to_governance_html(), to_governance_pdf()

        Raises:
            ConfigurationError: If required config fields are missing
                (primary_fairness_metric, fairness_justification).
            ConfigurationError: If no sensitive attributes have been added.

        Example:
            >>> # Run audit with confidence intervals
            >>> results = audit.run(bootstrap_ci=True, n_bootstrap=1000)
            >>>
            >>> # View executive summary
            >>> results.plot_executive_summary()
            >>>
            >>> # Export full technical report (Data Scientist persona)
            >>> results.to_pdf("full_report.pdf")
            >>>
            >>> # Export streamlined report (Governance persona)
            >>> results.to_governance_pdf("governance.pdf")
            >>> # Or equivalently:
            >>> results.to_pdf("governance.pdf", persona="governance")
        """
        # Validate config
        config_issues = self.config.validate()
        errors = [i for i in config_issues if i.startswith("ERROR")]
        if errors:
            raise ConfigurationError(
                "fairness_config",
                "Configuration errors:\n" + "\n".join(errors) + "\n\n"
                "Use audit.suggest_fairness_metric() for recommendations, then set:\n"
                "  config.primary_fairness_metric = FairnessMetric.EQUALIZED_ODDS\n"
                "  config.fairness_justification = 'Your justification here'",
            )

        if not self.sensitive_attributes:
            raise ConfigurationError(
                "sensitive_attributes",
                "At least one sensitive attribute required.\n"
                "Use audit.suggest_attributes() to see suggestions, then:\n"
                "  audit.accept_suggested_attributes([1, 2])",
            )

        # Warn about non-errors
        warnings = [i for i in config_issues if i.startswith("WARNING")]
        for w in warnings:
            logger.warning(w)

        results = AuditResults(config=self.config, threshold=self.threshold)

        # Section 1: Descriptive Statistics
        from faircareai.metrics.descriptive import compute_cohort_summary

        # Build sensitive attributes dict for metrics functions
        results.descriptive_stats = compute_cohort_summary(
            df=self.df,
            y_true_col=self.target_col,
            y_prob_col=self.pred_col,
            sensitive_attrs={
                a.name: {"column": a.column, "reference": a.reference}
                for a in self.sensitive_attributes
            },
        )

        # Section 2: Overall Performance
        from faircareai.metrics.performance import compute_overall_performance

        y_true = self.df[self.target_col].to_numpy()
        y_prob = self.df[self.pred_col].to_numpy()

        results.overall_performance = compute_overall_performance(
            y_true=y_true,
            y_prob=y_prob,
            threshold=self.threshold,
            bootstrap_ci=bootstrap_ci,
            n_bootstrap=n_bootstrap,
        )

        # Section 3: Subgroup Performance
        from faircareai.metrics.subgroup import compute_subgroup_metrics

        results.subgroup_performance = {}
        for attr in self.sensitive_attributes:
            results.subgroup_performance[attr.name] = compute_subgroup_metrics(
                df=self.df,
                y_prob_col=self.pred_col,
                y_true_col=self.target_col,
                group_col=attr.column,
                threshold=self.threshold,
                reference=attr.reference,
                bootstrap_ci=bootstrap_ci,
                n_bootstrap=n_bootstrap,
            )

        # Section 4: Fairness Metrics
        from faircareai.metrics.fairness import compute_fairness_metrics

        results.fairness_metrics = {}
        for attr in self.sensitive_attributes:
            results.fairness_metrics[attr.name] = compute_fairness_metrics(
                df=self.df,
                y_prob_col=self.pred_col,
                y_true_col=self.target_col,
                group_col=attr.column,
                threshold=self.threshold,
                reference=attr.reference,
            )

        # Section 5: Intersectional Analysis
        from faircareai.metrics.subgroup import compute_intersectional

        results.intersectional = {}
        for intersection in self.intersections:
            key = " x ".join(intersection)
            cols = [self._get_attr_column(a) for a in intersection]
            min_n_val = self.config.get_threshold("min_subgroup_n", 100)
            results.intersectional[key] = compute_intersectional(
                df=self.df,
                y_prob_col=self.pred_col,
                y_true_col=self.target_col,
                group_cols=cols,
                threshold=self.threshold,
                min_n=int(min_n_val) if min_n_val is not None else 100,
            )

        # Section 6: Generate Flags
        results.flags = self._generate_flags(results)

        # Section 7: Governance Recommendation
        results.governance_recommendation = self._generate_recommendation(results)

        # Store reference for visualization
        results._audit = self

        return results

    def _get_attr_column(self, name: str) -> str:
        """Get column name for attribute."""
        for attr in self.sensitive_attributes:
            if attr.name == name:
                return attr.column
        raise ConfigurationError("attribute_name", f"Attribute not found: {name}")

    def _generate_flags(self, results: AuditResults) -> list[dict]:
        """Generate warning/error flags based on thresholds.

        Consolidates all threshold checks across sample sizes, fairness
        metrics, and data quality into a unified flag list.
        """
        flags: list[dict] = []
        thresholds = self.config.thresholds

        # Check each category separately for maintainability
        flags.extend(self._check_subgroup_sizes(results, thresholds))
        flags.extend(self._check_fairness_violations(results, thresholds))
        flags.extend(self._check_data_quality(results, thresholds))

        return flags

    def _check_subgroup_sizes(self, results: AuditResults, thresholds: dict) -> list[dict]:
        """Check subgroup sample sizes against minimum threshold.

        Flags groups with sample sizes below the configured minimum,
        which may lead to unstable statistical estimates.
        """
        flags = []
        min_n = thresholds.get("min_subgroup_n", 100)

        for attr_name, subgroup_data in results.subgroup_performance.items():
            if not isinstance(subgroup_data, dict):
                continue

            groups_data = subgroup_data.get("groups", subgroup_data)
            for group, metrics in groups_data.items():
                if group in ("reference", "attribute", "threshold"):
                    continue
                if not isinstance(metrics, dict):
                    continue

                n = metrics.get("n", 0)
                if n < min_n:
                    flags.append(
                        {
                            "severity": "warning",
                            "category": "sample_size",
                            "attribute": attr_name,
                            "group": group,
                            "message": f"Subgroup n={n} < {min_n}",
                            "details": (
                                f"Small sample size may lead to unstable estimates "
                                f"for {attr_name}:{group}"
                            ),
                            "chai_criteria": "AC1.CR82",
                        }
                    )

        return flags

    def _check_fairness_violations(self, results: AuditResults, thresholds: dict) -> list[dict]:
        """Check fairness metrics against configured thresholds.

        Evaluates demographic parity ratios (EEOC 80% rule) and
        equalized odds differences for each sensitive attribute.
        """
        flags = []
        dp_range = thresholds.get("demographic_parity_ratio", (0.8, 1.25))
        eo_threshold = thresholds.get("equalized_odds_diff", 0.1)

        for attr_name, fairness_data in results.fairness_metrics.items():
            if not isinstance(fairness_data, dict):
                continue

            # Demographic parity (EEOC 80% rule)
            dp_ratios = fairness_data.get("demographic_parity_ratio", {})
            if isinstance(dp_ratios, dict):
                for group, ratio in dp_ratios.items():
                    if ratio is not None and (ratio < dp_range[0] or ratio > dp_range[1]):
                        flags.append(
                            {
                                "severity": "warning",
                                "category": "fairness",
                                "metric": "demographic_parity",
                                "attribute": attr_name,
                                "group": group,
                                "value": ratio,
                                "threshold": dp_range,
                                "message": (
                                    f"Demographic parity ratio {ratio:.2f} "
                                    f"outside [{dp_range[0]}, {dp_range[1]}]"
                                ),
                                "details": (
                                    f"Selection rates differ significantly between "
                                    f"{group} and reference"
                                ),
                                "chai_criteria": "AC1.CR92",
                            }
                        )

            # Equalized odds (TPR/FPR parity)
            eo_diffs = fairness_data.get("equalized_odds_diff", {})
            if isinstance(eo_diffs, dict):
                for group, diff in eo_diffs.items():
                    if diff is not None and abs(diff) > eo_threshold:
                        flags.append(
                            {
                                "severity": "warning",
                                "category": "fairness",
                                "metric": "equalized_odds",
                                "attribute": attr_name,
                                "group": group,
                                "value": diff,
                                "threshold": eo_threshold,
                                "message": f"Equalized odds difference {diff:.3f} > {eo_threshold}",
                                "details": (
                                    f"TPR/FPR differs significantly between {group} and reference"
                                ),
                                "chai_criteria": "AC1.CR92",
                            }
                        )

        return flags

    def _check_data_quality(self, results: AuditResults, thresholds: dict) -> list[dict]:
        """Check data quality metrics against thresholds.

        Flags attributes with high missing data rates that may
        bias fairness estimates.
        """
        flags = []
        max_missing = thresholds.get("max_missing_rate", 0.10)

        attr_distributions = results.descriptive_stats.get("attribute_distributions", {})
        for attr_name, attr_dist in attr_distributions.items():
            if not isinstance(attr_dist, dict):
                continue

            missing_rate = attr_dist.get("pct_missing", 0)
            if missing_rate > max_missing:
                flags.append(
                    {
                        "severity": "warning",
                        "category": "data_quality",
                        "attribute": attr_name,
                        "value": missing_rate,
                        "threshold": max_missing,
                        "message": f"Missing rate {missing_rate:.1%} > {max_missing:.0%}",
                        "details": (
                            f"High missing data for {attr_name} may bias fairness estimates"
                        ),
                        "chai_criteria": "AC1.CR68",
                    }
                )

        return flags

    def _generate_recommendation(self, results: AuditResults) -> dict:
        """
        Compute summary statistics for the audit results.

        Counts metrics by threshold status for presentation to governance.
        Interpretation of these counts rests with your organization.
        """
        errors = [f for f in results.flags if f.get("severity") == "error"]
        warnings = [f for f in results.flags if f.get("severity") == "warning"]

        # Count total checks performed (attributes * check_types)
        # Each attribute has: sample_size, demographic_parity, equalized_odds checks
        n_checks_per_attr = 3
        total_checks = len(results.subgroup_performance) * n_checks_per_attr
        n_pass = max(0, total_checks - len(warnings) - len(errors))

        # Determine status based on error/warning counts
        if len(errors) > 0:
            status = "REVIEW"
            advisory = "Issues detected that may warrant review before deployment."
        elif len(warnings) > 0:
            status = "CONDITIONAL"
            advisory = "Some metrics near threshold — consider documented mitigation."
        else:
            status = "READY"
            advisory = "No significant issues detected at current thresholds."

        # CHAI governance disclaimer
        disclaimer = (
            "This is CHAI-grounded guidance. Final deployment decisions "
            "rest with clinical stakeholders and governance committees."
        )

        return {
            "methodology": (
                "Metrics computed per Van Calster et al. (2025) methodology. "
                "Interpretation rests with your organization's governance process."
            ),
            # Status fields for CLI and governance display
            "status": status,
            "advisory": advisory,
            "disclaimer": disclaimer,
            # Threshold counts
            "outside_threshold_count": len(errors),
            "near_threshold_count": len(warnings),
            "within_threshold_count": n_pass,
            "total_checks": total_checks,
            "outside_threshold_items": errors,
            "near_threshold_items": warnings,
            "primary_fairness_metric": (
                self.config.primary_fairness_metric.value
                if self.config.primary_fairness_metric
                else None
            ),
            "justification_provided": bool(self.config.fairness_justification),
            # Backward compatibility aliases
            "n_errors": len(errors),
            "n_warnings": len(warnings),
            "n_pass": n_pass,
            "errors": errors,
            "warnings": warnings,
        }


# Legacy alias for backward compatibility
FairAudit = FairCareAudit


# Legacy AuditResult dataclass (for backward compatibility)
@dataclass
class AuditResult:
    """Legacy container for audit results. Use AuditResults instead."""

    model_name: str
    audit_date: str
    n_samples: int
    threshold: float
    group_col: str
    metrics_df: pl.DataFrame
    disparities_df: pl.DataFrame
    pass_count: int
    warn_count: int
    fail_count: int
    worst_disparity: tuple[str, str, float] | None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate text summary of audit results."""
        lines = [
            "=" * 60,
            f"  {self.model_name} - Fairness Analysis Results",
            "=" * 60,
            "",
            f"  Within Threshold: {self.pass_count}",
            f"  Near Threshold: {self.warn_count}",
            f"  Outside Threshold: {self.fail_count}",
            "",
            f"  N = {self.n_samples:,} | Classification Threshold = {self.threshold:.0%}",
            "",
        ]

        if self.worst_disparity:
            group, metric, value = self.worst_disparity
            lines.append(f"  Largest Disparity: {group}")
            lines.append(f"    {metric.upper()}: {value:+.1%} vs reference")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
