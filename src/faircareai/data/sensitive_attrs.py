"""
Sensitive Attribute Auto-Detection

Suggests sensitive attributes based on common healthcare column patterns.
User can accept, modify, or ignore suggestions.

Methodology: CHAI RAIC Checkpoint 1 (protected attribute documentation).
Note: Suggestions require explicit user acceptance.
"""

from typing import Any

import polars as pl

# Common column name patterns for sensitive attributes in healthcare
SUGGESTED_PATTERNS: dict[str, dict[str, Any]] = {
    "race": {
        "patterns": [
            "race",
            "ethnicity",
            "race_eth",
            "patient_race",
            "race_cd",
            "race_ethnicity",
        ],
        "suggested_reference": "White",
        "clinical_justification": (
            "Required for CMS health equity monitoring and HEDIS reporting. "
            "Helps identify potential disparities in care delivery."
        ),
    },
    "sex": {
        "patterns": ["sex", "gender", "patient_sex", "sex_cd", "birth_sex"],
        "suggested_reference": "Male",
        "clinical_justification": (
            "Biological sex may influence disease prevalence and treatment response. "
            "Important for identifying sex-based disparities in care."
        ),
    },
    "age_group": {
        "patterns": ["age_group", "age_cat", "age_bucket", "age_band", "age_category"],
        "suggested_reference": None,  # Will use largest group
        "clinical_justification": (
            "Age affects baseline risk and model generalizability. "
            "Older populations may have different risk profiles."
        ),
    },
    "insurance": {
        "patterns": [
            "insurance",
            "payer",
            "insurance_type",
            "coverage",
            "payer_type",
            "payer_category",
        ],
        "suggested_reference": "Commercial",
        "clinical_justification": (
            "Insurance status correlates with access to care and outcomes. "
            "Medicaid/uninsured may face access barriers affecting model performance."
        ),
    },
    "language": {
        "patterns": ["language", "primary_language", "lang", "language_cd", "preferred_language"],
        "suggested_reference": "English",
        "clinical_justification": (
            "Language barriers affect care quality and documentation completeness. "
            "Limited English proficiency patients may have different care patterns."
        ),
    },
    "disability": {
        "patterns": ["disability", "disabled", "disability_status", "functional_status"],
        "suggested_reference": "No",
        "clinical_justification": (
            "Disability status affects care access and outcome measurement. "
            "Important for ADA compliance and equitable care delivery."
        ),
    },
}


def suggest_sensitive_attributes(df: pl.DataFrame) -> list[dict]:
    """
    Scan DataFrame columns and suggest likely sensitive attributes.

    Args:
        df: Polars DataFrame with patient data.

    Returns:
        List of suggestions that user can accept/modify/ignore.
        Each suggestion contains:
        - suggested_name: Display name for the attribute
        - detected_column: Actual column name in data
        - unique_values: Preview of values (first 10)
        - n_unique: Number of unique values
        - missing_rate: Proportion of missing values
        - suggested_reference: Suggested reference group
        - clinical_justification: Why this attribute matters
        - accepted: Always False (user must explicitly accept)
    """
    suggestions = []
    columns_lower = {c.lower(): c for c in df.columns}

    for attr_name, config in SUGGESTED_PATTERNS.items():
        for pattern in config["patterns"]:
            if pattern in columns_lower:
                actual_col = columns_lower[pattern]
                col_data = df[actual_col]

                # Get unique values
                unique_vals = col_data.drop_nulls().unique().sort().to_list()

                # Calculate missing rate
                missing_rate = col_data.null_count() / len(df)

                suggestions.append(
                    {
                        "suggested_name": attr_name,
                        "detected_column": actual_col,
                        "unique_values": unique_vals[:10],  # First 10 for preview
                        "n_unique": len(unique_vals),
                        "missing_rate": float(missing_rate),
                        "suggested_reference": config["suggested_reference"],
                        "clinical_justification": config["clinical_justification"],
                        "accepted": False,  # User must explicitly accept
                    }
                )
                break  # Only match first pattern per attribute

    return suggestions


def display_suggestions(suggestions: list[dict]) -> str:
    """Format suggestions for display in notebook/CLI.

    Args:
        suggestions: List of suggestion dicts from suggest_sensitive_attributes().

    Returns:
        Formatted string for display.
    """
    if not suggestions:
        return (
            "=" * 60
            + "\n"
            + "NO SENSITIVE ATTRIBUTES DETECTED\n"
            + "=" * 60
            + "\n\n"
            + "FairCareAI did not detect common sensitive attribute columns.\n"
            + "You can manually add attributes using:\n"
            + "  audit.add_sensitive_attribute(name='race', column='my_race_col')\n"
            + "=" * 60
        )

    lines = [
        "=" * 60,
        "SUGGESTED SENSITIVE ATTRIBUTES",
        "=" * 60,
        "",
        "FairCareAI detected the following potential sensitive attributes.",
        "Review and accept/modify as appropriate for your use case.",
        "",
    ]

    for i, s in enumerate(suggestions, 1):
        values_preview = str(s["unique_values"][:5])
        if len(s["unique_values"]) > 5:
            values_preview = values_preview[:-1] + ", ...]"

        lines.extend(
            [
                f"[{i}] {s['suggested_name'].upper()}",
                f"    Column: {s['detected_column']}",
                f"    Values: {values_preview}",
                f"    N unique: {s['n_unique']}",
                f"    Missing: {s['missing_rate']:.1%}",
                f"    Suggested reference: {s['suggested_reference'] or '(largest group)'}",
                f"    Justification: {s['clinical_justification'][:60]}...",
                "",
            ]
        )

    lines.extend(
        [
            "=" * 60,
            "To accept suggestions:",
            "  audit.accept_suggested_attributes([1, 2])  # by index",
            "  audit.accept_suggested_attributes(['race', 'sex'])  # by name",
            "",
            "To modify reference group:",
            "  audit.accept_suggested_attributes([1], modify={'race': {'reference': 'Black'}})",
            "",
            "To add custom attributes:",
            "  audit.add_sensitive_attribute(name='custom', column='my_col', reference='Group A')",
            "=" * 60,
        ]
    )

    return "\n".join(lines)


def validate_attribute(
    df: pl.DataFrame,
    name: str,
    column: str,
    reference: str | None = None,
    categories: list[str] | None = None,
) -> list[str]:
    """Validate a sensitive attribute against the data.

    Args:
        df: Polars DataFrame with patient data.
        name: Display name for the attribute.
        column: Column name in data.
        reference: Reference group for comparisons.
        categories: Expected category values.

    Returns:
        List of validation issues (empty if valid).
    """
    issues = []

    if column not in df.columns:
        issues.append(f"Column '{column}' not found in data")
        return issues

    col_data = df[column]
    actual_values = col_data.drop_nulls().unique().to_list()

    if reference and reference not in actual_values:
        issues.append(f"Reference group '{reference}' not in data values: {actual_values[:10]}")

    if categories:
        missing = set(categories) - set(actual_values)
        if missing:
            issues.append(f"Expected categories not found: {missing}")

    # Warn about high missing rate
    missing_rate = col_data.null_count() / len(df)
    if missing_rate > 0.1:
        issues.append(
            f"High missing rate ({missing_rate:.1%}) for {name}. "
            "Consider how missingness may affect fairness analysis."
        )

    # Warn about small groups
    value_counts = df.group_by(column).len()
    min_count_val = value_counts["len"].min()
    min_count: int | None = None
    if min_count_val is not None and isinstance(min_count_val, (int, float)):
        min_count = int(min_count_val)
    if min_count is not None and min_count < 100:
        small_groups = value_counts.filter(pl.col("len") < 100)[column].to_list()
        issues.append(
            f"Small subgroups (n<100) detected: {small_groups}. "
            "Results may have wide confidence intervals."
        )

    return issues


def get_reference_group(
    df: pl.DataFrame,
    column: str,
    suggested_reference: str | None = None,
) -> str:
    """Determine the reference group for comparisons.

    Args:
        df: Polars DataFrame with patient data.
        column: Column name for the attribute.
        suggested_reference: User-suggested reference group.

    Returns:
        Reference group to use (suggested if valid, else largest group).
    """
    col_data = df[column]
    actual_values = col_data.drop_nulls().unique().to_list()

    # Use suggested if valid
    if suggested_reference and suggested_reference in actual_values:
        return suggested_reference

    # Otherwise use largest group
    value_counts = df.group_by(column).len().sort("len", descending=True)
    if len(value_counts) == 0:
        raise ValueError(f"Column '{column}' has no data")
    return str(value_counts[column][0])
