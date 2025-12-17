"""
Synthetic Data Generator for ICU Mortality Prediction

Generates realistic ICU mortality prediction data with intentional
demographic disparities for fairness auditing demonstrations.
"""

from typing import Any

import numpy as np
import polars as pl


def generate_icu_mortality_data(
    n_samples: int = 2000,
    seed: int = 42,
    disparity_strength: float = 0.08,
    threshold: float = 0.5,
) -> pl.DataFrame:
    """
    Generate synthetic ICU mortality prediction data.

    Creates a dataset with realistic demographic distributions and
    intentional disparities to demonstrate fairness auditing.

    Args:
        n_samples: Number of patient records to generate.
        seed: Random seed for reproducibility.
        disparity_strength: Magnitude of TPR disparity (default 8%).
        threshold: Decision threshold for y_pred (default 0.5).

    Returns:
        Polars DataFrame with columns:
        - patient_id: Unique identifier
        - age_group: 18-44, 45-64, 65-79, 80+
        - sex: Male, Female
        - race_ethnicity: White, Black, Hispanic, Asian, Other
        - insurance: Private, Medicare, Medicaid, Uninsured
        - language: English, Spanish, Other
        - y_true: Ground truth mortality (0/1)
        - y_prob: Model predicted probability (0.0-1.0)
        - y_pred: Binary prediction at threshold (0/1)

    Example:
        >>> df = generate_icu_mortality_data(n_samples=1000, seed=42)
        >>> df.shape
        (1000, 9)
    """
    rng = np.random.default_rng(seed)

    # Define demographic distributions (based on US ICU demographics)
    demographics: dict[str, dict[str, Any]] = {
        "age_group": {
            "values": ["18-44", "45-64", "65-79", "80+"],
            "probs": [0.15, 0.30, 0.35, 0.20],
            "mortality_modifier": [0.6, 0.8, 1.0, 1.4],  # Age affects mortality
        },
        "sex": {
            "values": ["Male", "Female"],
            "probs": [0.55, 0.45],
            "mortality_modifier": [1.05, 0.95],
        },
        "race_ethnicity": {
            "values": ["White", "Black", "Hispanic", "Asian", "Other"],
            "probs": [0.40, 0.20, 0.175, 0.10, 0.125],
            "mortality_modifier": [1.0, 1.15, 0.95, 0.85, 1.05],
        },
        "insurance": {
            "values": ["Private", "Medicare", "Medicaid", "Uninsured"],
            "probs": [0.30, 0.35, 0.20, 0.15],
            "mortality_modifier": [0.85, 1.15, 1.05, 1.20],
        },
        "language": {
            "values": ["English", "Spanish", "Other"],
            "probs": [0.85, 0.12, 0.03],  # Small "Other" for ghosting demo
            "mortality_modifier": [1.0, 1.0, 1.05],
        },
    }

    # TPR modifiers per group (1.0 = reference, <1.0 = disparity)
    tpr_modifiers = {
        "race_ethnicity": {
            "White": 1.0,
            "Black": 1.0 - disparity_strength,  # Key disparity
            "Hispanic": 0.98,
            "Asian": 0.99,
            "Other": 0.97,
        },
        "insurance": {
            "Private": 1.0,
            "Medicare": 0.98,
            "Medicaid": 0.95,
            "Uninsured": 1.0 - disparity_strength * 1.2,  # Larger disparity
        },
        "language": {
            "English": 1.0,
            "Spanish": 0.97,
            "Other": 0.94,
        },
        "age_group": {
            "18-44": 1.0,
            "45-64": 0.99,
            "65-79": 0.98,
            "80+": 0.95,
        },
        "sex": {
            "Male": 1.0,
            "Female": 0.99,
        },
    }

    # Generate demographics
    data = {"patient_id": np.arange(1, n_samples + 1)}

    for attr, config in demographics.items():
        values = list(config["values"])
        probs = list(config["probs"])
        data[attr] = rng.choice(values, size=n_samples, p=probs)

    # Base mortality rate (ICU mortality ~15%)
    base_mortality = 0.15

    # Calculate individual mortality probabilities
    mortality_probs = np.full(n_samples, base_mortality)

    for attr, config in demographics.items():
        values_list = list(config["values"])
        modifiers_list = list(config["mortality_modifier"])
        for i, val in enumerate(values_list):
            mask = data[attr] == val
            mortality_probs[mask] *= modifiers_list[i]

    # Clip to valid range
    mortality_probs = np.clip(mortality_probs, 0.02, 0.50)

    # Generate ground truth outcomes
    y_true = (rng.random(n_samples) < mortality_probs).astype(int)
    data["y_true"] = y_true

    # Generate model predictions
    # For a good classifier, positive cases should have higher predictions
    # We create predictions that center around 0.7 for positives and 0.3 for negatives

    y_prob = np.zeros(n_samples, dtype=np.float64)

    # Positive cases: predictions centered around 0.7
    pos_mask = y_true == 1
    y_prob[pos_mask] = rng.normal(0.70, 0.15, pos_mask.sum())

    # Negative cases: predictions centered around 0.25
    neg_mask = y_true == 0
    y_prob[neg_mask] = rng.normal(0.25, 0.12, neg_mask.sum())

    # Apply TPR modifiers (shift predictions down for disadvantaged groups)
    for attr, modifiers in tpr_modifiers.items():
        for group, modifier in modifiers.items():
            # For true positives in this group, reduce prediction if modifier < 1
            mask = (data[attr] == group) & (y_true == 1)
            if modifier < 1.0:
                # Shift predictions down, making FN more likely
                shift = (1.0 - modifier) * 0.4  # Scale the disparity effect
                y_prob[mask] -= shift

    # Clip to valid probability range
    y_prob = np.clip(y_prob, 0.01, 0.99)
    data["y_prob"] = y_prob

    # Binary predictions at threshold
    data["y_pred"] = (y_prob >= threshold).astype(int)

    # Create Polars DataFrame
    df = pl.DataFrame(data)

    # Reorder columns
    return df.select(
        [
            "patient_id",
            "age_group",
            "sex",
            "race_ethnicity",
            "insurance",
            "language",
            "y_true",
            "y_prob",
            "y_pred",
        ]
    )


def get_data_summary(df: pl.DataFrame) -> dict:
    """
    Get summary statistics for the synthetic dataset.

    Args:
        df: DataFrame from generate_icu_mortality_data()

    Returns:
        Dictionary with summary statistics.
    """
    n = len(df)
    n_deaths = df["y_true"].sum()
    n_predicted_deaths = df["y_pred"].sum()

    return {
        "n_samples": n,
        "n_deaths": n_deaths,
        "mortality_rate": n_deaths / n,
        "n_predicted_deaths": n_predicted_deaths,
        "prediction_rate": n_predicted_deaths / n,
        "demographics": {
            col: df[col].value_counts().sort("count", descending=True).to_dicts()
            for col in ["age_group", "sex", "race_ethnicity", "insurance", "language"]
        },
    }
