"""
Pytest fixtures for FairCareAI tests.
"""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_binary_data() -> pl.DataFrame:
    """Create sample data for binary classification fairness testing."""
    np.random.seed(42)
    n = 300

    # Create demographic groups with different sizes
    race = ["White"] * 150 + ["Black"] * 100 + ["Hispanic"] * 50

    # Create risk scores with some group differences
    risk_scores = np.concatenate(
        [
            np.random.beta(2, 5, 150),  # White
            np.random.beta(2.5, 4, 100),  # Black - slightly higher risk
            np.random.beta(2, 5, 50),  # Hispanic
        ]
    )

    # Create outcomes correlated with risk scores
    outcomes = (np.random.random(n) < risk_scores).astype(int)

    return pl.DataFrame(
        {
            "risk_score": risk_scores,
            "readmitted": outcomes,
            "race": race,
        }
    )


@pytest.fixture
def sample_multigroup_data() -> pl.DataFrame:
    """Create sample data with multiple sensitive attributes."""
    np.random.seed(42)
    n = 400

    return pl.DataFrame(
        {
            "probability": np.random.uniform(0, 1, n),
            "outcome": np.random.binomial(1, 0.3, n),
            "race": np.random.choice(["White", "Black", "Hispanic", "Asian"], n),
            "sex": np.random.choice(["Male", "Female"], n),
            "insurance": np.random.choice(["Private", "Medicare", "Medicaid"], n),
        }
    )


@pytest.fixture
def perfect_parity_data() -> pl.DataFrame:
    """Create data with perfect parity across groups."""
    np.random.seed(42)
    n_per_group = 100

    # Same distribution for all groups
    data = []
    for race in ["White", "Black", "Hispanic"]:
        scores = np.random.uniform(0, 1, n_per_group)
        outcomes = (np.random.uniform(0, 1, n_per_group) < 0.5).astype(int)
        for score, outcome in zip(scores, outcomes):
            data.append({"risk_score": score, "outcome": outcome, "race": race})

    return pl.DataFrame(data)


@pytest.fixture
def severe_disparity_data() -> pl.DataFrame:
    """Create data with severe disparity between groups."""
    np.random.seed(42)

    # Group A: High TPR
    n_a = 100
    scores_a = np.random.uniform(0.6, 1.0, n_a)
    outcomes_a = np.ones(n_a, dtype=int)

    # Group B: Low TPR (model misses cases)
    n_b = 100
    scores_b = np.random.uniform(0.0, 0.4, n_b)
    outcomes_b = np.ones(n_b, dtype=int)

    return pl.DataFrame(
        {
            "risk_score": np.concatenate([scores_a, scores_b]),
            "outcome": np.concatenate([outcomes_a, outcomes_b]),
            "race": ["White"] * n_a + ["Black"] * n_b,
        }
    )


@pytest.fixture
def clustered_data() -> pl.DataFrame:
    """Create data with patient-level clustering (multiple encounters per patient)."""
    np.random.seed(42)
    n_patients = 50
    encounters_per_patient = 5
    n = n_patients * encounters_per_patient

    patient_ids = np.repeat(np.arange(n_patients), encounters_per_patient)
    groups = np.repeat(["A"] * 25 + ["B"] * 25, encounters_per_patient)

    # Patient-level characteristics (constant within patient)
    patient_risk = np.repeat(np.random.uniform(0.2, 0.8, n_patients), encounters_per_patient)

    # Encounter-level noise
    encounter_noise = np.random.normal(0, 0.1, n)

    y_prob = np.clip(patient_risk + encounter_noise, 0, 1)
    y_true = (np.random.uniform(0, 1, n) < patient_risk).astype(int)

    return pl.DataFrame(
        {
            "patient_id": patient_ids,
            "group": groups,
            "y_prob": y_prob,
            "y_true": y_true,
        }
    )
