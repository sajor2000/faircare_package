"""
Central citation and methodology references for FairCareAI.

Combines two complementary frameworks:
- CHAI RAIC: Governance structure (what to document)
- Van Calster et al.: Metrics methodology (how to measure)
"""

# ============================================================================
# CHAI - Coalition for Health AI (Governance Framework)
# ============================================================================
CHAI_CITATION = {
    "short": "CHAI RAIC Checkpoint 1 (2024)",
    "full": (
        "Coalition for Health AI. RAIC Checkpoint 1: Model Card for "
        "Responsible AI in Clinical Practice. Version 1.0. 2024."
    ),
    "url": "https://www.coalitionforhealthai.org",
    "purpose": "Governance criteria and documentation requirements",
}

# CHAI criteria codes used in this package
CHAI_CRITERIA = {
    "AC1.CR92": "Bias Testing - Disparity analysis across demographic groups",
    "AC1.CR95": "Performance Reporting - Subgroup performance metrics",
    "AC1.CR102": "Calibration Assessment - Predicted vs observed outcomes",
}

# ============================================================================
# Van Calster et al. - Metrics Methodology
# ============================================================================
VAN_CALSTER_CITATION = {
    "short": "Van Calster et al. (2025)",
    "full": (
        "Van Calster B, Collins GS, Vickers AJ, et al. "
        "Evaluation of performance measures in predictive artificial intelligence "
        "models to support medical decisions. Lancet Digit Health 2025."
    ),
    "doi": "10.1016/j.landig.2025.100916",
    "url": "https://doi.org/10.1016/j.landig.2025.100916",
    "purpose": "Metrics selection and interpretation methodology",
}

# ============================================================================
# TRIPOD+AI - Reporting Guidelines
# ============================================================================
TRIPOD_AI_CITATION = {
    "short": "Collins et al. (2024)",
    "full": (
        "Collins GS, Moons KGM, Dhiman P, et al. "
        "TRIPOD+AI statement: updated guidance for reporting clinical "
        "prediction models that use regression or machine learning methods. BMJ 2024."
    ),
    "doi": "10.1136/bmj-2023-078378",
    "purpose": "Transparent reporting standards",
}

# ============================================================================
# Fairness Theory
# ============================================================================
FAIRNESS_CITATIONS = {
    "chouldechova": {
        "short": "Chouldechova (2017)",
        "full": (
            "Chouldechova A. Fair prediction with disparate impact: "
            "A study of bias in recidivism prediction instruments. Big Data 2017."
        ),
    },
    "kleinberg": {
        "short": "Kleinberg et al. (2017)",
        "full": (
            "Kleinberg J, Mullainathan S, Raghavan M. "
            "Inherent Trade-Offs in the Fair Determination of Risk Scores. ITCS 2017."
        ),
    },
}

# ============================================================================
# Combined Methodology Statements
# ============================================================================
METHODOLOGY_STATEMENT = (
    "Metrics computed per Van Calster et al. (2025) methodology, "
    "organized following CHAI RAIC Checkpoint 1 governance criteria. "
    "Interpretation rests with your organization's governance process."
)

# Short version for UI
METHODOLOGY_DISCLAIMER = (
    "Metrics per Van Calster et al. (2025) and CHAI RAIC criteria. "
    "Interpretation rests with your organization."
)

# Very short version for tight spaces
METHODOLOGY_SHORT = "Van Calster et al. (2025) / CHAI RAIC"
