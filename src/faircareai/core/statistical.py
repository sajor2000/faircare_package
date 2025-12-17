"""
Statistical Functions for FairCareAI

Implements confidence intervals and hypothesis tests following
the statistical standards in CLAUDE.md:
- Wilson score CI for single proportions
- Newcombe-Wilson CI for proportion differences
- Clopper-Pearson exact CI for extreme proportions
"""

import math

from scipy import stats


def wilson_score_ci(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    The Wilson score interval is recommended over the Wald interval
    for its better coverage properties, especially near 0 or 1.

    Args:
        successes: Number of successes (positive outcomes).
        trials: Total number of trials.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound) for the proportion.

    Raises:
        ValueError: If successes > trials or values are negative.

    Example:
        >>> wilson_score_ci(80, 100)
        (0.711, 0.869)

    Reference:
        Wilson, E.B. (1927). Probable inference, the law of succession,
        and statistical inference.
    """
    if successes < 0 or trials < 0:
        raise ValueError("successes and trials must be non-negative")
    if successes > trials:
        raise ValueError(f"successes ({successes}) cannot exceed trials ({trials})")

    if trials == 0:
        return (0.0, 1.0)

    p_hat = successes / trials
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    z2 = z * z

    denominator = 1 + z2 / trials
    center = (p_hat + z2 / (2 * trials)) / denominator
    margin = (z / denominator) * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * trials)) / trials)

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


def clopper_pearson_ci(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute Clopper-Pearson exact confidence interval for a proportion.

    Use this instead of Wilson when p < 0.01 or p > 0.99.

    Args:
        successes: Number of successes (positive outcomes).
        trials: Total number of trials.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound) for the proportion.

    Reference:
        Clopper, C. & Pearson, E.S. (1934). The use of confidence or
        fiducial limits illustrated in the case of the binomial.
    """
    if successes < 0 or trials < 0:
        raise ValueError("successes and trials must be non-negative")
    if successes > trials:
        raise ValueError(f"successes ({successes}) cannot exceed trials ({trials})")

    if trials == 0:
        return (0.0, 1.0)

    alpha = 1 - confidence

    if successes == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)

    if successes == trials:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)

    return (lower, upper)


def newcombe_wilson_ci(
    successes1: int,
    trials1: int,
    successes2: int,
    trials2: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute Newcombe-Wilson confidence interval for difference of proportions.

    Uses the Newcombe hybrid score method (Method 10 in Newcombe 1998).

    Args:
        successes1: Successes in group 1 (reference).
        trials1: Trials in group 1.
        successes2: Successes in group 2 (comparison).
        trials2: Trials in group 2.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound) for p2 - p1 difference.

    Example:
        >>> newcombe_wilson_ci(79, 100, 71, 100)  # ~8% difference
        (-0.17, 0.01)

    Reference:
        Newcombe, R.G. (1998). Interval estimation for the difference
        between independent proportions: comparison of eleven methods.
    """
    if trials1 == 0 or trials2 == 0:
        return (-1.0, 1.0)

    p1 = successes1 / trials1
    p2 = successes2 / trials2
    diff = p2 - p1

    # Get Wilson CIs for each proportion
    l1, u1 = wilson_score_ci(successes1, trials1, confidence)
    l2, u2 = wilson_score_ci(successes2, trials2, confidence)

    # Newcombe hybrid method
    lower = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)

    return (max(-1.0, lower), min(1.0, upper))


def get_sample_status(n: int, n_positive: int | None = None) -> str:
    """
    Determine sample size status per CLAUDE.md guidelines.

    Args:
        n: Total sample size.
        n_positive: Number of positive cases (optional).

    Returns:
        Status string: "ADEQUATE", "MODERATE", "LOW", or "VERY_LOW"
    """
    if n < 10:
        return "VERY_LOW"
    if n < 30:
        return "LOW"
    if n < 50:
        return "MODERATE"

    # Check np and n(1-p) rule if we have positive count
    if n_positive is not None:
        n_negative = n - n_positive
        if n_positive < 5 or n_negative < 5:
            return "LOW"
        if n_positive < 10 or n_negative < 10:
            return "MODERATE"

    return "ADEQUATE"


def get_sample_warning(
    group: str,
    n: int,
    n_positive: int | None = None,
    status: str | None = None,
) -> str | None:
    """
    Generate appropriate warning message for sample size.

    Args:
        group: Group name for message.
        n: Sample size.
        n_positive: Number of positive cases.
        status: Pre-computed status (optional).

    Returns:
        Warning message or None if adequate sample.
    """
    if status is None:
        status = get_sample_status(n, n_positive)

    if status == "ADEQUATE":
        return None

    if status == "VERY_LOW":
        return (
            f"CAUTION: Results for {group} based on very small sample (n={n}). "
            "Estimates are highly uncertain. Consider excluding from conclusions."
        )

    if status == "LOW":
        return (
            f"Note: Results for {group} based on limited sample (n={n}). "
            "Interpret with caution. Confidence intervals are wide."
        )

    if status == "MODERATE":
        return (
            f"Results for {group} based on moderate sample (n={n}). "
            "Consider this limitation when interpreting."
        )

    return None


def z_test_two_proportions(
    successes1: int,
    trials1: int,
    successes2: int,
    trials2: int,
) -> tuple[float, float]:
    """
    Two-proportion z-test for difference in proportions.

    Args:
        successes1: Successes in group 1.
        trials1: Trials in group 1.
        successes2: Successes in group 2.
        trials2: Trials in group 2.

    Returns:
        Tuple of (z_statistic, p_value).
    """
    if trials1 == 0 or trials2 == 0:
        return (0.0, 1.0)

    p1 = successes1 / trials1
    p2 = successes2 / trials2

    # Pooled proportion under null hypothesis
    p_pooled = (successes1 + successes2) / (trials1 + trials2)

    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / trials1 + 1 / trials2))

    if se == 0:
        return (0.0, 1.0)

    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return (z, p_value)
