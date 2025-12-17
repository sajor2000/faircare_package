# Contributing to FairCareAI

Thank you for your interest in contributing to FairCareAI! This document provides guidelines for contributing to the project.

---

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Support newcomers
- Prioritize patient safety and equity in all discussions

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) `uv` for faster dependency management

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/sajor2000/faircare_package.git
cd faircareai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# For PDF/PPTX export testing
pip install -e ".[export]"

# Or install everything
pip install -e ".[all]"
```

### Using uv (Faster Alternative)

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,export]"
```

### Verify Setup

```bash
# Run tests
pytest tests/ -v

# Type checking
mypy src/faircareai

# Linting
ruff check src/

# Format check
ruff format --check src/
```

---

## Project Structure

```
faircareai/
├── src/faircareai/      # Main package
│   ├── core/            # Orchestration, config, results
│   ├── metrics/         # Performance and fairness metrics
│   ├── visualization/   # Plotly charts, themes
│   ├── reports/         # HTML/PDF/PPTX generation
│   ├── dashboard/       # Streamlit app
│   ├── data/            # Data utilities
│   └── fairness/        # Metric decision support
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Example scripts
└── notebooks/           # Jupyter tutorials
```

### Key Files

| File | Purpose |
|------|---------|
| `core/audit.py` | Main `FairCareAudit` class |
| `core/config.py` | Configuration and enums |
| `core/results.py` | `AuditResults` container |
| `metrics/fairness.py` | Fairness metric implementations |
| `visualization/themes.py` | Okabe-Ito palette, WCAG compliance |

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_audit.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=faircareai --cov-report=html
open htmlcov/index.html
```

### Test Markers

```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run integration tests only
pytest tests/ -m integration
```

### Writing Tests

- Place tests in `tests/` directory
- Name files `test_*.py`
- Use descriptive test names: `test_method_condition_expected`
- Use fixtures from `conftest.py`

```python
# tests/test_example.py
import pytest
import polars as pl
from faircareai import FairCareAudit

class TestExampleFeature:
    def test_feature_with_valid_input_returns_expected(self, sample_df):
        """Feature should return X when given valid input."""
        audit = FairCareAudit(sample_df, pred_col="y_prob", target_col="y_true")
        result = audit.some_method()
        assert result == expected_value

    def test_feature_with_invalid_input_raises(self, sample_df):
        """Feature should raise ValueError for invalid input."""
        audit = FairCareAudit(sample_df, pred_col="y_prob", target_col="y_true")
        with pytest.raises(ValueError, match="specific message"):
            audit.some_method(invalid_param)
```

---

## Code Style

### Formatting

FairCareAI uses [Ruff](https://github.com/astral-sh/ruff) for formatting and linting.

```bash
# Auto-format
ruff format src/ tests/

# Check linting
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/
```

### Type Hints

All public APIs require type hints. Use strict mypy settings:

```bash
mypy src/faircareai
```

```python
# Good
def compute_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ...

# Better - with docstring
def compute_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the metric value.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probabilities [0, 1].

    Returns:
        Metric value in range [0, 1].

    Raises:
        ValueError: If arrays have different lengths.
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int = 10) -> dict:
    """Short description of function.

    Longer description if needed, explaining behavior,
    edge cases, or important notes.

    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 10.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is empty.
        TypeError: When param2 is not an integer.

    Example:
        >>> result = function_name("test", 20)
        >>> print(result)
        {'key': 'value'}
    """
    ...
```

### Import Order

Ruff handles this automatically. Manual order:

1. Standard library
2. Third-party packages
3. Local imports

```python
import json
from pathlib import Path

import polars as pl
import plotly.graph_objects as go

from faircareai.core.config import FairnessConfig
from faircareai.core.logging import get_logger
```

---

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

- Write code following style guidelines
- Add/update tests
- Update documentation if needed

### 3. Run Checks

```bash
# Format
ruff format src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/faircareai

# Tests
pytest tests/ -v
```

### 4. Commit

Use clear commit messages:

```bash
git commit -m "Add equalized odds computation for subgroups

- Implement compute_equalized_odds() function
- Add bootstrap CI support
- Update tests for edge cases

Refs: #123"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- Clear title describing the change
- Description of what and why
- Link to related issues
- Screenshots for UI changes

### 6. Review Process

- Maintainers will review within 1-2 business days
- Address feedback promptly
- Request re-review when ready

---

## Adding New Features

### New Fairness Metric

1. Add to `FairnessMetric` enum in `core/config.py`
2. Implement in `metrics/fairness.py`
3. Add to decision tree in `fairness/decision_tree.py`
4. Write tests in `tests/test_fairness_metrics.py`
5. Update documentation in `docs/METHODOLOGY.md`

### New Visualization

1. Add function in appropriate `visualization/*.py` file
2. Follow Okabe-Ito palette (use `themes.py`)
3. Ensure WCAG 2.1 AA compliance
4. Add to `AuditResults` if needed
5. Write tests

### New Export Format

1. Add to `reports/generator.py`
2. Update `AuditResults.to_*()` method
3. Add to `[export]` dependencies if needed
4. Test on all platforms

---

## Scientific Standards

### Citation Requirements

All statistical methods must be properly cited:

```python
def wilson_score_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute Wilson score confidence interval.

    Reference:
        Wilson, E. B. (1927). Probable inference, the law of succession,
        and statistical inference. Journal of the American Statistical
        Association, 22(158), 209-212.
    """
    ...
```

### Methodology Documentation

- New metrics must be documented in `docs/METHODOLOGY.md`
- Include mathematical definition
- Cite source paper
- Explain when to use

### Validation

- Compare results against established implementations
- Document validation in tests or separate validation script
- Include edge case handling

---

## Accessibility Guidelines

### Visualization

- Use Okabe-Ito palette from `visualization/themes.py`
- Ensure 4.5:1 contrast ratio for text
- Don't rely on color alone to convey information
- Add alt text to all figures

### Documentation

- Use clear, simple language
- Provide code examples
- Include screenshots for UI features

---

## Release Process

Maintainers handle releases:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag
4. GitHub Actions builds and publishes to PyPI

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Security**: Email security@example.com (do not open public issue)

---

## Recognition

Contributors are recognized in:

- `CHANGELOG.md` for each release
- GitHub Contributors page
- Annual acknowledgments

Thank you for contributing to healthcare AI fairness!
