"""Test script to verify interactive charts are embedded in HTML reports."""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from faircareai import FairCareAudit
from faircareai.core.config import FairnessConfig, FairnessMetric, UseCaseType

# Create test data
np.random.seed(42)
n = 400

df = pl.DataFrame({
    "outcome": np.random.binomial(1, 0.3, n),
    "probability": np.random.beta(2, 5, n),
    "race": np.random.choice(["White", "Black", "Asian", "Hispanic"], n),
    "sex": np.random.choice(["Male", "Female"], n),
})

# Initialize audit
audit = FairCareAudit(
    data=df,
    pred_col="probability",
    target_col="outcome",
)

# Configure attributes
audit.add_sensitive_attribute(name="race", column="race", reference="White")
audit.add_sensitive_attribute(name="sex", column="sex", reference="Male")

# Set config
audit.config = FairnessConfig(
    model_name="Test Model for HTML Charts",
    primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
    fairness_justification="Testing interactive charts in HTML reports",
    use_case_type=UseCaseType.SCREENING,
)

# Run audit
print("Running audit...")
results = audit.run(bootstrap_ci=False)

# Generate HTML report
with tempfile.TemporaryDirectory() as tmp_dir:
    output_path = Path(tmp_dir) / "test_report.html"
    print(f"Generating HTML report to {output_path}...")
    html_path = results.to_html(output_path)

    # Read and check content
    content = html_path.read_text()

    # Check for Plotly.js
    if "plotly" in content.lower():
        print("✅ Plotly.js script found in HTML")
    else:
        print("❌ Plotly.js script NOT found in HTML")

    # Check for interactive chart divs
    if 'class="plotly-graph-div"' in content:
        print("✅ Plotly chart divs found in HTML")
    else:
        print("❌ Plotly chart divs NOT found in HTML")

    # Check for chart placeholders (should NOT be there anymore)
    if "[Interactive" in content:
        print("❌ Chart placeholders still present in HTML")
    else:
        print("✅ No chart placeholders found (good!)")

    # Count how many charts were embedded
    chart_count = content.count('class="plotly-graph-div"')
    print(f"📊 Found {chart_count} interactive charts in HTML")

    print(f"\n✅ HTML report generated successfully at: {html_path}")
    print(f"File size: {html_path.stat().st_size / 1024:.1f} KB")
