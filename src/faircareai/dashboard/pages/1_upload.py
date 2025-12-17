"""
FairCareAI - Data Upload Page

Handles data upload with validation and preview.
WCAG 2.1 AA compliant with clear error messaging.
"""

import polars as pl
import streamlit as st

from faircareai.dashboard.components.accessibility import (
    announce_status_change,
    render_semantic_heading,
    render_skip_link,
)
from faircareai.dashboard.components.audience_toggle import (
    render_audience_toggle,
)
from faircareai.visualization.themes import GOVERNANCE_DISCLAIMER_SHORT


def validate_dataframe(df: pl.DataFrame) -> tuple[bool, list[str]]:
    """Validate uploaded DataFrame for required columns.

    Args:
        df: Uploaded DataFrame.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors = []

    # Check for outcome column
    if "y_true" not in df.columns:
        errors.append("Missing required column: `y_true` (binary outcome: 0 or 1)")

    # Check for prediction column
    has_prob = "y_prob" in df.columns
    has_pred = "y_pred" in df.columns
    if not has_prob and not has_pred:
        errors.append(
            "Missing prediction column: need either `y_prob` (probabilities) or `y_pred` (binary)"
        )

    # Validate y_true is binary
    if "y_true" in df.columns:
        unique_values = df["y_true"].unique().to_list()
        if not set(unique_values).issubset({0, 1}):
            errors.append(
                f"Column `y_true` must be binary (0/1). Found values: {unique_values[:10]}"
            )

    # Validate y_prob is in [0, 1]
    if "y_prob" in df.columns:
        min_prob = df["y_prob"].min()
        max_prob = df["y_prob"].max()
        if min_prob < 0 or max_prob > 1:
            errors.append(
                f"Column `y_prob` must be in range [0, 1]. Found range: [{min_prob:.4f}, {max_prob:.4f}]"
            )

    # Check for at least one potential demographic column
    reserved_cols = {"y_true", "y_prob", "y_pred", "id", "patient_id", "index"}
    potential_demo_cols = [c for c in df.columns if c not in reserved_cols]
    if not potential_demo_cols:
        errors.append(
            "No demographic columns found. Need at least one column for fairness analysis."
        )

    return len(errors) == 0, errors


def detect_demographic_columns(df: pl.DataFrame) -> list[dict]:
    """Detect potential demographic columns with metadata.

    Args:
        df: DataFrame to analyze.

    Returns:
        List of column info dictionaries.
    """
    reserved_cols = {"y_true", "y_prob", "y_pred", "id", "patient_id", "index"}
    columns = []

    for col in df.columns:
        if col in reserved_cols:
            continue

        col_data = df[col]
        n_unique = col_data.n_unique()
        dtype = str(col_data.dtype)

        # Categorize column type
        if n_unique <= 20:
            col_type = "categorical"
            sample_values = col_data.unique().to_list()[:5]
        elif "int" in dtype.lower() or "float" in dtype.lower():
            col_type = "numeric"
            sample_values = [f"min: {col_data.min()}", f"max: {col_data.max()}"]
        else:
            col_type = "text"
            sample_values = col_data.head(3).to_list()

        # Check for likely demographic indicators
        demographic_keywords = [
            "race",
            "ethnicity",
            "gender",
            "sex",
            "age",
            "insurance",
            "language",
            "income",
            "education",
            "zip",
            "region",
            "site",
        ]
        is_likely_demo = any(kw in col.lower() for kw in demographic_keywords)

        columns.append(
            {
                "name": col,
                "type": col_type,
                "n_unique": n_unique,
                "sample_values": sample_values,
                "is_likely_demographic": is_likely_demo,
                "null_count": col_data.null_count(),
            }
        )

    return columns


def render_upload_page():
    """Render the data upload page."""
    render_skip_link()

    render_semantic_heading("Data Upload", level=1, id="page-title")

    st.caption(f"Step 1 of 4 | {GOVERNANCE_DISCLAIMER_SHORT}")

    # Audience toggle
    st.markdown("---")
    audience = render_audience_toggle()
    st.markdown("---")

    # Instructions based on audience
    if audience == "governance":
        st.markdown("""
        ### Upload Your Model's Predictions

        To analyze your AI model for fairness, we need:

        1. **Patient outcomes** - What actually happened (column: `y_true`)
        2. **Model predictions** - What the model predicted (column: `y_prob` or `y_pred`)
        3. **Patient demographics** - Groups to compare (e.g., race, insurance, age)

        Your data scientist can prepare this file for you.
        """)
    else:
        st.markdown("""
        ### Data Requirements

        Upload a CSV with the following columns:

        | Column | Type | Description |
        |--------|------|-------------|
        | `y_true` | Binary (0/1) | Ground truth outcome |
        | `y_prob` | Float [0,1] | Predicted probability (preferred) |
        | `y_pred` | Binary (0/1) | Binary prediction (alternative) |
        | `<demographic>` | Any | One or more demographic attributes |

        **Notes:**
        - Minimum recommended sample size: 500 per group
        - All rows should be from the same evaluation cohort
        - Ensure representative sampling across demographics
        """)

    # Demo data option
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file with patient outcomes and model predictions",
        )

    with col2:
        st.markdown("#### Or Try Demo Data")
        if st.button(
            "Load Demo Dataset",
            type="secondary",
            use_container_width=True,
            help="Load synthetic ICU mortality prediction data for demonstration",
        ):
            from faircareai.data.synthetic import generate_icu_mortality_data

            demo_df = generate_icu_mortality_data(n_samples=2000, seed=42)
            st.session_state["uploaded_data"] = demo_df
            st.session_state["data_source"] = "demo"
            announce_status_change("Demo dataset loaded successfully")
            st.success("Demo dataset loaded! Click 'Continue to Analysis' below.")

    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pl.read_csv(uploaded_file)
            st.session_state["uploaded_data"] = df
            st.session_state["data_source"] = "uploaded"
            announce_status_change("File uploaded successfully")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            announce_status_change("Error reading uploaded file", priority="assertive")
            return

    # Show data preview and validation
    if "uploaded_data" in st.session_state:
        df = st.session_state["uploaded_data"]

        st.markdown("---")
        render_semantic_heading("Data Preview", level=2)

        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            if "y_true" in df.columns:
                prevalence = df["y_true"].mean()
                st.metric("Outcome Prevalence", f"{prevalence:.1%}")

        # Data preview
        with st.expander("View First 10 Rows", expanded=True):
            st.dataframe(df.head(10).to_pandas(), use_container_width=True)

        # Validation
        st.markdown("---")
        render_semantic_heading("Validation", level=2)

        is_valid, errors = validate_dataframe(df)

        if is_valid:
            st.success("Data validation passed! Your file has all required columns.")
            announce_status_change("Data validation passed")
        else:
            st.error("Data validation failed. Please fix the following issues:")
            for error in errors:
                st.markdown(f"- {error}")
            announce_status_change("Data validation failed", priority="assertive")

        # Column selection
        if is_valid:
            st.markdown("---")
            render_semantic_heading("Select Demographic Columns", level=2)

            if audience == "governance":
                st.markdown("""
                Select which columns represent patient groups you want to analyze for fairness.
                We've highlighted columns that look like demographic attributes.
                """)
            else:
                st.markdown("""
                Select protected attributes for fairness analysis.
                Columns with low cardinality (≤20 unique values) work best.
                """)

            detected_cols = detect_demographic_columns(df)

            # Group by type - select likely demographic columns
            likely_demo = [c for c in detected_cols if c["is_likely_demographic"]]
            default_selections = [c["name"] for c in likely_demo]

            selected_cols = st.multiselect(
                "Demographic columns to analyze",
                options=[c["name"] for c in detected_cols],
                default=default_selections[:3],  # Limit default to 3
                help="Select columns representing patient demographic groups",
            )

            # Show column details
            if selected_cols:
                with st.expander("Selected Column Details", expanded=False):
                    for col_info in detected_cols:
                        if col_info["name"] in selected_cols:
                            st.markdown(f"**{col_info['name']}**")
                            st.markdown(f"- Type: {col_info['type']}")
                            st.markdown(f"- Unique values: {col_info['n_unique']}")
                            st.markdown(f"- Sample: {col_info['sample_values']}")
                            if col_info["null_count"] > 0:
                                st.warning(f"- Missing values: {col_info['null_count']:,}")
                            st.markdown("---")

            st.session_state["selected_demographic_cols"] = selected_cols

            # Continue button
            st.markdown("---")
            if selected_cols:
                if st.button(
                    "Continue to Analysis",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state["data_validated"] = True
                    st.switch_page("pages/2_analysis.py")
            else:
                st.warning("Please select at least one demographic column to continue.")


# Run the page
render_upload_page()
