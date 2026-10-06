from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_academic_outcome_model.pkl"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "student_outcome_prediction.ipynb"
TARGET_COLUMN = "Target"
CLASS_LABELS = ["Dropout", "Enrolled", "Graduate"]
FINAL_MODEL_NAME = "Random Forest"
LEAKAGE_COLUMNS = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]
UCI_CATEGORY_LABELS: dict[str, dict[int, str]] = {
    "Marital status": {
        1: "Single",
        2: "Married",
        3: "Widower",
        4: "Divorced",
        5: "Facto union",
        6: "Legally separated",
    },
    "Application mode": {
        1: "1st phase — general contingent",
        2: "Ordinance No. 612/93",
        5: "1st phase — special contingent (Azores Island)",
        7: "Holders of other higher courses",
        10: "Ordinance No. 854-B/99",
        15: "International student (bachelor)",
        16: "1st phase — special contingent (Madeira Island)",
        17: "2nd phase — general contingent",
        18: "3rd phase — general contingent",
        26: "Ordinance No. 533-A/99, item b2 (Different Plan)",
        27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
        39: "Over 23 years old",
        42: "Transfer",
        43: "Change of course",
        44: "Technological specialization diploma holders",
        51: "Change of institution/course",
        53: "Short cycle diploma holders",
        57: "Change of institution/course (International)",
    },
    "Course": {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (evening attendance)",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication",
        9853: "Basic Education",
        9991: "Management (evening attendance)",
    },
    "Application order": {
        0: "First choice",
        1: "Second choice",
        2: "Third choice",
        3: "Fourth choice",
        4: "Fifth choice",
        5: "Sixth choice",
        6: "Seventh choice",
        7: "Eighth choice",
        8: "Ninth choice",
        9: "Last choice",
    },
    "Daytime/evening attendance": {0: "Evening", 1: "Daytime"},
    "Previous qualification": {
        1: "Secondary education",
        2: "Higher education — bachelor's degree",
        3: "Higher education — degree",
        4: "Higher education — master's",
        5: "Higher education — doctorate",
        6: "Frequency of higher education",
        9: "12th year of schooling — not completed",
        10: "11th year of schooling — not completed",
        12: "Other — 11th year of schooling",
        14: "10th year of schooling",
        15: "10th year of schooling — not completed",
        19: "Basic education 3rd cycle or equivalent",
        38: "Basic education 2nd cycle or equivalent",
        39: "Technological specialization course",
        40: "Higher education — 1st cycle degree",
        42: "Professional higher technical course",
        43: "Higher education — 2nd cycle master's",
    },
    "Nacionality": {
        1: "Portuguese",
        2: "German",
        6: "Spanish",
        11: "Italian",
        13: "Dutch",
        14: "English",
        17: "Lithuanian",
        21: "Angolan",
        22: "Cape Verdean",
        24: "Guinean",
        25: "Mozambican",
        26: "Santomean",
        32: "Turkish",
        41: "Brazilian",
        62: "Romanian",
        100: "Moldovan",
        101: "Mexican",
        103: "Ukrainian",
        105: "Russian",
        108: "Cuban",
        109: "Colombian",
    },
    "Mother's qualification": {
        1: "Secondary education",
        2: "Higher education — bachelor's degree",
        3: "Higher education — degree",
        4: "Higher education — master's",
        5: "Higher education — doctorate",
        6: "Frequency of higher education",
        9: "12th year of schooling — not completed",
        10: "11th year of schooling — not completed",
        12: "Other — 11th year of schooling",
        14: "10th year of schooling",
        15: "10th year of schooling — not completed",
        19: "Basic education 3rd cycle or equivalent",
        38: "Basic education 2nd cycle or equivalent",
        39: "Technological specialization course",
        40: "Higher education — 1st cycle degree",
        42: "Professional higher technical course",
        43: "Higher education — 2nd cycle master's",
    },
    "Father's qualification": {
        1: "Secondary education",
        2: "Higher education — bachelor's degree",
        3: "Higher education — degree",
        4: "Higher education — master's",
        5: "Higher education — doctorate",
        6: "Frequency of higher education",
        9: "12th year of schooling — not completed",
        10: "11th year of schooling — not completed",
        12: "Other — 11th year of schooling",
        14: "10th year of schooling",
        15: "10th year of schooling — not completed",
        19: "Basic education 3rd cycle or equivalent",
        38: "Basic education 2nd cycle or equivalent",
        39: "Technological specialization course",
        40: "Higher education — 1st cycle degree",
        42: "Professional higher technical course",
        43: "Higher education — 2nd cycle master's",
    },
    "Displaced": {0: "No", 1: "Yes"},
    "Educational special needs": {0: "No", 1: "Yes"},
    "Debtor": {0: "No", 1: "Yes"},
    "Tuition fees up to date": {0: "No", 1: "Yes"},
    "Gender": {0: "Female", 1: "Male"},
    "Scholarship holder": {0: "No", 1: "Yes"},
    "International": {0: "No", 1: "Yes"},
}

PREDICTION_SECTIONS = [
    (
        "Student Information",
        [
            "Marital status",
            "Nacionality",
            "Gender",
            "Age at enrollment",
            "Displaced",
            "Educational special needs",
            "International",
        ],
    ),
    (
        "Admission Information",
        [
            "Application mode",
            "Application order",
            "Course",
            "Daytime/evening attendance",
            "Admission grade",
        ],
    ),
    (
        "Previous Education",
        ["Previous qualification", "Previous qualification (grade)"],
    ),
    (
        "Family / Socioeconomic Information",
        [
            "Mother's qualification",
            "Father's qualification",
            "Mother's occupation",
            "Father's occupation",
            "Debtor",
            "Tuition fees up to date",
            "Scholarship holder",
        ],
    ),
    (
        "Other Enrollment Information",
        ["Unemployment rate", "Inflation rate", "GDP"],
    ),
]

st.set_page_config(
    page_title="Student Academic Outcome Prediction",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #102a43 0%, #176b87 100%);
        color: white;
        margin-bottom: 1.25rem;
    }
    .hero p {margin-bottom: 0; color: #e4f1f5;}
    div[data-testid="stMetric"] {
        background: #f4f8fb;
        border: 1px solid #e2eaf0;
        padding: 0.8rem;
        border-radius: 10px;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: #334e68 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #102a43 !important;
    }
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
        color: #334e68 !important;
    }
    .prediction-card {
        padding: 1.4rem 1.6rem;
        border: 1px solid #c9e2e8;
        border-left: 6px solid #176b87;
        border-radius: 12px;
        background: #f2f9fb;
        margin: 1rem 0;
    }
    .prediction-card .outcome {
        margin: 0.25rem 0 0;
        color: #102a43;
        font-size: 2rem;
        font-weight: 750;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


class _TableParser(HTMLParser):
    """Extract text rows from the HTML tables saved in the executed notebook."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        del _attrs
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self.tables.append([])
        elif self._table_depth == 1 and tag == "tr":
            self._row = []
        elif self._table_depth == 1 and tag in {"td", "th"}:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._table_depth == 1 and tag in {"td", "th"} and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif self._table_depth == 1 and tag == "tr" and self._row is not None:
            if any(self._row):
                self.tables[-1].append(self._row)
            self._row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1


@st.cache_data(show_spinner=False)
def load_dataset(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path, sep=";")


@st.cache_resource(show_spinner="Loading the fitted model...")
def load_model(path: str, modified_time: float) -> Any:
    del modified_time
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_notebook_tables(path: str, modified_time: float) -> tuple[list[list[list[str]]], list[list[list[str]]]]:
    del modified_time
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    evaluation_cell = next(
        (
            cell
            for cell in notebook.get("cells", [])
            if cell.get("id") == "heldout-model-evaluation"
        ),
        None,
    )
    cv_cell = next(
        (
            cell
            for cell in notebook.get("cells", [])
            if cell.get("id") == "training-cross-validation"
        ),
        None,
    )
    if evaluation_cell is None or cv_cell is None:
        raise ValueError("Executed evaluation/CV cells were not found in the project notebook.")

    def collect_tables(cell: dict[str, Any]) -> list[list[list[str]]]:
        parser = _TableParser()
        for output in cell.get("outputs", []):
            html_value = output.get("data", {}).get("text/html")
            if html_value:
                parser.feed("".join(html_value) if isinstance(html_value, list) else html_value)
        return parser.tables

    return collect_tables(evaluation_cell), collect_tables(cv_cell)


def _model_table(tables: list[list[list[str]]]) -> pd.DataFrame:
    metric_names = [
        "Accuracy",
        "Precision (macro)",
        "Recall (macro)",
        "F1 (macro)",
    ]
    for table in tables:
        header = next((row for row in table if set(metric_names).issubset(row)), None)
        if header is None:
            continue
        result: dict[str, dict[str, float]] = {}
        metric_indices = {metric: header.index(metric) for metric in metric_names}
        for row in table:
            if not row or row[0] not in {"Logistic Regression", "Random Forest"}:
                continue
            try:
                result[row[0]] = {
                    metric: float(row[index]) for metric, index in metric_indices.items()
                }
            except (IndexError, ValueError):
                continue
        if set(result) == {"Logistic Regression", "Random Forest"}:
            return pd.DataFrame.from_dict(result, orient="index")
    raise ValueError("Could not extract both model metric rows from saved notebook outputs.")


def _cv_table(tables: list[list[list[str]]]) -> pd.DataFrame:
    metric_names = ["CV macro F1 mean", "CV macro F1 std"]
    for table in tables:
        header = next((row for row in table if set(metric_names).issubset(row)), None)
        if header is None:
            continue
        result: dict[str, dict[str, float]] = {}
        for row in table:
            if not row or row[0] not in {"Logistic Regression", "Random Forest"}:
                continue
            try:
                result[row[0]] = {
                    metric: float(row[header.index(metric)]) for metric in metric_names
                }
            except (IndexError, ValueError):
                continue
        if set(result) == {"Logistic Regression", "Random Forest"}:
            return pd.DataFrame.from_dict(result, orient="index")
    raise ValueError("Could not extract training-only CV results from saved notebook outputs.")


def _confusion_table(tables: list[list[list[str]]]) -> pd.DataFrame:
    expected = set(CLASS_LABELS)
    random_forest_matrices: list[pd.DataFrame] = []
    for table in tables:
        header = next((row for row in table if expected.issubset(row)), None)
        if header is None:
            continue
        result: dict[str, list[int]] = {}
        for row in table:
            if row and row[0] in expected:
                try:
                    result[row[0]] = [int(row[header.index(label)]) for label in CLASS_LABELS]
                except (IndexError, ValueError):
                    continue
        if set(result) == expected:
            random_forest_matrices.append(
                pd.DataFrame.from_dict(result, orient="index", columns=CLASS_LABELS).reindex(CLASS_LABELS)
            )
    if random_forest_matrices:
        # Evaluation output is ordered Logistic Regression then Random Forest.
        return random_forest_matrices[-1]
    raise ValueError("Could not extract the Random Forest confusion matrix from notebook outputs.")


def _feature_importance(model: Any) -> pd.Series:
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        raise TypeError("The saved final estimator does not expose tree feature importances.")

    transformed_names = preprocessor.get_feature_names_out()
    raw_names = list(model.feature_names_in_)
    aggregated: dict[str, float] = {name: 0.0 for name in raw_names}
    for transformed_name, value in zip(transformed_names, estimator.feature_importances_):
        clean_name = transformed_name.split("__", 1)[-1]
        original_name = next(
            (
                name
                for name in raw_names
                if clean_name == name or clean_name.startswith(f"{name}_")
            ),
            None,
        )
        if original_name is None:
            raise ValueError(f"Could not map transformed predictor {transformed_name!r}.")
        aggregated[original_name] += float(value)
    return pd.Series(aggregated, name="Importance").sort_values(ascending=False)


def _validate_loaded_artifacts(df: pd.DataFrame, model: Any) -> tuple[list[str], list[str]]:
    feature_names = list(model.feature_names_in_)
    if len(feature_names) != 24:
        raise ValueError(f"Expected 24 saved model features, found {len(feature_names)}.")
    leaked = sorted(set(feature_names).intersection(LEAKAGE_COLUMNS))
    if leaked:
        raise ValueError(f"The saved model unexpectedly includes leakage features: {leaked}")
    missing = sorted(set(feature_names).difference(df.columns))
    if missing:
        raise ValueError(f"Model features missing from the supplied dataset: {missing}")

    preprocessor = model.named_steps.get("preprocess")
    if preprocessor is None:
        raise ValueError("The saved model does not contain its expected preprocessing pipeline.")
    estimator = model.named_steps.get("model")
    if estimator is None or estimator.__class__.__name__ != "RandomForestClassifier":
        raise ValueError("The saved final estimator is not the expected Random Forest model.")
    categorical: list[str] = []
    numerical: list[str] = []
    for transformer_name, _, columns in preprocessor.transformers_:
        if transformer_name == "categorical":
            categorical = list(columns)
        elif transformer_name == "numeric":
            numerical = list(columns)
    if set(categorical + numerical) != set(feature_names):
        raise ValueError("Could not match saved pipeline input columns to its transformers.")
    return categorical, numerical


def _format_feature_label(feature: str) -> str:
    labels = {
        "Nacionality": "Nationality",
        "Daytime/evening attendance": "Attendance schedule",
        "Previous qualification (grade)": "Previous qualification grade",
        "Mother's qualification": "Mother's education",
        "Father's qualification": "Father's education",
        "Mother's occupation": "Mother's occupation category",
        "Father's occupation": "Father's occupation category",
        "Tuition fees up to date": "Tuition fees up to date",
    }
    return labels.get(feature, feature.replace("_", " "))


def _category_label(feature: str, code: int) -> str:
    known_label = UCI_CATEGORY_LABELS.get(feature, {}).get(code)
    if known_label is not None:
        return f"{known_label} (code {code})"
    if feature.endswith("occupation"):
        return f"Occupation category code {code} (UCI label unavailable)"
    return f"Education category code {code} (UCI label unavailable)"


def _render_overview(df: pd.DataFrame, model: Any) -> None:
    st.markdown(
        """
        <div class="hero">
            <h2>Student Academic Outcome Prediction</h2>
            <p>Gupio Campus Placement AI/ML Assignment – Option 1</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Predict academic outcomes from enrollment-time information")
    st.write(
        "This project predicts whether a student will Dropout, remain Enrolled, or Graduate "
        "using only information intended to be available at enrollment."
    )
    st.write(
        "The objective is to provide an interpretable, leakage-aware demonstration of "
        "multiclass classification and class-level model evaluation."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students", f"{len(df):,}")
    col2.metric("Original columns", f"{df.shape[1]}")
    col3.metric("Primary model features", f"{len(model.feature_names_in_)}")
    col4.metric("Leakage features excluded", f"{len(LEAKAGE_COLUMNS)}")

    st.markdown("**Target classes**")
    st.write("Dropout · Enrolled · Graduate")
    st.info(
        "All 12 first- and second-semester performance variables are excluded: they are "
        "unavailable at enrollment and would cause prediction-time leakage."
    )


def _render_dataset(df: pd.DataFrame) -> None:
    st.subheader("Dataset overview")
    a, b, c = st.columns(3)
    a.metric("Rows × columns", f"{df.shape[0]:,} × {df.shape[1]}")
    b.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    c.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    counts = df[TARGET_COLUMN].value_counts().reindex(CLASS_LABELS, fill_value=0)
    percentages = (counts / len(df) * 100).rename("Percentage")
    distribution = pd.DataFrame({"Students": counts, "Percentage": percentages})
    st.markdown("#### Target distribution")
    st.bar_chart(distribution["Students"], color="#176b87")
    st.dataframe(distribution, width="stretch")

    st.markdown("#### Missing-value counts")
    missing = df.isna().sum().rename("Missing values")
    st.dataframe(missing.to_frame(), width="stretch")
    st.caption(f"Exact duplicate records: {df.duplicated().sum():,}")

    st.markdown("#### Enrollment-time feature vs target")
    left, right = st.columns(2)
    tuition = pd.crosstab(
        df["Tuition fees up to date"], df[TARGET_COLUMN], normalize="index"
    ).reindex(columns=CLASS_LABELS, fill_value=0)
    left.markdown("**Outcome share by tuition-fee status**")
    left.bar_chart(tuition, stack=True, color=["#d95f59", "#e6b655", "#4b9b72"])
    left.caption("Rows show observed outcome proportions within each tuition-status code.")

    age_data = df[["Age at enrollment", TARGET_COLUMN]].copy()
    age_data["Age band"] = pd.qcut(age_data["Age at enrollment"], q=4, duplicates="drop")
    age_share = pd.crosstab(
        age_data["Age band"], age_data[TARGET_COLUMN], normalize="index"
    ).reindex(columns=CLASS_LABELS, fill_value=0)
    age_share.index = age_share.index.astype(str)
    right.markdown("**Outcome share by enrollment-age quartile**")
    right.bar_chart(age_share, stack=True, color=["#d95f59", "#e6b655", "#4b9b72"])
    right.caption("Age bands are calculated from the actual dataset; values are descriptive, not causal.")


def _render_performance(notebook_path: Path) -> None:
    st.subheader("Training-only model selection")
    try:
        evaluation_tables, cv_tables = load_notebook_tables(
            str(notebook_path), notebook_path.stat().st_mtime
        )
        cv_results = _cv_table(cv_tables)
        test_metrics = _model_table(evaluation_tables)
        confusion = _confusion_table(evaluation_tables)
    except (OSError, json.JSONDecodeError, StopIteration, ValueError) as error:
        st.error(
            "Could not read actual saved model evaluation outputs from the notebook. "
            f"Metrics are not displayed: {error}"
        )
        return

    st.caption("Actual results parsed from the executed notebook; no metrics are recomputed or fabricated.")
    st.markdown("**Five-fold stratified CV macro F1 (training data only)**")
    st.dataframe(cv_results.style.format("{:.4f}"), width="stretch")
    selected = cv_results["CV macro F1 mean"].idxmax()
    st.success(f"Selected using training-only CV macro F1: **{selected}**")

    st.markdown("**Held-out test-set comparison**")
    st.dataframe(test_metrics.style.format("{:.4f}"), width="stretch")
    st.markdown("**Final Random Forest confusion matrix**")
    st.caption("Rows are actual labels; columns are predicted labels.")
    st.dataframe(confusion, width="stretch")
    st.bar_chart(confusion, color=["#176b87", "#e6b655", "#4b9b72"])
    st.warning(
        "Class frequencies are uneven (Enrolled is the smallest class). The Random Forest "
        "has lower Enrolled recall than Logistic Regression, so accuracy alone is not a "
        "sufficient summary."
    )


def _render_importance(model: Any) -> None:
    st.subheader("Random Forest feature importance")
    importance = _feature_importance(model).head(15).sort_values()
    st.bar_chart(importance, horizontal=True, color="#176b87")
    st.dataframe(
        importance.sort_values(ascending=False).rename("Importance").to_frame().style.format("{:.4f}"),
        width="stretch",
    )
    st.info("Feature importance indicates predictive association and does not prove causation.")


def _render_prediction(df: pd.DataFrame, model: Any, categorical_features: list[str]) -> None:
    st.subheader("Predict a student's academic outcome")
    st.caption(
        "Complete the student and enrollment fields. The 12 first- and second-semester "
        "performance variables are intentionally excluded to prevent data leakage."
    )
    feature_names = list(model.feature_names_in_)
    st.info(
        "Prediction uses enrollment-time information only. First- and second-semester "
        "academic-performance variables are intentionally excluded to prevent data leakage."
    )
    numeric_features = set(feature_names).difference(categorical_features)
    section_features = {
        feature for _, section in PREDICTION_SECTIONS for feature in section
    }
    if section_features != set(feature_names):
        st.error("The prediction form sections do not match the saved model's feature list.")
        return

    submitted_values: dict[str, Any] = {}
    with st.form("student_prediction_form"):
        for section_title, section_columns in PREDICTION_SECTIONS:
            st.markdown(f"#### {section_title}")
            columns = st.columns(2)
            for index, feature in enumerate(section_columns):
                container = columns[index % 2]
                label = _format_feature_label(feature)
                series = df[feature].dropna()
                if feature in categorical_features:
                    codes = sorted(int(value) for value in series.unique())
                    if not codes:
                        st.error(f"No available category values for {label}.")
                        st.stop()
                    label_to_code = {
                        _category_label(feature, code): code for code in codes
                    }
                    selected_label = container.selectbox(
                        label,
                        options=list(label_to_code),
                        key=f"predict_{feature}",
                    )
                    submitted_values[feature] = label_to_code[selected_label]
                elif feature in numeric_features:
                    if series.empty:
                        st.error(f"No numeric values are available for {label}.")
                        st.stop()
                    minimum = float(series.min())
                    maximum = float(series.max())
                    default = float(series.median())
                    if pd.api.types.is_integer_dtype(series.dtype):
                        submitted_values[feature] = container.number_input(
                            label,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(round(default)),
                            step=1,
                            key=f"predict_{feature}",
                        )
                    else:
                        step = max((maximum - minimum) / 100, 0.01)
                        submitted_values[feature] = container.number_input(
                            label,
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            step=step,
                            format="%.4f",
                            key=f"predict_{feature}",
                        )
                else:
                    st.error(f"No input control is configured for {label}.")
                    st.stop()
        submitted = st.form_submit_button(
            "Predict Outcome", type="primary", width="stretch"
        )

    if submitted:
        missing_values = [feature for feature in feature_names if feature not in submitted_values]
        if missing_values:
            st.error(f"Please complete all required fields: {missing_values}")
            return
        try:
            input_frame = pd.DataFrame(
                [[submitted_values[feature] for feature in feature_names]],
                columns=feature_names,
            )
        except (KeyError, TypeError, ValueError) as error:
            st.error(f"Could not validate the submitted student information: {error}")
            return

        leaked_features = set(input_frame.columns).intersection(LEAKAGE_COLUMNS)
        if leaked_features:
            st.error(f"Prediction blocked: leakage features detected: {sorted(leaked_features)}")
            return
        if list(input_frame.columns) != feature_names:
            st.error("Prediction blocked: input feature names or order do not match the saved model.")
            return
        for feature in categorical_features:
            allowed_codes = set(int(value) for value in df[feature].dropna().unique())
            if int(input_frame.iloc[0][feature]) not in allowed_codes:
                st.error(f"Invalid encoded category selected for {_format_feature_label(feature)}.")
                return
        for feature in numeric_features:
            value = float(input_frame.iloc[0][feature])
            bounds = df[feature].dropna()
            if not pd.notna(value) or value < float(bounds.min()) or value > float(bounds.max()):
                st.error(f"Enter a valid in-range value for {_format_feature_label(feature)}.")
                return

        try:
            prediction = str(model.predict(input_frame)[0])
        except (AttributeError, TypeError, ValueError) as error:
            st.error(f"The saved model could not process this input: {error}")
            return
        if prediction not in CLASS_LABELS:
            st.error(f"The saved model returned an unexpected outcome: {prediction!r}")
            return

        st.markdown(
            f"""
            <div class="prediction-card">
                <div>🎓 Predicted Academic Outcome</div>
                <p class="outcome">{prediction.upper()}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            "Based on the information provided at enrollment, the model predicts that "
            f"this student is most likely to be: **{prediction.upper()}**."
        )
        st.caption(
            "This is a machine-learning prediction, not a guarantee of the student's future outcome."
        )
        if callable(getattr(model, "predict_proba", None)):
            try:
                probabilities = model.predict_proba(input_frame)[0]
            except (AttributeError, TypeError, ValueError) as error:
                st.error(f"The saved model could not calculate class probabilities: {error}")
                return
            probability_table = pd.DataFrame(
                {"Probability": probabilities},
                index=model.classes_,
            ).reindex(CLASS_LABELS)
            st.markdown("#### Model class probabilities")
            st.dataframe(
                probability_table.style.format("{:.1%}"),
                width="stretch",
            )
            st.bar_chart(probability_table, color="#176b87")


def _render_methodology() -> None:
    st.subheader("Methodology")
    st.markdown(
        """
        **Dataset**  
        ↓  
        **Inspection**  
        ↓  
        **Leakage prevention**  
        ↓  
        **EDA**  
        ↓  
        **Stratified train/test split**  
        ↓  
        **Leakage-safe preprocessing pipeline**  
        ↓  
        **Logistic Regression baseline**  
        ↓  
        **Random Forest**  
        ↓  
        **Training-only cross-validation (macro F1)**  
        ↓  
        **Held-out test evaluation**  
        ↓  
        **Final model selection**  
        ↓  
        **Prediction**
        """
    )
    st.info(
        "The 12 first- and second-semester performance variables were excluded from the "
        "primary model because they are not available at enrollment."
    )
    st.write(
        "The saved artifact is the fitted preprocessing-and-model pipeline. The application "
        "loads it with joblib and does not retrain or modify it."
    )


def main() -> None:
    st.title("Student Academic Outcome Prediction")
    st.caption("Gupio Campus Placement AI/ML Assignment – Option 1")
    st.sidebar.title("Project navigation")
    section = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Dataset & EDA",
            "Model Performance",
            "Feature Importance",
            "Student Prediction",
            "Methodology",
        ],
    )

    if not DATA_PATH.is_file():
        st.error(f"Dataset file not found: {DATA_PATH}")
        st.stop()
    if not MODEL_PATH.is_file():
        st.error(
            f"Saved model not found: {MODEL_PATH}. The app requires the existing fitted "
            "pipeline and does not train a replacement."
        )
        st.stop()

    try:
        df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        categorical_features, _ = _validate_loaded_artifacts(df, model)
    except (OSError, ValueError, EOFError, TypeError) as error:
        st.error(f"Could not load or validate the supplied project artifacts: {error}")
        st.stop()

    if section == "Overview":
        _render_overview(df, model)
    elif section == "Dataset & EDA":
        _render_dataset(df)
    elif section == "Model Performance":
        if not NOTEBOOK_PATH.is_file():
            st.error(f"Executed notebook results were not found: {NOTEBOOK_PATH}")
        else:
            _render_performance(NOTEBOOK_PATH)
    elif section == "Feature Importance":
        try:
            _render_importance(model)
        except (AttributeError, TypeError, ValueError) as error:
            st.error(f"Could not calculate importance from the saved Random Forest: {error}")
    elif section == "Student Prediction":
        _render_prediction(df, model, categorical_features)
    else:
        _render_methodology()


if __name__ == "__main__":
    main()
