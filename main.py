
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import plotly.express as px

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EduShield AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# TARGET MAPPING
# ============================================================

TARGET_MAPPING = {
    "Dropout": 0,
    "Graduate": 1,
    "Enrolled": 2,
}

REVERSE_TARGET_MAPPING = {
    0: "Dropout",
    1: "Graduate",
    2: "Enrolled",
}

VALID_TARGETS = set(TARGET_MAPPING.keys())


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state():
    defaults = {
        "model": None,
        "model_trained": False,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "feature_names": [],
        "processed_df": None,
        "raw_df": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# DATA HELPERS
# ============================================================

def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def detect_target_column(df):
    for column in df.columns:
        if str(column).strip().lower() == "target":
            return column
    return None


def load_csv_safely(uploaded_file):
    """
    Supports:
      - UCI-style semicolon CSV
      - comma-separated CSV
      - tab-separated CSV

    The UCI Student Dropout dataset is commonly semicolon-separated.
    """

    separators = [";", ",", "\t"]
    last_error = None

    for separator in separators:
        try:
            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                sep=separator,
                encoding="utf-8-sig",
            )

            if len(df.columns) <= 1:
                continue

            df = clean_column_names(df)

            target_column = detect_target_column(df)

            if target_column is not None and target_column != "Target":
                df = df.rename(
                    columns={target_column: "Target"}
                )

            return df

        except Exception as exc:
            last_error = exc

    raise ValueError(
        f"Unable to read CSV file. {last_error}"
    )


def validate_dataset(df):
    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    df = clean_column_names(df)

    if "Target" not in df.columns:
        raise ValueError(
            "Required column 'Target' was not found."
        )

    df["Target"] = (
        df["Target"]
        .astype(str)
        .str.strip()
    )

    actual_targets = set(
        df["Target"].dropna().unique()
    )

    invalid_targets = actual_targets - VALID_TARGETS

    if invalid_targets:
        raise ValueError(
            "Invalid Target values found: "
            f"{sorted(invalid_targets)}. "
            "Expected: Dropout, Graduate, Enrolled."
        )

    return df


def make_sample_dataset():
    """Fallback only when no dataset is supplied."""
    return pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 5],
        "Marital status": [1, 1, 1, 2, 1, 1],
        "Application mode": [17, 17, 1, 39, 17, 18],
        "Application order": [5, 1, 1, 1, 2, 1],
        "Course": [171, 171, 9254, 9773, 171, 8014],
        "Daytime/evening attendance": [1, 1, 1, 1, 1, 1],
        "Previous qualification": [1, 1, 1, 1, 1, 1],
        "Previous qualification (grade)": [126, 125, 137, 120, 130, 118],
        "Nacionality": [1, 1, 1, 1, 1, 1],
        "Mother's qualification": [1, 19, 3, 1, 13, 19],
        "Father's qualification": [19, 19, 19, 1, 19, 19],
        "Mother's occupation": [5, 9, 2, 5, 4, 9],
        "Father's occupation": [5, 9, 3, 5, 4, 9],
        "Admission grade": [122.6, 119.8, 144.7, 120.0, 140.0, 118.0],
        "Displaced": [0, 1, 0, 0, 0, 1],
        "Educational special needs": [0, 0, 0, 0, 0, 0],
        "Debtor": [0, 0, 0, 0, 0, 1],
        "Tuition fees up to date": [1, 1, 1, 1, 1, 0],
        "Gender": [0, 0, 1, 1, 0, 1],
        "Scholarship holder": [1, 0, 0, 1, 1, 0],
        "Age at enrollment": [18, 18, 19, 21, 20, 25],
        "International": [0, 0, 0, 0, 0, 0],
        "Curricular units 1st sem (credited)": [0, 0, 0, 0, 0, 0],
        "Curricular units 1st sem (enrolled)": [6, 6, 6, 6, 6, 6],
        "Curricular units 1st sem (evaluations)": [6, 8, 6, 6, 6, 5],
        "Curricular units 1st sem (approved)": [6, 4, 0, 3, 5, 1],
        "Curricular units 1st sem (grade)": [14.5, 11.6, 0.0, 10.5, 13.5, 8.0],
        "Curricular units 1st sem (without evaluations)": [0, 0, 0, 0, 0, 1],
        "Curricular units 2nd sem (credited)": [0, 0, 0, 0, 0, 0],
        "Curricular units 2nd sem (enrolled)": [6, 6, 6, 6, 6, 6],
        "Curricular units 2nd sem (evaluations)": [7, 9, 0, 6, 7, 4],
        "Curricular units 2nd sem (approved)": [6, 0, 0, 4, 5, 1],
        "Curricular units 2nd sem (grade)": [12.43, 0.0, 0.0, 11.0, 13.0, 7.5],
        "Curricular units 2nd sem (without evaluations)": [0, 0, 0, 0, 0, 2],
        "Unemployment rate": [11.1, 11.1, 16.2, 10.8, 12.0, 15.0],
        "Inflation rate": [0.6, 0.6, 0.3, 1.0, 0.5, 0.8],
        "GDP": [2.02, 2.02, -0.92, 1.5, 1.2, 0.2],
        "Target": [
            "Graduate",
            "Dropout",
            "Dropout",
            "Enrolled",
            "Graduate",
            "Dropout",
        ],
    })


def load_data(uploaded_file=None):
    """
    Load uploaded data.

    If no file is uploaded, try common local paths and finally use
    a tiny fallback dataset.
    """

    if uploaded_file is not None:
        try:
            df = load_csv_safely(uploaded_file)
            df = validate_dataset(df)

            st.success(
                f"✅ Successfully loaded uploaded file! "
                f"{df.shape[0]:,} students × {df.shape[1]} columns"
            )

            return df

        except Exception as exc:
            st.error(f"❌ Error loading uploaded CSV: {exc}")

            if "df" in locals():
                st.write("Detected columns:")
                st.write(df.columns.tolist())

            st.stop()

    local_paths = [
        "./data/student_dropout_data.csv",
        "../data/student_dropout_data.csv",
        "./student_dropout_data.csv",
        "./data.csv",
    ]

    for path in local_paths:
        try:
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
            )
            df = clean_column_names(df)

            if "Target" in df.columns:
                df = validate_dataset(df)
                st.success(
                    f"Successfully loaded data from {path}"
                )
                return df

        except Exception:
            continue

    st.warning(
        "⚠️ No dataset uploaded. A small sample dataset is being used."
    )

    return make_sample_dataset()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_data(df):
    """
    Preprocess data for the Random Forest model.

    Important:
    - Target remains separate.
    - ID is removed.
    - Numeric missing values use median.
    - Categorical features are label encoded.
    - Target is mapped to 0/1/2.
    """

    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    processed_df = clean_column_names(df)

    if "Target" not in processed_df.columns:
        raise ValueError(
            "Dataset must contain a 'Target' column."
        )

    processed_df["Target"] = (
        processed_df["Target"]
        .astype(str)
        .str.strip()
    )

    invalid_targets = (
        set(processed_df["Target"].dropna().unique())
        - VALID_TARGETS
    )

    if invalid_targets:
        raise ValueError(
            f"Invalid Target values: {sorted(invalid_targets)}"
        )

    # Remove ID because it is not a useful predictive feature.
    if "id" in processed_df.columns:
        processed_df = processed_df.drop(
            columns=["id"]
        )

    # Identify feature columns before target encoding.
    feature_columns = [
        col
        for col in processed_df.columns
        if col != "Target"
    ]

    # Try to convert columns to numeric ONLY when all non-null
    # values can be converted. This avoids pandas errors and
    # preserves genuine categorical columns.
    for col in feature_columns:
        original = processed_df[col]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        non_empty = original.notna()

        if non_empty.any() and converted[non_empty].notna().all():
            processed_df[col] = converted

    # Fill numeric missing values.
    numeric_columns = processed_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    for col in numeric_columns:
        if processed_df[col].isnull().any():
            median_value = processed_df[col].median()

            if pd.isna(median_value):
                median_value = 0

            processed_df[col] = (
                processed_df[col]
                .fillna(median_value)
            )

    # Encode remaining categorical feature columns.
    categorical_columns = processed_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    categorical_columns = [
        col
        for col in categorical_columns
        if col != "Target"
    ]

    for col in categorical_columns:
        processed_df[col] = (
            processed_df[col]
            .astype(str)
            .fillna("Unknown")
        )

        encoder = LabelEncoder()

        processed_df[col] = encoder.fit_transform(
            processed_df[col]
        )

    # Encode target.
    processed_df["Target"] = (
        processed_df["Target"]
        .map(TARGET_MAPPING)
    )

    if processed_df["Target"].isnull().any():
        raise ValueError(
            "Target mapping failed. "
            "Expected Dropout, Graduate, Enrolled."
        )

    # Final numeric conversion for model features.
    feature_columns = [
        col
        for col in processed_df.columns
        if col != "Target"
    ]

    for col in feature_columns:
        processed_df[col] = pd.to_numeric(
            processed_df[col],
            errors="coerce",
        )

        if processed_df[col].isnull().any():
            median_value = processed_df[col].median()

            if pd.isna(median_value):
                median_value = 0

            processed_df[col] = (
                processed_df[col]
                .fillna(median_value)
            )

    processed_df = processed_df.dropna()

    if processed_df.empty:
        raise ValueError(
            "No valid rows remain after preprocessing."
        )

    X = processed_df.drop(
        columns=["Target"]
    )

    y = processed_df["Target"].astype(int)

    return X, y, processed_df


# ============================================================
# DATA OVERVIEW
# ============================================================

def display_data_quality(df):
    st.markdown("### 🔍 Data Quality Assessment")

    col1, col2 = st.columns(2)

    with col1:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(
            ascending=False
        )

        if len(missing):
            fig, ax = plt.subplots(figsize=(8, 5))
            missing.plot(kind="bar", ax=ax)
            ax.set_title("Missing Data Across Columns")
            ax.set_ylabel("Missing Values")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.success("✅ No missing data found.")

    with col2:
        dtype_counts = df.dtypes.astype(str).value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(
            dtype_counts.values,
            labels=dtype_counts.index,
            autopct="%1.1f%%",
        )
        ax.set_title("Data Type Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def display_demographics(df):
    st.markdown("### 👥 Student Demographics")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "Age at enrollment" in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(
                pd.to_numeric(
                    df["Age at enrollment"],
                    errors="coerce"
                ).dropna(),
                bins=20,
            )
            ax.set_title("Age at Enrollment")
            ax.set_xlabel("Age")
            ax.set_ylabel("Students")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        if "Gender" in df.columns:
            counts = df["Gender"].value_counts()
            labels = [
                "Female" if int(v) == 0 else "Male"
                for v in counts.index
            ]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.pie(
                counts.values,
                labels=labels,
                autopct="%1.1f%%",
            )
            ax.set_title("Gender Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col3:
        if "Scholarship holder" in df.columns:
            counts = df["Scholarship holder"].value_counts()

            labels = [
                "No Scholarship" if int(v) == 0 else "Scholarship"
                for v in counts.index
            ]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.pie(
                counts.values,
                labels=labels,
                autopct="%1.1f%%",
            )
            ax.set_title("Scholarship Status")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


def display_academic_performance(df):
    st.markdown("### 📚 Academic Performance")

    academic_columns = [
        "Admission grade",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
    ]

    available = [
        col
        for col in academic_columns
        if col in df.columns
    ]

    if not available:
        st.info("No academic performance columns found.")
        return

    for col in available:
        fig, ax = plt.subplots(figsize=(9, 4))

        for outcome in [
            "Dropout",
            "Enrolled",
            "Graduate",
        ]:
            values = pd.to_numeric(
                df.loc[
                    df["Target"] == outcome,
                    col
                ],
                errors="coerce",
            ).dropna()

            if len(values):
                ax.hist(
                    values,
                    bins=15,
                    alpha=0.5,
                    label=outcome,
                )

        ax.set_title(f"{col} by Student Outcome")
        ax.set_xlabel(col)
        ax.set_ylabel("Students")
        ax.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def display_performance_summary(df):
    st.markdown("### 📈 Performance Summary")

    col1, col2, col3 = st.columns(3)

    if "Admission grade" in df.columns:
        graduate = pd.to_numeric(
            df.loc[
                df["Target"] == "Graduate",
                "Admission grade"
            ],
            errors="coerce",
        ).mean()

        dropout = pd.to_numeric(
            df.loc[
                df["Target"] == "Dropout",
                "Admission grade"
            ],
            errors="coerce",
        ).mean()

        graduate = 0 if pd.isna(graduate) else graduate
        dropout = 0 if pd.isna(dropout) else dropout

        col1.metric(
            "Graduate Admission Grade",
            f"{graduate:.1f}",
            f"{graduate - dropout:.1f} vs Dropout",
        )

    if "Curricular units 1st sem (grade)" in df.columns:
        graduate = pd.to_numeric(
            df.loc[
                df["Target"] == "Graduate",
                "Curricular units 1st sem (grade)"
            ],
            errors="coerce",
        ).mean()

        dropout = pd.to_numeric(
            df.loc[
                df["Target"] == "Dropout",
                "Curricular units 1st sem (grade)"
            ],
            errors="coerce",
        ).mean()

        graduate = 0 if pd.isna(graduate) else graduate
        dropout = 0 if pd.isna(dropout) else dropout

        col2.metric(
            "Graduate 1st Sem Grade",
            f"{graduate:.1f}",
            f"{graduate - dropout:.1f} vs Dropout",
        )

        values = pd.to_numeric(
            df["Curricular units 1st sem (grade)"],
            errors="coerce",
        )

        high = df.loc[
            values > 15
        ]

        success_rate = (
            (high["Target"] == "Graduate").mean() * 100
            if len(high)
            else 0
        )

        col3.metric(
            "Success Rate (Grade > 15)",
            f"{success_rate:.1f}%",
        )


def display_risk_factors(df):
    st.markdown("### ⚠️ Key Risk Factors")

    risk_factors = []

    if "Curricular units 1st sem (approved)" in df.columns:
        values = pd.to_numeric(
            df["Curricular units 1st sem (approved)"],
            errors="coerce",
        )

        risk_factors.append(
            (
                "Low 1st Sem Approved Units",
                int((values <= 2).sum()),
            )
        )

    if "Age at enrollment" in df.columns:
        values = pd.to_numeric(
            df["Age at enrollment"],
            errors="coerce",
        )

        risk_factors.append(
            (
                "Age > 25",
                int((values > 25).sum()),
            )
        )

    if "Scholarship holder" in df.columns:
        values = pd.to_numeric(
            df["Scholarship holder"],
            errors="coerce",
        )

        risk_factors.append(
            (
                "No Scholarship",
                int((values == 0).sum()),
            )
        )

    if "Displaced" in df.columns:
        values = pd.to_numeric(
            df["Displaced"],
            errors="coerce",
        )

        risk_factors.append(
            (
                "Displaced",
                int((values == 1).sum()),
            )
        )

    if not risk_factors:
        st.info("No supported risk-factor columns found.")
        return

    risk_df = pd.DataFrame(
        risk_factors,
        columns=[
            "Risk Factor",
            "Number of Students",
        ],
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        risk_df["Risk Factor"],
        risk_df["Number of Students"],
    )
    ax.set_title("Students by Risk Factor")
    ax.set_ylabel("Number of Students")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.dataframe(
        risk_df,
        use_container_width=True,
    )


def display_data_structure(df):
    st.markdown("### 📋 Data Sample & Structure")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(
            df.head(10),
            use_container_width=True,
        )

    with col2:
        st.metric(
            "Rows",
            f"{len(df):,}",
        )
        st.metric(
            "Columns",
            f"{len(df.columns):,}",
        )

        st.write("### Target Distribution")

        st.dataframe(
            df["Target"]
            .value_counts()
            .rename("Students"),
            use_container_width=True,
        )


def display_feature_explorer(df):
    st.markdown("### 🔧 Interactive Feature Explorer")

    numeric_cols = [
        col
        for col in df.select_dtypes(
            include=np.number
        ).columns
        if col != "id"
    ]

    if not numeric_cols:
        st.info("No numeric features available.")
        return

    feature = st.selectbox(
        "Select a feature",
        numeric_cols,
        key="feature_explorer",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write(df[feature].describe())

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            df[feature].dropna(),
            bins=25,
        )
        ax.set_title(
            f"Distribution of {feature}"
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        temp = df[
            [feature, "Target"]
        ].copy()

        temp["Target Code"] = (
            temp["Target"]
            .map(TARGET_MAPPING)
        )

        correlation = temp[
            feature
        ].corr(
            temp["Target Code"]
        )

        st.metric(
            "Correlation with Target Code",
            "N/A"
            if pd.isna(correlation)
            else f"{correlation:.3f}",
        )


def display_column_info(df):
    st.markdown("### 📋 Detailed Column Information")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Overview",
            "Statistics",
            "Missing Data",
            "Data Types",
        ]
    )

    with tab1:
        st.write("Rows:", len(df))
        st.write("Columns:", len(df.columns))
        st.write(df.columns.tolist())

    with tab2:
        st.dataframe(
            df.describe(
                include="all"
            ).transpose(),
            use_container_width=True,
        )

    with tab3:
        missing = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": df.isnull().sum().values,
            "Missing %": (
                df.isnull().mean().values * 100
            ).round(2),
        })

        st.dataframe(
            missing[
                missing["Missing Count"] > 0
            ],
            use_container_width=True,
        )

    with tab4:
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Null": df.isnull().sum().values,
        })

        st.dataframe(
            dtype_df,
            use_container_width=True,
        )


# ============================================================
# EDA
# ============================================================

def display_eda(df):
    st.markdown("## 📈 Exploratory Data Analysis")

    sub_menu = st.selectbox(
        "Select Analysis",
        [
            "Distribution Analysis",
            "Correlation Analysis",
            "Outcome Analysis",
            "Academic Performance",
        ],
        key="eda_menu",
    )

    if sub_menu == "Distribution Analysis":
        numeric_cols = [
            col
            for col in df.select_dtypes(
                include=np.number
            ).columns
            if col != "id"
        ]

        default_features = [
            col
            for col in [
                "Age at enrollment",
                "Admission grade",
                "Curricular units 1st sem (grade)",
            ]
            if col in numeric_cols
        ]

        selected = st.multiselect(
            "Select up to 4 features",
            numeric_cols,
            default=default_features,
            max_selections=4,
        )

        if not selected:
            st.info("Select at least one feature.")
            return

        for i in range(0, len(selected), 2):
            columns = st.columns(
                min(2, len(selected) - i)
            )

            for j, feature in enumerate(
                selected[i:i + 2]
            ):
                with columns[j]:
                    fig, ax = plt.subplots(
                        figsize=(8, 4)
                    )

                    sns.histplot(
                        df[feature].dropna(),
                        kde=True,
                        ax=ax,
                    )

                    ax.set_title(
                        f"Distribution of {feature}"
                    )

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

    elif sub_menu == "Correlation Analysis":
        numeric_cols = [
            col
            for col in df.select_dtypes(
                include=np.number
            ).columns
            if col != "id"
        ]

        selected = st.multiselect(
            "Select features",
            numeric_cols,
            default=[
                col
                for col in [
                    "Age at enrollment",
                    "Admission grade",
                    "Curricular units 1st sem (grade)",
                    "Curricular units 2nd sem (grade)",
                ]
                if col in numeric_cols
            ],
        )

        if len(selected) < 2:
            st.info("Select at least two features.")
            return

        corr = df[selected].corr()

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
        )

        ax.set_title(
            "Feature Correlation Matrix"
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    elif sub_menu == "Outcome Analysis":
        col1, col2 = st.columns(2)

        with col1:
            counts = df["Target"].value_counts()

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
            )

            ax.set_title(
                "Student Outcome Distribution"
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            total = len(df)

            for outcome in [
                "Dropout",
                "Enrolled",
                "Graduate",
            ]:
                count = int(
                    (
                        df["Target"] == outcome
                    ).sum()
                )

                st.metric(
                    outcome,
                    f"{count:,}",
                    f"{count / total * 100:.1f}%",
                )

    else:
        display_academic_performance(df)


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(X_train, y_train):
    progress = st.progress(0)
    status = st.empty()

    status.write("Starting model training...")
    progress.progress(20)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    status.write(
        "Fitting Random Forest..."
    )
    progress.progress(50)

    model.fit(
        X_train,
        y_train,
    )

    progress.progress(100)
    status.success(
        "Model training complete!"
    )

    return model


def visualize_model_results(
    model,
    X_test,
    y_test,
):
    st.markdown(
        "## 📊 Model Evaluation Results"
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    st.metric(
        "Model Accuracy",
        f"{accuracy:.2%}",
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Classification Report",
            "Confusion Matrix",
            "Feature Importance",
        ]
    )

    with tab1:
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df,
            use_container_width=True,
        )

    with tab2:
        cm = confusion_matrix(
            y_test,
            y_pred,
        )

        labels = [
            REVERSE_TARGET_MAPPING.get(
                int(c),
                str(c),
            )
            for c in sorted(
                np.unique(
                    np.concatenate(
                        [
                            y_test.values,
                            y_pred,
                        ]
                    )
                )
            )
        ]

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            "Confusion Matrix"
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": model.feature_importances_,
        }).sort_values(
            "Importance",
            ascending=False,
        )

        top = importance_df.head(15)

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        sns.barplot(
            data=top,
            x="Importance",
            y="Feature",
            ax=ax,
        )

        ax.set_title(
            "Top 15 Feature Importances"
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(
            importance_df,
            use_container_width=True,
        )


# ============================================================
# SHAP
# ============================================================

def normalize_shap_values(
    explainer,
    X_data,
    class_index,
):
    values = explainer.shap_values(
        X_data
    )

    if isinstance(values, list):
        class_index = min(
            class_index,
            len(values) - 1,
        )
        return np.asarray(
            values[class_index]
        )

    values = np.asarray(values)

    if values.ndim == 3:
        # samples x features x classes
        return values[
            :,
            :,
            class_index,
        ]

    if values.ndim == 2:
        return values

    if values.ndim == 1:
        return values.reshape(
            1,
            -1,
        )

    raise ValueError(
        f"Unsupported SHAP output shape: {values.shape}"
    )


def display_global_feature_importance(
    model,
    X_train,
    X_test,
    y_test,
    feature_names,
):
    st.markdown(
        "### 🌍 Global Feature Importance"
    )

    method = st.selectbox(
        "Select Importance Method",
        [
            "Built-in Feature Importance",
            "SHAP Global Importance",
            "Permutation Importance",
        ],
        key="importance_method",
    )

    try:
        if method == "Built-in Feature Importance":
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_,
            }).sort_values(
                "Importance",
                ascending=False,
            )

            fig = px.bar(
                importance_df.head(15),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Features",
                color="Importance",
            )

            fig.update_layout(
                yaxis={
                    "categoryorder":
                    "total ascending"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                importance_df,
                use_container_width=True,
            )

        elif method == "SHAP Global Importance":
            sample = X_train.sample(
                n=min(100, len(X_train)),
                random_state=42,
            )

            with st.spinner(
                "Calculating SHAP values..."
            ):
                explainer = shap.TreeExplainer(
                    model
                )

                values = explainer.shap_values(
                    sample
                )

            if isinstance(values, list):
                class_importances = [
                    np.abs(
                        np.asarray(v)
                    ).mean(axis=0)
                    for v in values
                ]

                mean_importance = np.mean(
                    np.vstack(
                        class_importances
                    ),
                    axis=0,
                )

            else:
                values = np.asarray(
                    values
                )

                if values.ndim == 3:
                    mean_importance = (
                        np.abs(values)
                        .mean(axis=(0, 2))
                    )
                elif values.ndim == 2:
                    mean_importance = (
                        np.abs(values)
                        .mean(axis=0)
                    )
                else:
                    raise ValueError(
                        f"Unsupported SHAP shape: "
                        f"{values.shape}"
                    )

            if len(mean_importance) != len(
                feature_names
            ):
                raise ValueError(
                    "SHAP feature count does not "
                    "match model feature count."
                )

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean Absolute SHAP":
                    mean_importance,
            }).sort_values(
                "Mean Absolute SHAP",
                ascending=False,
            )

            fig = px.bar(
                shap_df.head(15),
                x="Mean Absolute SHAP",
                y="Feature",
                orientation="h",
                title="Global SHAP Importance",
                color="Mean Absolute SHAP",
            )

            fig.update_layout(
                yaxis={
                    "categoryorder":
                    "total ascending"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                shap_df,
                use_container_width=True,
            )

        else:
            perm = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=5,
                random_state=42,
                n_jobs=-1,
            )

            perm_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance":
                    perm.importances_mean,
                "Std":
                    perm.importances_std,
            }).sort_values(
                "Importance",
                ascending=False,
            )

            fig = px.bar(
                perm_df.head(15),
                x="Importance",
                y="Feature",
                orientation="h",
                error_x="Std",
                title="Permutation Feature Importance",
                color="Importance",
            )

            fig.update_layout(
                yaxis={
                    "categoryorder":
                    "total ascending"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                perm_df,
                use_container_width=True,
            )

    except Exception as exc:
        st.error(
            f"Feature importance error: {exc}"
        )


def display_local_explanation(
    model,
    X_train,
    X_test,
    feature_names,
):
    st.markdown(
        "### 🔍 Local Prediction Explanation"
    )

    if len(X_test) == 0:
        st.warning(
            "Test set is empty."
        )
        return

    student_idx = st.selectbox(
        "Select student from test set",
        range(len(X_test)),
        format_func=lambda x:
            f"Student {x + 1}",
        key="local_student",
    )

    student = X_test.iloc[
        student_idx:student_idx + 1
    ]

    prediction = int(
        model.predict(student)[0]
    )

    probabilities = model.predict_proba(
        student
    )[0]

    predicted_name = (
        REVERSE_TARGET_MAPPING.get(
            prediction,
            str(prediction),
        )
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Outcome",
        predicted_name,
    )

    col2.metric(
        "Confidence",
        f"{np.max(probabilities):.2%}",
    )

    dropout_index = (
        list(model.classes_).index(0)
        if 0 in model.classes_
        else None
    )

    col3.metric(
        "Dropout Probability",
        "N/A"
        if dropout_index is None
        else f"{probabilities[dropout_index]:.2%}",
    )

    probability_df = pd.DataFrame({
        "Outcome": [
            REVERSE_TARGET_MAPPING.get(
                int(c),
                str(c),
            )
            for c in model.classes_
        ],
        "Probability": probabilities,
    })

    fig = px.bar(
        probability_df,
        x="Outcome",
        y="Probability",
        title="Prediction Probabilities",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    method = st.selectbox(
        "Explanation Method",
        [
            "SHAP Local Explanation",
            "LIME Explanation",
        ],
        key="local_method",
    )

    try:
        if method == "SHAP Local Explanation":
            explainer = shap.TreeExplainer(
                model
            )

            class_index = list(
                model.classes_
            ).index(prediction)

            values = normalize_shap_values(
                explainer,
                student,
                class_index,
            )

            values = np.asarray(
                values
            )[0]

            contribution_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": values,
                "Value":
                    student.iloc[0].values,
            })

            contribution_df[
                "Absolute SHAP"
            ] = contribution_df[
                "SHAP Value"
            ].abs()

            contribution_df = (
                contribution_df
                .sort_values(
                    "Absolute SHAP",
                    ascending=False,
                )
            )

            fig = px.bar(
                contribution_df.head(10),
                x="SHAP Value",
                y="Feature",
                orientation="h",
                color="SHAP Value",
                title=(
                    f"Top Features Affecting "
                    f"{predicted_name}"
                ),
            )

            fig.update_layout(
                yaxis={
                    "categoryorder":
                    "total ascending"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                contribution_df[
                    [
                        "Feature",
                        "SHAP Value",
                        "Value",
                    ]
                ].head(10),
                use_container_width=True,
            )

        else:
            explainer = (
                lime.lime_tabular
                .LimeTabularExplainer(
                    X_train.values,
                    feature_names=list(
                        feature_names
                    ),
                    class_names=[
                        REVERSE_TARGET_MAPPING.get(
                            int(c),
                            str(c),
                        )
                        for c in model.classes_
                    ],
                    mode="classification",
                    discretize_continuous=True,
                    random_state=42,
                )
            )

            explanation = (
                explainer.explain_instance(
                    student.iloc[0].values,
                    model.predict_proba,
                    num_features=min(
                        15,
                        len(feature_names),
                    ),
                )
            )

            fig = (
                explanation
                .as_pyplot_figure()
            )

            st.pyplot(
                fig,
                clear_figure=True,
            )

            lime_df = pd.DataFrame(
                explanation.as_list(),
                columns=[
                    "Feature",
                    "Weight",
                ],
            )

            st.dataframe(
                lime_df,
                use_container_width=True,
            )

    except Exception as exc:
        st.error(
            f"Explanation error: {exc}"
        )


# ============================================================
# FEATURE IMPACT
# ============================================================

def display_feature_impact_analysis(
    model,
    X_test,
    feature_names,
    df,
):
    st.markdown(
        "### 📈 Feature Impact Analysis"
    )

    numeric_features = [
        feature
        for feature in feature_names
        if feature in df.columns
        and pd.api.types.is_numeric_dtype(
            df[feature]
        )
    ]

    if not numeric_features:
        st.info(
            "No numeric features available."
        )
        return

    selected = st.selectbox(
        "Select feature",
        numeric_features,
        key="impact_feature",
    )

    series = pd.to_numeric(
        df[selected],
        errors="coerce",
    ).dropna()

    if series.empty:
        st.warning(
            "Selected feature has no numeric values."
        )
        return

    min_value = float(series.min())
    max_value = float(series.max())
    mean_value = float(series.mean())

    if min_value == max_value:
        feature_range = np.array([
            min_value
        ])
    else:
        feature_range = np.linspace(
            min_value,
            max_value,
            40,
        )

    base_student = X_test.iloc[
        0:1
    ].copy()

    if selected not in base_student.columns:
        st.warning(
            "Selected feature is not in model input."
        )
        return

    results = []

    for value in feature_range:
        row = base_student.copy()
        row[selected] = value

        probabilities = model.predict_proba(
            row
        )[0]

        for class_value, probability in zip(
            model.classes_,
            probabilities,
        ):
            results.append({
                "Feature Value": value,
                "Outcome":
                    REVERSE_TARGET_MAPPING.get(
                        int(class_value),
                        str(class_value),
                    ),
                "Probability": probability,
            })

    impact_df = pd.DataFrame(
        results
    )

    fig = px.line(
        impact_df,
        x="Feature Value",
        y="Probability",
        color="Outcome",
        title=(
            f"Impact of {selected} "
            "on Predictions"
        ),
    )

    fig.add_vline(
        x=mean_value,
        line_dash="dash",
        annotation_text=(
            f"Mean: {mean_value:.2f}"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# INDIVIDUAL PREDICTION
# ============================================================

def individual_dropout_prediction(
    model,
    X,
    X_train,
    feature_names,
):
    st.markdown(
        "### 🎯 Individual Student Prediction"
    )

    feature_names = list(
        feature_names
    )

    input_data = {}

    midpoint = max(
        1,
        len(feature_names) // 2
    )

    left_features = feature_names[
        :midpoint
    ]

    right_features = feature_names[
        midpoint:
    ]

    left, right = st.columns(2)

    for container, features in [
        (left, left_features),
        (right, right_features),
    ]:

        with container:
            for feature in features:

                series = pd.to_numeric(
                    X[feature],
                    errors="coerce",
                ).dropna()

                if series.empty:
                    minimum = 0.0
                    maximum = 1.0
                    default = 0.0
                else:
                    minimum = float(
                        series.min()
                    )
                    maximum = float(
                        series.max()
                    )
                    default = float(
                        series.mean()
                    )

                if minimum == maximum:
                    input_data[feature] = minimum

                    st.number_input(
                        feature,
                        value=minimum,
                        disabled=True,
                        key=f"input_{feature}",
                    )
                else:
                    input_data[feature] = (
                        st.number_input(
                            feature,
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            key=f"input_{feature}",
                        )
                    )

    if st.button(
        "🔮 Predict with Explanation",
        type="primary",
        key="prediction_button",
    ):

        try:
            row = {
                feature:
                    float(
                        input_data[feature]
                    )
                for feature in feature_names
            }

            input_df = pd.DataFrame(
                [row],
                columns=feature_names,
            )

            prediction = int(
                model.predict(
                    input_df
                )[0]
            )

            probabilities = (
                model.predict_proba(
                    input_df
                )[0]
            )

            predicted_name = (
                REVERSE_TARGET_MAPPING.get(
                    prediction,
                    str(prediction),
                )
            )

            probability_map = {
                int(c): float(p)
                for c, p in zip(
                    model.classes_,
                    probabilities,
                )
            }

            dropout_probability = (
                probability_map.get(
                    0,
                    0.0,
                )
            )

            st.markdown(
                "### 📊 Prediction Results"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Predicted Outcome",
                predicted_name,
            )

            col2.metric(
                "Prediction Confidence",
                f"{max(probabilities):.2%}",
            )

            col3.metric(
                "Dropout Probability",
                f"{dropout_probability:.2%}",
            )

            probability_df = pd.DataFrame({
                "Outcome": [
                    REVERSE_TARGET_MAPPING.get(
                        int(c),
                        str(c),
                    )
                    for c in model.classes_
                ],
                "Probability": probabilities,
            })

            fig = px.bar(
                probability_df,
                x="Outcome",
                y="Probability",
                title="Student Outcome Probabilities",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            if prediction == 0:
                st.error(
                    f"⚠️ Dropout Risk Detected "
                    f"({dropout_probability:.2%})"
                )

                st.markdown(
                    "### 💡 Suggested Support Areas"
                )

                st.write(
                    "• Academic counselling and tutoring"
                )
                st.write(
                    "• Financial aid review"
                )
                st.write(
                    "• Regular academic progress monitoring"
                )
                st.write(
                    "• Student engagement support"
                )
            else:
                st.success(
                    f"✅ Predicted as "
                    f"{predicted_name}"
                )

            # SHAP explanation.
            st.markdown(
                "### 🔍 Prediction Explanation"
            )

            try:
                explainer = shap.TreeExplainer(
                    model
                )

                class_index = list(
                    model.classes_
                ).index(prediction)

                shap_values = (
                    normalize_shap_values(
                        explainer,
                        input_df,
                        class_index,
                    )
                )

                shap_values = np.asarray(
                    shap_values
                )[0]

                contribution_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": shap_values,
                    "Value":
                        input_df.iloc[0].values,
                })

                contribution_df[
                    "Absolute SHAP"
                ] = contribution_df[
                    "SHAP Value"
                ].abs()

                contribution_df = (
                    contribution_df
                    .sort_values(
                        "Absolute SHAP",
                        ascending=False,
                    )
                )

                top = contribution_df.head(
                    10
                )

                fig = px.bar(
                    top,
                    x="SHAP Value",
                    y="Feature",
                    orientation="h",
                    color="SHAP Value",
                    title=(
                        "Top Features Affecting "
                        f"{predicted_name} Prediction"
                    ),
                )

                fig.update_layout(
                    yaxis={
                        "categoryorder":
                        "total ascending"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.dataframe(
                    top[
                        [
                            "Feature",
                            "SHAP Value",
                            "Value",
                        ]
                    ],
                    use_container_width=True,
                )

            except Exception as exc:
                st.warning(
                    "Prediction completed, but "
                    f"SHAP explanation is unavailable: {exc}"
                )

        except Exception as exc:
            st.error(
                f"Prediction error: {exc}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    init_session_state()

    st.title(
        "🎓 EduShield AI"
    )

    st.subheader(
        "Early Student Dropout Risk Detection "
        "and Support System"
    )

    st.write(
        "Machine Learning based student outcome "
        "prediction with explainable insights."
    )

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.title(
        "📌 Navigation"
    )

    choice = st.sidebar.radio(
        "Select Module",
        [
            "Data Overview",
            "Exploratory Data Analysis",
            "Model Training & Evaluation",
            "Dropout Prediction",
        ],
    )

    st.sidebar.markdown(
        "---"
    )

    uploaded_file = st.sidebar.file_uploader(
        "📂 Upload Student CSV",
        type=["csv"],
        help=(
            "Supports comma-separated and semicolon-separated CSVs. "
            "Required Target values: Dropout, Enrolled, Graduate."
        ),
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    try:
        df = load_data(
            uploaded_file
        )

        X, y, processed_df = (
            preprocess_data(df)
        )

    except Exception as exc:
        st.error(
            f"❌ Dataset processing failed: {exc}"
        )

        st.info(
            "Expected dataset: Student Dropout dataset "
            "with Target = Dropout, Enrolled or Graduate."
        )

        st.stop()

    # --------------------------------------------------------
    # TRAIN/TEST SPLIT
    # --------------------------------------------------------

    if y.nunique() < 2:
        st.error(
            "Dataset must contain at least two target classes."
        )
        st.stop()

    try:
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )
        )

    except ValueError:
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
            )
        )

    # Store current dataset in session.
    st.session_state.raw_df = df
    st.session_state.processed_df = processed_df
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.feature_names = (
        X.columns.tolist()
    )

    st.sidebar.write(
        f"**Students:** {len(df):,}"
    )

    st.sidebar.write(
        f"**Features:** {X.shape[1]:,}"
    )

    st.sidebar.write(
        "**Target:** Dropout / Enrolled / Graduate"
    )

    # --------------------------------------------------------
    # DATA OVERVIEW
    # --------------------------------------------------------

    if choice == "Data Overview":

        st.header(
            "📊 Data Overview"
        )

        sub_menu = st.selectbox(
            "Select Analysis",
            [
                "Data Quality Assessment",
                "Demographics Insights",
                "Academic Performance",
                "Performance Summary",
                "Risk Factors Analysis",
                "Data Structure",
                "Feature Explorer",
                "Column Information",
            ],
        )

        if sub_menu == "Data Quality Assessment":

            total = len(df)

            dropout_rate = (
                (
                    df["Target"] == "Dropout"
                ).sum()
                / total
                * 100
            )

            graduate_rate = (
                (
                    df["Target"] == "Graduate"
                ).sum()
                / total
                * 100
            )

            missing_percentage = (
                df.isnull().sum().sum()
                / (
                    df.shape[0]
                    * df.shape[1]
                )
                * 100
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            col1.metric(
                "Total Students",
                f"{total:,}",
            )

            col2.metric(
                "Dropout Rate",
                f"{dropout_rate:.1f}%",
            )

            col3.metric(
                "Graduate Rate",
                f"{graduate_rate:.1f}%",
            )

            col4.metric(
                "Missing Data",
                f"{missing_percentage:.1f}%",
            )

            display_data_quality(df)

        elif sub_menu == "Demographics Insights":
            display_demographics(df)

        elif sub_menu == "Academic Performance":
            display_academic_performance(df)

        elif sub_menu == "Performance Summary":
            display_performance_summary(df)

        elif sub_menu == "Risk Factors Analysis":
            display_risk_factors(df)

        elif sub_menu == "Data Structure":
            display_data_structure(df)

        elif sub_menu == "Feature Explorer":
            display_feature_explorer(df)

        else:
            display_column_info(df)

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    elif choice == "Exploratory Data Analysis":
        display_eda(df)

    # --------------------------------------------------------
    # MODEL TRAINING
    # --------------------------------------------------------

    elif choice == "Model Training & Evaluation":

        st.header(
            "🤖 Model Training & Evaluation"
        )

        sub_menu = st.selectbox(
            "Select Action",
            [
                "Train Model",
                "View Results",
                "Model Explainability",
            ],
        )

        if sub_menu == "Train Model":

            st.markdown(
                "### Target Label Encoding"
            )

            mapping_df = pd.DataFrame({
                "Original": [
                    "Dropout",
                    "Graduate",
                    "Enrolled",
                ],
                "Encoded": [
                    0,
                    1,
                    2,
                ],
            })

            st.dataframe(
                mapping_df,
                use_container_width=True,
            )

            st.write(
                "Target classes:",
                sorted(
                    df["Target"].unique()
                ),
            )

            if st.button(
                "🚀 Start Training",
                type="primary",
            ):

                with st.spinner(
                    "Training Random Forest..."
                ):
                    model = train_model(
                        X_train,
                        y_train,
                    )

                st.session_state.model = (
                    model
                )

                st.session_state.model_trained = (
                    True
                )

                st.success(
                    "✅ Model trained successfully!"
                )

        elif sub_menu == "View Results":

            if not st.session_state.model_trained:
                st.warning(
                    "Please train the model first."
                )
            else:
                visualize_model_results(
                    st.session_state.model,
                    X_test,
                    y_test,
                )

        else:

            if not st.session_state.model_trained:
                st.warning(
                    "Please train the model first."
                )

            else:

                explainability_type = (
                    st.selectbox(
                        "Select Explainability Analysis",
                        [
                            "Global Feature Importance",
                            "Local Prediction Explanation",
                            "Feature Impact Analysis",
                        ],
                    )
                )

                if (
                    explainability_type
                    == "Global Feature Importance"
                ):
                    display_global_feature_importance(
                        st.session_state.model,
                        X_train,
                        X_test,
                        y_test,
                        st.session_state.feature_names,
                    )

                elif (
                    explainability_type
                    == "Local Prediction Explanation"
                ):
                    display_local_explanation(
                        st.session_state.model,
                        X_train,
                        X_test,
                        st.session_state.feature_names,
                    )

                else:
                    display_feature_impact_analysis(
                        st.session_state.model,
                        X_test,
                        st.session_state.feature_names,
                        df,
                    )

    # --------------------------------------------------------
    # DROPOUT PREDICTION
    # --------------------------------------------------------

    elif choice == "Dropout Prediction":

        st.header(
            "🔮 Dropout Prediction"
        )

        if not st.session_state.model_trained:

            st.warning(
                "Please train the model first."
            )

            st.info(
                "Go to "
                "**Model Training & Evaluation → "
                "Train Model**."
            )

        else:

            individual_dropout_prediction(
                st.session_state.model,
                X,
                X_train,
                st.session_state.feature_names,
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
