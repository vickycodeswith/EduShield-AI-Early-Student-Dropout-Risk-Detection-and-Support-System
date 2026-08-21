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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

TARGET_MAPPING = {
    "Graduate": 1,
    "Dropout": 0,
    "Enrolled": 2
}

REVERSE_TARGET_MAPPING = {
    0: "Dropout",
    1: "Graduate",
    2: "Enrolled"
}

EXPECTED_TARGETS = set(TARGET_MAPPING.keys())


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_column_names(df):
    """Clean CSV headers safely."""
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def get_numeric_columns(df, exclude=None):
    exclude = set(exclude or [])
    return [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in exclude
    ]


def safe_mean(series):
    value = pd.to_numeric(series, errors="coerce").mean()
    return 0.0 if pd.isna(value) else float(value)


def class_name(value):
    return REVERSE_TARGET_MAPPING.get(int(value), f"Class {value}")


# ============================================================
# CSV LOADING - FIXED
# ============================================================

def load_csv_safely(uploaded_file):
    """
    Load comma-, semicolon-, or tab-separated CSV files.

    The UCI Student Dropout dataset is commonly semicolon-separated,
    while many other CSV files are comma-separated.
    """

    separators = [";", ",", "\t"]
    last_error = None

    for separator in separators:
        try:
            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                sep=separator,
                encoding="utf-8-sig"
            )

            # Wrong separator often produces one giant column.
            if len(df.columns) <= 1:
                continue

            df = clean_column_names(df)

            # Detect Target case-insensitively.
            target_column = None
            for col in df.columns:
                if str(col).strip().lower() == "target":
                    target_column = col
                    break

            if target_column is not None and target_column != "Target":
                df = df.rename(columns={target_column: "Target"})

            return df

        except Exception as exc:
            last_error = exc

    raise ValueError(
        f"Unable to read the CSV file. Details: {last_error}"
    )


def make_sample_dataset():
    """Small fallback dataset used only when no CSV is supplied."""
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
            "Dropout"
        ]
    })


def load_data(uploaded_file=None):
    """Load uploaded data or a local/fallback dataset."""

    if uploaded_file is not None:
        try:
            df = load_csv_safely(uploaded_file)

            if "Target" not in df.columns:
                st.error("❌ Invalid dataset: 'Target' column was not found.")
                st.info(
                    "Upload the Student Dropout dataset containing "
                    "a column named 'Target'."
                )
                st.write("Detected columns:")
                st.write(df.columns.tolist())
                st.stop()

            # Normalize target values.
            df["Target"] = (
                df["Target"]
                .astype(str)
                .str.strip()
            )

            actual_targets = set(df["Target"].dropna().unique())
            invalid_targets = actual_targets - EXPECTED_TARGETS

            if invalid_targets:
                st.error("❌ Invalid values were found in the Target column.")
                st.write("Detected Target values:", sorted(actual_targets))
                st.info(
                    "Expected Target values: Dropout, Enrolled, Graduate."
                )
                st.stop()

            st.success(
                f"✅ Successfully loaded uploaded file! "
                f"{df.shape[0]:,} students × {df.shape[1]} columns"
            )

            return df

        except Exception as exc:
            st.error(f"❌ Error loading uploaded CSV: {exc}")
            st.stop()

    # Try common local paths.
    potential_paths = [
        "./data/student_dropout_data.csv",
        "../data/student_dropout_data.csv",
        "./student_dropout_data.csv",
        "./data.csv"
    ]

    for path in potential_paths:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
            df = clean_column_names(df)

            if "Target" in df.columns:
                df["Target"] = df["Target"].astype(str).str.strip()
                st.success(f"Successfully loaded data from {path}")
                return df
        except Exception:
            continue

    st.warning(
        "⚠️ No dataset was uploaded. A small sample dataset is being used."
    )
    return make_sample_dataset()


# ============================================================
# PREPROCESSING - FIXED
# ============================================================

def preprocess_data(df):
    """
    Prepare data for Random Forest.

    Important:
    - Target must exist.
    - Dropout/Graduate/Enrolled are mapped to 0/1/2.
    - ID is removed.
    - Numeric missing values use the median.
    - Object categorical features are label encoded.
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
        - EXPECTED_TARGETS
    )

    if invalid_targets:
        raise ValueError(
            f"Invalid Target values: {sorted(invalid_targets)}. "
            "Expected Dropout, Graduate, Enrolled."
        )

    # Remove ID from model features.
    if "id" in processed_df.columns:
        processed_df = processed_df.drop(columns=["id"])

    # Convert numeric-looking columns where possible.
    for col in processed_df.columns:
        if col != "Target":
            processed_df[col] = pd.to_numeric(
                processed_df[col],
                errors="ignore"
            )

    # Fill numeric missing values.
    numeric_cols = processed_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    for col in numeric_cols:
        if processed_df[col].isna().any():
            median_value = processed_df[col].median()
            if pd.isna(median_value):
                median_value = 0
            processed_df[col] = processed_df[col].fillna(median_value)

    # Encode any remaining categorical feature columns.
    categorical_cols = processed_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    categorical_cols = [
        col for col in categorical_cols
        if col != "Target"
    ]

    for col in categorical_cols:
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
    processed_df["Target"] = processed_df["Target"].map(
        TARGET_MAPPING
    )

    if processed_df["Target"].isna().any():
        raise ValueError(
            "Target mapping failed. Expected Dropout, Graduate, Enrolled."
        )

    # Convert all feature columns to numeric.
    feature_cols = [
        col for col in processed_df.columns
        if col != "Target"
    ]

    for col in feature_cols:
        processed_df[col] = pd.to_numeric(
            processed_df[col],
            errors="coerce"
        )

        if processed_df[col].isna().any():
            median_value = processed_df[col].median()
            if pd.isna(median_value):
                median_value = 0
            processed_df[col] = processed_df[col].fillna(median_value)

    processed_df = processed_df.dropna()

    if processed_df.empty:
        raise ValueError(
            "No valid rows remain after preprocessing."
        )

    X = processed_df.drop(columns=["Target"])
    y = processed_df["Target"].astype(int)

    return X, y, processed_df


# ============================================================
# DATA OVERVIEW
# ============================================================

def display_data_quality(df):
    st.markdown("## 🔍 Data Quality Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Missing Data by Column")

        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

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
        st.markdown("### Data Type Distribution")

        dtype_counts = df.dtypes.astype(str).value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(
            dtype_counts.values,
            labels=dtype_counts.index,
            autopct="%1.1f%%"
        )
        ax.set_title("Proportion of Data Types")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def display_demographics(df):
    st.markdown("## 👥 Student Demographics")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "Age at enrollment" in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df["Age at enrollment"].dropna(), bins=20)
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
                "Female" if int(x) == 0 else "Male"
                for x in counts.index
            ]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.pie(counts.values, labels=labels, autopct="%1.1f%%")
            ax.set_title("Gender Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col3:
        if "Scholarship holder" in df.columns:
            counts = df["Scholarship holder"].value_counts()
            labels = [
                "No Scholarship" if int(x) == 0 else "Scholarship"
                for x in counts.index
            ]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.pie(counts.values, labels=labels, autopct="%1.1f%%")
            ax.set_title("Scholarship Status")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


def display_academic_performance(df):
    st.markdown("## 📚 Academic Performance")

    academic_columns = [
        "Admission grade",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)"
    ]

    available = [
        col for col in academic_columns
        if col in df.columns
    ]

    if not available:
        st.info("No academic performance columns found.")
        return

    for col in available:
        fig, ax = plt.subplots(figsize=(9, 4))

        for outcome in df["Target"].dropna().unique():
            values = df.loc[
                df["Target"] == outcome,
                col
            ].dropna()

            if len(values):
                ax.hist(
                    values,
                    bins=15,
                    alpha=0.5,
                    label=outcome
                )

        ax.set_title(f"{col} by Student Outcome")
        ax.set_xlabel(col)
        ax.set_ylabel("Students")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def display_performance_summary(df):
    st.markdown("## 📈 Performance Summary")

    col1, col2, col3 = st.columns(3)

    if "Admission grade" in df.columns:
        graduate_mean = safe_mean(
            df.loc[df["Target"] == "Graduate", "Admission grade"]
        )
        dropout_mean = safe_mean(
            df.loc[df["Target"] == "Dropout", "Admission grade"]
        )

        col1.metric(
            "Graduate Admission Grade",
            f"{graduate_mean:.1f}",
            f"{graduate_mean - dropout_mean:.1f} vs Dropout"
        )

    if "Curricular units 1st sem (grade)" in df.columns:
        graduate_mean = safe_mean(
            df.loc[
                df["Target"] == "Graduate",
                "Curricular units 1st sem (grade)"
            ]
        )
        dropout_mean = safe_mean(
            df.loc[
                df["Target"] == "Dropout",
                "Curricular units 1st sem (grade)"
            ]
        )

        col2.metric(
            "Graduate 1st Sem Grade",
            f"{graduate_mean:.1f}",
            f"{graduate_mean - dropout_mean:.1f} vs Dropout"
        )

        high = df[
            df["Curricular units 1st sem (grade)"] > 15
        ]
        success_rate = (
            (high["Target"] == "Graduate").mean() * 100
            if len(high)
            else 0
        )

        col3.metric(
            "Success Rate (Grade > 15)",
            f"{success_rate:.1f}%"
        )


def display_risk_factors(df):
    st.markdown("## ⚠️ Key Risk Factors")

    risk_factors = []

    if "Curricular units 1st sem (approved)" in df.columns:
        risk_factors.append((
            "Low 1st Sem Approved Units",
            int(
                (df["Curricular units 1st sem (approved)"] <= 2)
                .sum()
            )
        ))

    if "Age at enrollment" in df.columns:
        risk_factors.append((
            "Age > 25",
            int((df["Age at enrollment"] > 25).sum())
        ))

    if "Scholarship holder" in df.columns:
        risk_factors.append((
            "No Scholarship",
            int((df["Scholarship holder"] == 0).sum())
        ))

    if "Displaced" in df.columns:
        risk_factors.append((
            "Displaced",
            int((df["Displaced"] == 1).sum())
        ))

    if not risk_factors:
        st.info("No supported risk-factor columns found.")
        return

    risk_df = pd.DataFrame(
        risk_factors,
        columns=["Risk Factor", "Number of Students"]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        risk_df["Risk Factor"],
        risk_df["Number of Students"]
    )
    ax.set_title("Students by Risk Factor")
    ax.set_ylabel("Number of Students")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.dataframe(risk_df, use_container_width=True)


def display_data_structure(df):
    st.markdown("## 📋 Data Sample & Structure")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    with col2:
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", f"{len(df.columns):,}")
        st.write("### Target Distribution")
        st.dataframe(
            df["Target"].value_counts().rename("Students"),
            use_container_width=True
        )


def display_feature_explorer(df):
    st.markdown("## 🔧 Interactive Feature Explorer")

    numeric_cols = get_numeric_columns(
        df,
        exclude=["id"]
    )

    if not numeric_cols:
        st.info("No numeric features available.")
        return

    feature = st.selectbox(
        "Select a feature",
        numeric_cols,
        key="feature_explorer_feature"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write(df[feature].describe())

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[feature].dropna(), bins=25)
        ax.set_title(f"Distribution of {feature}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        temp = df[[feature, "Target"]].copy()
        temp["Target Code"] = temp["Target"].map(TARGET_MAPPING)

        correlation = temp[feature].corr(temp["Target Code"])

        st.metric(
            "Correlation with Target Code",
            f"{correlation:.3f}" if not pd.isna(correlation)
            else "N/A"
        )

        temp["bin"] = pd.qcut(
            temp[feature],
            q=5,
            duplicates="drop"
        )

        dropout_rate = (
            temp.groupby("bin", observed=True)["Target"]
            .apply(lambda x: (x == "Dropout").mean() * 100)
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        dropout_rate.plot(kind="bar", ax=ax)
        ax.set_ylabel("Dropout Rate (%)")
        ax.set_title(f"Dropout Rate by {feature}")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def display_column_info(df):
    st.markdown("## 📋 Detailed Column Information")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Statistics",
        "Missing Data",
        "Data Types"
    ])

    with tab1:
        st.write("**Rows:**", len(df))
        st.write("**Columns:**", len(df.columns))
        st.write("**Column Names:**")
        st.write(df.columns.tolist())

    with tab2:
        st.dataframe(
            df.describe(include="all").transpose(),
            use_container_width=True
        )

    with tab3:
        missing = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": df.isnull().sum().values,
            "Missing %": (
                df.isnull().mean().values * 100
            ).round(2)
        })

        st.dataframe(
            missing[missing["Missing Count"] > 0],
            use_container_width=True
        )

    with tab4:
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Null": df.isnull().sum().values
        })

        st.dataframe(
            dtype_df,
            use_container_width=True
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
            "Academic Performance"
        ],
        key="eda_submenu"
    )

    if sub_menu == "Distribution Analysis":
        numeric_cols = get_numeric_columns(
            df,
            exclude=["id"]
        )

        selected = st.multiselect(
            "Select up to 4 features",
            numeric_cols,
            default=[
                c for c in [
                    "Age at enrollment",
                    "Admission grade",
                    "Curricular units 1st sem (grade)"
                ]
                if c in numeric_cols
            ],
            max_selections=4
        )

        if not selected:
            st.info("Select at least one feature.")
            return

        cols = st.columns(min(2, len(selected)))

        for i, feature in enumerate(selected):
            with cols[i % len(cols)]:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(
                    df[feature].dropna(),
                    kde=True,
                    ax=ax
                )
                ax.set_title(f"Distribution of {feature}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    elif sub_menu == "Correlation Analysis":
        numeric_cols = get_numeric_columns(
            df,
            exclude=["id"]
        )

        selected = st.multiselect(
            "Select features",
            numeric_cols,
            default=[
                c for c in [
                    "Age at enrollment",
                    "Admission grade",
                    "Curricular units 1st sem (grade)",
                    "Curricular units 2nd sem (grade)"
                ]
                if c in numeric_cols
            ]
        )

        if len(selected) < 2:
            st.info("Select at least two features.")
            return

        corr = df[selected].corr()

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax
        )
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    elif sub_menu == "Outcome Analysis":
        col1, col2 = st.columns(2)

        with col1:
            counts = df["Target"].value_counts()

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%"
            )
            ax.set_title("Student Outcome Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            total = len(df)

            for outcome in [
                "Dropout",
                "Enrolled",
                "Graduate"
            ]:
                count = int(
                    (df["Target"] == outcome).sum()
                )
                st.metric(
                    outcome,
                    f"{count:,}",
                    f"{count / total * 100:.1f}%"
                )

    else:
        display_academic_performance(df)


# ============================================================
# MODEL
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
        n_jobs=-1
    )

    status.write("Fitting Random Forest...")
    progress.progress(50)

    model.fit(X_train, y_train)

    progress.progress(100)
    status.success("Model training complete!")

    return model


def visualize_model_results(model, X_test, y_test):
    st.markdown("## 📊 Model Evaluation Results")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.metric(
        "Model Accuracy",
        f"{accuracy:.2%}"
    )

    tab1, tab2, tab3 = st.tabs([
        "Classification Report",
        "Confusion Matrix",
        "Feature Importance"
    ])

    with tab1:
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(
            report_df,
            use_container_width=True
        )

    with tab2:
        cm = confusion_matrix(
            y_test,
            y_pred
        )

        labels = [
            class_name(c)
            for c in sorted(np.unique(y_test))
        ]

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        top = importance_df.head(15)

        fig, ax = plt.subplots(figsize=(9, 6))

        sns.barplot(
            data=top,
            x="Importance",
            y="Feature",
            ax=ax
        )

        ax.set_title("Top 15 Feature Importances")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(
            importance_df,
            use_container_width=True
        )


# ============================================================
# SHAP HELPERS
# ============================================================

def get_shap_class_values(explainer, X_data, class_index):
    """
    Normalize SHAP output across different SHAP versions.

    Returns:
        1D array for one row or
        2D array (samples x features) for multiple rows.
    """

    values = explainer.shap_values(X_data)

    if isinstance(values, list):
        class_index = min(
            class_index,
            len(values) - 1
        )
        return np.asarray(values[class_index])

    values = np.asarray(values)

    if values.ndim == 3:
        # Common newer SHAP format:
        # samples x features x classes
        return values[:, :, class_index]

    if values.ndim == 2:
        return values

    if values.ndim == 1:
        return values.reshape(1, -1)

    raise ValueError(
        f"Unsupported SHAP output shape: {values.shape}"
    )


def display_global_feature_importance(
    model,
    feature_names
):
    st.markdown("### 🌍 Global Feature Importance")

    method = st.selectbox(
        "Select Importance Method",
        [
            "Built-in Feature Importance",
            "SHAP Global Importance",
            "Permutation Importance"
        ],
        key="global_importance_method"
    )

    feature_names = list(feature_names)

    try:
        if method == "Built-in Feature Importance":

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(
                "Importance",
                ascending=False
            )

            fig = px.bar(
                importance_df.head(15),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Features",
                color="Importance"
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                importance_df,
                use_container_width=True
            )

        elif method == "SHAP Global Importance":

            X_train = st.session_state.get(
                "X_train"
            )

            if X_train is None:
                st.warning(
                    "Train the model first."
                )
                return

            sample = X_train.sample(
                n=min(100, len(X_train)),
                random_state=42
            )

            with st.spinner(
                "Calculating SHAP values..."
            ):
                explainer = shap.TreeExplainer(model)
                values = explainer.shap_values(sample)

            if isinstance(values, list):
                class_importances = [
                    np.abs(np.asarray(v)).mean(axis=0)
                    for v in values
                ]
                mean_importance = np.mean(
                    np.vstack(class_importances),
                    axis=0
                )
            else:
                values = np.asarray(values)

                if values.ndim == 3:
                    mean_importance = np.abs(
                        values
                    ).mean(axis=(0, 2))
                elif values.ndim == 2:
                    mean_importance = np.abs(
                        values
                    ).mean(axis=0)
                else:
                    raise ValueError(
                        f"Unsupported SHAP shape: {values.shape}"
                    )

            if len(mean_importance) != len(feature_names):
                raise ValueError(
                    "SHAP feature count does not match model feature count."
                )

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean Absolute SHAP": mean_importance
            }).sort_values(
                "Mean Absolute SHAP",
                ascending=False
            )

            fig = px.bar(
                shap_df.head(15),
                x="Mean Absolute SHAP",
                y="Feature",
                orientation="h",
                title="Global SHAP Importance",
                color="Mean Absolute SHAP"
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                shap_df,
                use_container_width=True
            )

        else:
            X_test = st.session_state.get(
                "X_test"
            )
            y_test = st.session_state.get(
                "y_test"
            )

            if X_test is None or y_test is None:
                st.warning(
                    "Train the model first."
                )
                return

            perm = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )

            perm_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": perm.importances_mean,
                "Std": perm.importances_std
            }).sort_values(
                "Importance",
                ascending=False
            )

            fig = px.bar(
                perm_df.head(15),
                x="Importance",
                y="Feature",
                orientation="h",
                error_x="Std",
                title="Permutation Feature Importance",
                color="Importance"
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                perm_df,
                use_container_width=True
            )

    except Exception as exc:
        st.error(
            f"Feature importance error: {exc}"
        )


def display_local_explanation(
    model,
    X_train,
    X_test,
    feature_names
):
    st.markdown("### 🔍 Local Prediction Explanation")

    if len(X_test) == 0:
        st.warning("Test set is empty.")
        return

    student_idx = st.selectbox(
        "Select student from test set",
        range(len(X_test)),
        format_func=lambda x: f"Student {x + 1}",
        key="local_student"
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

    predicted_name = class_name(
        prediction
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Outcome",
        predicted_name
    )

    col2.metric(
        "Confidence",
        f"{np.max(probabilities):.2%}"
    )

    col3.metric(
        "Dropout Probability",
        f"{probabilities[list(model.classes_).index(0)]:.2%}"
        if 0 in model.classes_
        else "N/A"
    )

    prob_df = pd.DataFrame({
        "Outcome": [
            class_name(c)
            for c in model.classes_
        ],
        "Probability": probabilities
    })

    fig = px.bar(
        prob_df,
        x="Outcome",
        y="Probability",
        title="Prediction Probabilities"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    method = st.selectbox(
        "Explanation Method",
        [
            "SHAP Local Explanation",
            "LIME Explanation"
        ],
        key="local_explanation_method"
    )

    try:
        if method == "SHAP Local Explanation":

            explainer = shap.TreeExplainer(model)

            class_index = list(
                model.classes_
            ).index(prediction)

            values = get_shap_class_values(
                explainer,
                student,
                class_index
            )

            values = np.asarray(values)[0]

            expected = explainer.expected_value

            if isinstance(expected, (list, np.ndarray)):
                expected = np.asarray(
                    expected
                ).reshape(-1)[class_index]

            expected = float(
                np.asarray(expected).reshape(-1)[0]
            )

            explanation = shap.Explanation(
                values=values,
                base_values=expected,
                data=student.iloc[0].values,
                feature_names=list(feature_names)
            )

            st.markdown("#### 🌊 SHAP Waterfall")

            try:
                shap.plots.waterfall(
                    explanation,
                    max_display=15,
                    show=False
                )
                st.pyplot(
                    plt.gcf(),
                    clear_figure=True
                )
            except Exception as exc:
                st.warning(
                    f"Waterfall plot unavailable: {exc}"
                )

            contribution_df = pd.DataFrame({
                "Feature": list(feature_names),
                "SHAP Value": values,
                "Value": student.iloc[0].values
            })

            contribution_df["Absolute SHAP"] = (
                contribution_df["SHAP Value"].abs()
            )

            contribution_df = contribution_df.sort_values(
                "Absolute SHAP",
                ascending=False
            )

            st.markdown(
                "#### 🎯 Top Contributing Features"
            )

            st.dataframe(
                contribution_df.head(10),
                use_container_width=True
            )

            fig = px.bar(
                contribution_df.head(10),
                x="SHAP Value",
                y="Feature",
                orientation="h",
                color="SHAP Value",
                title=f"SHAP Contributions for {predicted_name}"
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=list(feature_names),
                class_names=[
                    class_name(c)
                    for c in model.classes_
                ],
                mode="classification",
                discretize_continuous=True,
                random_state=42
            )

            explanation = explainer.explain_instance(
                student.iloc[0].values,
                model.predict_proba,
                num_features=min(
                    15,
                    len(feature_names)
                )
            )

            st.markdown("#### 🍃 LIME Explanation")

            fig = explanation.as_pyplot_figure()
            st.pyplot(
                fig,
                clear_figure=True
            )

            lime_df = pd.DataFrame(
                explanation.as_list(),
                columns=["Feature", "Weight"]
            )

            st.dataframe(
                lime_df,
                use_container_width=True
            )

    except Exception as exc:
        st.error(
            f"Explanation error: {exc}"
        )


def display_feature_impact_analysis(
    model,
    X_train,
    X_test,
    feature_names,
    df
):
    st.markdown("### 📈 Feature Impact Analysis")

    numeric_features = [
        f for f in feature_names
        if f in df.columns
        and pd.api.types.is_numeric_dtype(
            df[f]
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
        key="impact_feature"
    )

    stats = df[selected].describe()

    min_value = float(stats["min"])
    max_value = float(stats["max"])
    mean_value = float(stats["mean"])

    if min_value == max_value:
        feature_range = np.array([min_value])
    else:
        feature_range = np.linspace(
            min_value,
            max_value,
            40
        )

    sample_student = X_test.iloc[
        0:1
    ].copy()

    if selected not in sample_student.columns:
        st.warning(
            "Selected feature is not part of the model input."
        )
        return

    results = []

    for value in feature_range:
        row = sample_student.copy()
        row[selected] = value
        probabilities = model.predict_proba(row)[0]

        for class_value, probability in zip(
            model.classes_,
            probabilities
        ):
            results.append({
                "Feature Value": value,
                "Outcome": class_name(class_value),
                "Probability": probability
            })

    impact_df = pd.DataFrame(results)

    fig = px.line(
        impact_df,
        x="Feature Value",
        y="Probability",
        color="Outcome",
        title=f"Impact of {selected} on Predictions"
    )

    fig.add_vline(
        x=mean_value,
        line_dash="dash",
        annotation_text=f"Mean: {mean_value:.2f}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Minimum",
        f"{min_value:.2f}"
    )

    col2.metric(
        "Mean",
        f"{mean_value:.2f}"
    )

    col3.metric(
        "Maximum",
        f"{max_value:.2f}"
    )


# ============================================================
# INDIVIDUAL PREDICTION
# ============================================================

def individual_dropout_prediction_with_explanation(
    model,
    X,
    X_train,
    feature_names
):
    st.markdown(
        "### 🎯 Individual Student Prediction with Explanation"
    )

    feature_names = list(feature_names)

    input_data = {}

    left, right = st.columns(2)

    midpoint = len(feature_names) // 2

    for column_container, features in [
        (left, feature_names[:midpoint]),
        (right, feature_names[midpoint:])
    ]:

        with column_container:

            for feature in features:

                if feature not in X.columns:
                    continue

                series = pd.to_numeric(
                    X[feature],
                    errors="coerce"
                ).dropna()

                if series.empty:
                    default = 0.0
                    min_value = 0.0
                    max_value = 1.0
                else:
                    min_value = float(series.min())
                    max_value = float(series.max())
                    default = float(series.mean())

                if min_value == max_value:
                    input_data[feature] = min_value
                    st.number_input(
                        feature,
                        value=float(min_value),
                        disabled=True,
                        key=f"prediction_{feature}"
                    )
                else:
                    input_data[feature] = st.number_input(
                        feature,
                        min_value=min_value,
                        max_value=max_value,
                        value=default,
                        key=f"prediction_{feature}"
                    )

    if st.button(
        "🔮 Predict with Explanation",
        type="primary",
        key="individual_prediction_button"
    ):

        try:
            ordered = {
                feature: float(
                    input_data.get(
                        feature,
                        float(X[feature].median())
                    )
                )
                for feature in feature_names
            }

            input_df = pd.DataFrame(
                [ordered],
                columns=feature_names
            )

            prediction = int(
                model.predict(input_df)[0]
            )

            prediction_proba = model.predict_proba(
                input_df
            )[0]

            predicted_name = class_name(
                prediction
            )

            class_to_probability = {
                int(c): float(p)
                for c, p in zip(
                    model.classes_,
                    prediction_proba
                )
            }

            dropout_probability = class_to_probability.get(
                0,
                0.0
            )

            st.markdown("### 📊 Prediction Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Predicted Outcome",
                predicted_name
            )

            col2.metric(
                "Prediction Confidence",
                f"{max(prediction_proba):.2%}"
            )

            col3.metric(
                "Dropout Probability",
                f"{dropout_probability:.2%}"
            )

            probability_df = pd.DataFrame({
                "Outcome": [
                    class_name(c)
                    for c in model.classes_
                ],
                "Probability": prediction_proba
            })

            fig = px.bar(
                probability_df,
                x="Outcome",
                y="Probability",
                title="Student Outcome Probabilities"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            if prediction == 0:
                st.error(
                    f"⚠️ High Dropout Risk Detected "
                    f"({dropout_probability:.2%})"
                )

                st.markdown("### 💡 Recommended Support")
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
                    f"✅ Low Dropout Risk — "
                    f"Predicted as {predicted_name}"
                )

            # SHAP explanation.
            st.markdown("### 🔍 Prediction Explanation")

            try:
                explainer = shap.TreeExplainer(model)

                class_index = list(
                    model.classes_
                ).index(prediction)

                values = get_shap_class_values(
                    explainer,
                    input_df,
                    class_index
                )

                values = np.asarray(values)[0]

                contribution_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP_Value": values,
                    "Value": input_df.iloc[0].values
                })

                contribution_df["Abs_SHAP"] = (
                    contribution_df["SHAP_Value"].abs()
                )

                contribution_df = contribution_df.sort_values(
                    "Abs_SHAP",
                    ascending=False
                )

                top = contribution_df.head(10)

                fig = px.bar(
                    top,
                    x="SHAP_Value",
                    y="Feature",
                    orientation="h",
                    color="SHAP_Value",
                    title=f"Top Features Affecting {predicted_name} Prediction"
                )

                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.dataframe(
                    top[
                        [
                            "Feature",
                            "SHAP_Value",
                            "Value"
                        ]
                    ],
                    use_container_width=True
                )

            except Exception as exc:
                st.warning(
                    f"Prediction completed, but SHAP explanation "
                    f"is unavailable: {exc}"
                )

        except Exception as exc:
            st.error(
                f"Prediction error: {exc}"
            )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.markdown(
        "<div style='font-size: 24px; font-weight: bold;'>"
        "🎓 Student Dropout Prediction Dashboard"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        This interactive dashboard helps predict and analyze
        factors contributing to student dropout.
        Use the sections below to explore the dataset,
        perform EDA, train a model, evaluate it, and generate
        explainable individual predictions.
        """
    )

    # Session state.
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False

    if "model" not in st.session_state:
        st.session_state.model = None

    # Sidebar.
    st.sidebar.markdown(
        "<div style='font-size: 20px; font-weight: bold;'>"
        "Navigation"
        "</div>",
        unsafe_allow_html=True
    )

    menu = [
        "Data Overview",
        "Exploratory Data Analysis",
        "Model Training & Evaluation",
        "Dropout Prediction"
    ]

    choice = st.sidebar.radio(
        "Select Module",
        menu,
        label_visibility="collapsed"
    )

    st.sidebar.markdown(
        "<div style='font-size: 16px; font-weight: bold;'>"
        "Data Input"
        "</div>",
        unsafe_allow_html=True
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload your Student Dropout CSV",
        type=["csv"],
        help=(
            "Supports comma-separated and semicolon-separated CSV files. "
            "The dataset must contain Target with Dropout, Enrolled, Graduate."
        )
    )

    # --------------------------------------------------------
    # LOAD + PREPROCESS DATA
    # --------------------------------------------------------

    try:
        df = load_data(uploaded_file)

        X, y, processed_df = preprocess_data(df)

    except Exception as exc:
        st.error(
            f"❌ Dataset processing failed: {exc}"
        )
        st.info(
            "Expected dataset: Student Dropout dataset with "
            "a Target column containing Dropout, Enrolled and Graduate."
        )
        st.stop()

    # --------------------------------------------------------
    # TRAIN/TEST SPLIT
    # --------------------------------------------------------

    if y.nunique() < 2:
        st.error(
            "The dataset needs at least two target classes."
        )
        st.stop()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

    # Keep data available for explainability.
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.feature_names = X.columns.tolist()

    # Sidebar summary.
    st.sidebar.markdown("---")
    st.sidebar.write(
        f"**Rows:** {len(df):,}"
    )
    st.sidebar.write(
        f"**Features:** {X.shape[1]:,}"
    )
    st.sidebar.write(
        f"**Target:** Dropout / Enrolled / Graduate"
    )

    # --------------------------------------------------------
    # DATA OVERVIEW
    # --------------------------------------------------------

    if choice == "Data Overview":

        st.markdown("## 📊 Data Overview")
        st.write(
            "Explore dataset structure, quality, demographics "
            "and academic performance."
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
                "Column Information"
            ],
            key="data_overview_submenu"
        )

        if sub_menu == "Data Quality Assessment":

            col1, col2, col3, col4 = st.columns(4)

            total = len(df)

            dropout_rate = (
                (df["Target"] == "Dropout").sum()
                / total * 100
            )

            graduate_rate = (
                (df["Target"] == "Graduate").sum()
                / total * 100
            )

            missing_pct = (
                df.isnull().sum().sum()
                / (df.shape[0] * df.shape[1])
                * 100
            )

            col1.metric(
                "Total Students",
                f"{total:,}"
            )

            col2.metric(
                "Dropout Rate",
                f"{dropout_rate:.1f}%"
            )

            col3.metric(
                "Graduate Rate",
                f"{graduate_rate:.1f}%"
            )

            col4.metric(
                "Missing Data",
                f"{missing_pct:.1f}%"
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

        elif sub_menu == "Column Information":
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

        st.markdown("## 🤖 Model Training & Evaluation")

        sub_menu = st.selectbox(
            "Select Action",
            [
                "Train Model",
                "View Results",
                "Model Explainability"
            ],
            key="model_submenu"
        )

        if sub_menu == "Train Model":

            st.markdown("### Target Label Encoding")

            mapping_df = pd.DataFrame({
                "Original": list(TARGET_MAPPING.keys()),
                "Encoded": list(TARGET_MAPPING.values())
            })

            st.dataframe(
                mapping_df,
                use_container_width=True
            )

            st.write(
                "Target values:",
                processed_df["Target"].unique().tolist()
            )

            if st.button(
                "🚀 Start Training",
                type="primary",
                key="start_training"
            ):

                with st.spinner(
                    "Training Random Forest..."
                ):

                    model = train_model(
                        X_train,
                        y_train
                    )

                    st.session_state.model = model
                    st.session_state.model_trained = True

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
                    y_test
                )

        elif sub_menu == "Model Explainability":

            if not st.session_state.model_trained:
                st.warning(
                    "Please train the model first."
                )
            else:

                explainability_type = st.selectbox(
                    "Select Explainability Analysis",
                    [
                        "Global Feature Importance",
                        "Local Prediction Explanation",
                        "Feature Impact Analysis"
                    ],
                    key="explainability_type"
                )

                if explainability_type == "Global Feature Importance":

                    display_global_feature_importance(
                        st.session_state.model,
                        st.session_state.feature_names
                    )

                elif explainability_type == "Local Prediction Explanation":

                    display_local_explanation(
                        st.session_state.model,
                        st.session_state.X_train,
                        st.session_state.X_test,
                        st.session_state.feature_names
                    )

                else:

                    display_feature_impact_analysis(
                        st.session_state.model,
                        st.session_state.X_train,
                        st.session_state.X_test,
                        st.session_state.feature_names,
                        df
                    )

    # --------------------------------------------------------
    # DROPOUT PREDICTION
    # --------------------------------------------------------

    elif choice == "Dropout Prediction":

        st.markdown("## 🔮 Dropout Prediction")

        if not st.session_state.model_trained:

            st.warning(
                "Please train the model first."
            )

            st.info(
                "Go to **Model Training & Evaluation → Train Model**."
            )

        else:

            individual_dropout_prediction_with_explanation(
                st.session_state.model,
                X,
                st.session_state.X_train,
                st.session_state.feature_names
            )


if __name__ == "__main__":
    main()
