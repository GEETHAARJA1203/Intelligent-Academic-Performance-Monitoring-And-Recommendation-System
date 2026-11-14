
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import html
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Academic Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "academic_performance_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "student_academic_data.csv"
)

SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "Department",
    "Semester",
    "Attendance",
    "Internal_Marks",
    "Assignment_Marks",
    "Quiz_Marks",
    "Study_Hours",
    "Previous_CGPA",
    "Backlogs",
    "Assignment_Submission_Rate",
    "Learning_Activity"
]

NUMERIC_FEATURES = [
    "Semester",
    "Attendance",
    "Internal_Marks",
    "Assignment_Marks",
    "Quiz_Marks",
    "Study_Hours",
    "Previous_CGPA",
    "Backlogs",
    "Assignment_Submission_Rate"
]


# ============================================================
# HTML RENDERING FUNCTION
# ============================================================

def render_html(content):
    if content is None:
        return

    cleaned_html = "\n".join(
        line.strip()
        for line in str(content).strip().splitlines()
    )

    st.markdown(
        cleaned_html,
        unsafe_allow_html=True
    )


def safe_text(value, default="Not available"):
    if pd.isna(value):
        return default

    return html.escape(str(value))


# ============================================================
# CUSTOM CSS
# ============================================================

render_html("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #172554 0%,
        #1e3a8a 100%
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.dashboard-header {
    background: linear-gradient(
        135deg,
        #1e3a8a,
        #2563eb
    );
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}

.dashboard-header h1 {
    margin: 0;
    font-size: 34px;
    font-weight: 700;
}

.dashboard-header p {
    margin-top: 8px;
    font-size: 16px;
    opacity: 0.9;
}

.metric-card {
    background: white;
    border-radius: 16px;
    padding: 22px;
    min-height: 145px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

.metric-icon {
    font-size: 28px;
    margin-bottom: 8px;
}

.metric-title {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #0f172a;
    font-size: 25px;
    font-weight: 700;
    margin-top: 5px;
    overflow-wrap: anywhere;
}

.metric-description {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 6px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 30px;
    margin-bottom: 15px;
}

.recommendation-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 12px;
    border-left: 5px solid #2563eb;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
}

.recommendation-card h4 {
    margin-top: 0;
    color: #1e3a8a;
}

.recommendation-card p {
    color: #475569;
}

.status-good {
    background: #dcfce7;
    color: #166534;
    padding: 12px 18px;
    border-radius: 10px;
    font-weight: 600;
}

.status-warning {
    background: #fef3c7;
    color: #92400e;
    padding: 12px 18px;
    border-radius: 10px;
    font-weight: 600;
}

.status-danger {
    background: #fee2e2;
    color: #991b1b;
    padding: 12px 18px;
    border-radius: 10px;
    font-weight: 600;
}

.footer {
    text-align: center;
    padding: 25px;
    color: #64748b;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""")


# ============================================================
# LOAD STUDENT RECORDS FROM SQLITE DATABASE
# ============================================================

try:
    from database import get_students

    @st.cache_data(ttl=30)
    def load_student_data():
        records = get_students()

        rows = [
            dict(record)
            for record in records
        ]

        if not rows:
            return pd.DataFrame()

        database_df = pd.DataFrame(rows)

        aliases = {
            "student_name": "name",
            "study_time": "studytime",
            "final_marks": "final_grade",
            "predicted_score": "predicted_final_marks"
        }

        database_df = database_df.rename(
            columns=aliases
        )

        database_df.columns = (
            database_df.columns.astype(str)
            .str.strip()
            .str.lower()
        )

        return database_df

    df = load_student_data()

except Exception as error:
    st.error(
        "Unable to load student records from SQLite."
    )
    st.code(str(error))
    st.stop()


if df.empty:
    st.warning(
        "No student records were found in SQLite."
    )
    st.stop()


# ============================================================
# LOAD ACADEMIC PERFORMANCE MODEL
# ============================================================

model = None
model_available = False
model_load_error = None

if os.path.isfile(MODEL_PATH):

    try:
        model = joblib.load(MODEL_PATH)

        if (
            hasattr(model, "predict")
            and hasattr(model, "named_steps")
            and "preprocessor" in model.named_steps
            and "model" in model.named_steps
        ):
            model_available = True

        else:
            model_load_error = (
                "The saved model is not the expected "
                "preprocessing pipeline."
            )

    except Exception as error:
        model_load_error = str(error)

else:
    model_load_error = (
        f"Model file not found: {MODEL_PATH}"
    )


if not model_available:
    st.error(
        "Academic performance model could not be loaded."
    )
    st.code(
        model_load_error or "Unknown model error."
    )


# ============================================================
# FIND TARGET COLUMN
# ============================================================

target_candidates = [
    "final_grade",
    "final_marks",
    "final_score",
    "target",
    "performance",
    "score",
    "marks",
    "grade",
    "result"
]

target_column = next(
    (
        col for col in target_candidates
        if col in df.columns
    ),
    None
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="text-align:center;">
        <div style="font-size:50px;">🎓</div>
        <h2>Student Dashboard</h2>
        <p>AI Academic Analytics</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")



# ============================================================
# ROLE-BASED STUDENT SELECTION
# ============================================================

if "student_id" not in df.columns:
    st.error("Student_ID is missing from database records.")
    st.stop()

student_ids = sorted(
    df["student_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if not student_ids:
    st.warning("No Student_ID values are available.")
    st.stop()

user = st.session_state.get("user")

if not user:
    st.error("Please login to access the dashboard.")
    st.stop()

user_role = str(user.get("role", "")).strip().lower()

if user_role == "student":

    student_id = user.get("student_id")

    if not student_id:
        st.error("No Student ID is linked to this account.")
        st.stop()

    student_id = str(student_id)

    if student_id not in student_ids:
        st.error(
            f"No academic record found for Student ID: {student_id}"
        )
        st.stop()

    st.sidebar.success(f"Logged in: {user['username']}")
    st.sidebar.info(f"Your Student ID: {student_id}")

else:

    student_id = st.sidebar.selectbox(
        "Select Student ID",
        options=student_ids
    )

selected_rows = df[
    df["student_id"].astype(str) == str(student_id)
]

if selected_rows.empty:
    st.error("Selected student record was not found.")
    st.stop()

selected_student = selected_rows.iloc[0]

student_name = safe_text(
    selected_student.get("name", "Student"),
    "Student"
)

department = safe_text(
    selected_student.get("department", "Not available")
)

semester_value = pd.to_numeric(
    selected_student.get("semester", 1),
    errors="coerce"
)

semester = (
    int(semester_value)
    if pd.notna(semester_value) and semester_value >= 1
    else 1
)

st.sidebar.caption(f"Student: {student_name}")
st.sidebar.caption(f"Department: {department}")
st.sidebar.caption(f"Semester: {semester}")

st.sidebar.markdown("---")

st.sidebar.info(
    "This dashboard uses Machine Learning "
    "to analyse academic performance and "
    "provide early-warning insights."
)


# ============================================================
# HEADER
# ============================================================

render_html(f"""
<div class="dashboard-header">
    <h1>🎓 Welcome, {student_name}</h1>
    <p>Student Academic Performance Intelligence Dashboard</p>
</div>
""")


# ============================================================
# STUDENT INFORMATION
# ============================================================

render_html("""
<div class="section-title">
    👤 Student Information
</div>
""")

info_columns = st.columns(4)

student_information = [
    ("👤", "Student", student_name),
    ("🆔", "Student ID", safe_text(student_id)),
    ("🏫", "Department", department),
    ("📚", "Semester", str(semester))
]

for column, information in zip(
    info_columns,
    student_information
):
    icon, title, value = information

    with column:
        render_html(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """)


# ============================================================
# GET STUDENT VALUES
# ============================================================

def get_numeric_value(column_name, default=0):

    if column_name not in selected_student.index:
        return default

    value = pd.to_numeric(
        pd.Series(
            [selected_student[column_name]]
        ),
        errors="coerce"
    ).iloc[0]

    if pd.isna(value):
        return default

    return float(value)


studytime_col = next(
    (
        col for col in [
            "studytime",
            "study_time",
            "study_hours",
            "studyhours",
            "hours_studied"
        ]
        if col in df.columns
    ),
    None
)

absences_col = next(
    (
        col for col in [
            "absences",
            "absence",
            "absent_days",
            "absent",
            "attendance_absences"
        ]
        if col in df.columns
    ),
    None
)

failures_col = next(
    (
        col for col in [
            "failures",
            "failure",
            "previous_failures",
            "backlogs"
        ]
        if col in df.columns
    ),
    None
)

attendance_col = next(
    (
        col for col in [
            "attendance",
            "attendance_percentage",
            "attendance_percent"
        ]
        if col in df.columns
    ),
    None
)

studytime_value = get_numeric_value(
    studytime_col,
    0
)

absences_value = get_numeric_value(
    absences_col,
    0
)

failures_value = get_numeric_value(
    failures_col,
    0
)

attendance_value = get_numeric_value(
    attendance_col,
    0
)

internal_marks_value = get_numeric_value(
    "internal_marks",
    0
)

assignment_marks_value = get_numeric_value(
    "assignment_marks",
    0
)

quiz_marks_value = get_numeric_value(
    "quiz_marks",
    0
)

previous_cgpa_value = get_numeric_value(
    "previous_cgpa",
    0
)

current_cgpa_value = get_numeric_value(
    "current_cgpa",
    0
)

backlogs_value = get_numeric_value(
    "backlogs",
    0
)


# ============================================================
# PREDICTION USING THE SELECTED STUDENT'S DATA
# ============================================================

predicted_score = None
prediction_error = None
input_df = None

if model_available:

    try:
        student_values = {
            str(col).strip().lower(): selected_student[col]
            for col in selected_student.index
        }

        input_data = {}

        for feature in MODEL_FEATURES:

            feature_key = feature.lower()

            if feature_key not in student_values:
                raise ValueError(
                    f"Student record is missing required field: {feature}"
                )

            input_data[feature] = student_values[feature_key]

        input_df = pd.DataFrame(
            [input_data],
            columns=MODEL_FEATURES
        )

        # Convert numeric columns.
        for column in NUMERIC_FEATURES:

            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="coerce"
            )

        # Convert college marks into the 0–100
        # training scale used by the model.
        input_df["Internal_Marks"] = (
            input_df["Internal_Marks"] * 100 / 20
        )

        input_df["Assignment_Marks"] = (
            input_df["Assignment_Marks"] * 100 / 10
        )

        input_df["Quiz_Marks"] = (
            input_df["Quiz_Marks"] * 100 / 10
        )

        # Validate required numeric inputs.
        if input_df[NUMERIC_FEATURES].isnull().any().any():
            raise ValueError(
                "Required academic fields are missing or invalid. "
                "Please update the student's records."
            )

        if not np.isfinite(
            input_df[NUMERIC_FEATURES].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Invalid numeric values found in student records."
            )

        # Predict using the complete saved pipeline.
        prediction = model.predict(input_df)

        predicted_score = float(prediction[0])

        if not np.isfinite(predicted_score):
            raise ValueError(
                "The model returned an invalid prediction."
            )

    except Exception as error:
        prediction_error = str(error)
        predicted_score = None

else:
    prediction_error = (
        model_load_error or
        "Academic prediction model is unavailable."
    )


if predicted_score is None:
    st.warning("Student prediction is currently unavailable.")
    st.error(prediction_error or "Prediction could not be generated.")


# ============================================================
# PERFORMANCE STATUS AND RISK LEVEL
# ============================================================

stored_risk = str(
    selected_student.get("risk_level", "")
).strip().lower()

risk_mapping = {
    "low": "Low Risk",
    "low risk": "Low Risk",
    "moderate": "Moderate Risk",
    "moderate risk": "Moderate Risk",
    "medium": "Moderate Risk",
    "high": "High Risk",
    "high risk": "High Risk"
}

if stored_risk in risk_mapping:

    risk_level = risk_mapping[stored_risk]

elif predicted_score is not None:

    if predicted_score >= 75:
        risk_level = "Low Risk"

    elif predicted_score >= 50:
        risk_level = "Moderate Risk"

    else:
        risk_level = "High Risk"

else:
    risk_level = "Unavailable"


if predicted_score is None:

    performance_status = "Prediction Unavailable"
    performance_class = "status-warning"

elif predicted_score >= 75:

    performance_status = "Excellent Performance"
    performance_class = "status-good"

elif predicted_score >= 60:

    performance_status = "Good Performance"
    performance_class = "status-good"

elif predicted_score >= 40:

    performance_status = "Needs Attention"
    performance_class = "status-warning"

else:

    performance_status = "At Risk"
    performance_class = "status-danger"


# ============================================================
# PERFORMANCE OVERVIEW
# ============================================================

render_html("""
<div class="section-title">
    📊 Academic Performance Overview
</div>
""")

m1, m2, m3, m4 = st.columns(4)

with m1:

    score_display = (
        f"{predicted_score:.2f}/100"
        if predicted_score is not None
        else "Unavailable"
    )

    render_html(f"""
    <div class="metric-card">
        <div class="metric-icon">🎯</div>
        <div class="metric-title">Predicted Score</div>
        <div class="metric-value">{score_display}</div>
        <div class="metric-description">
            AI predicted academic marks
        </div>
    </div>
    """)


with m2:

    render_html(f"""
    <div class="metric-card">
        <div class="metric-icon">📈</div>
        <div class="metric-title">Performance</div>
        <div class="metric-value">{performance_status}</div>
        <div class="metric-description">
            Based on predicted score
        </div>
    </div>
    """)


with m3:

    render_html(f"""
    <div class="metric-card">
        <div class="metric-icon">⚠️</div>
        <div class="metric-title">Risk Level</div>
        <div class="metric-value">{risk_level}</div>
        <div class="metric-description">
            Stored risk classification or indicative estimate
        </div>
    </div>
    """)


with m4:

    render_html(f"""
    <div class="metric-card">
        <div class="metric-icon">📚</div>
        <div class="metric-title">Study Time</div>
        <div class="metric-value">{studytime_value:.1f}</div>
        <div class="metric-description">
            Study-time indicator
        </div>
    </div>
    """)


# ============================================================
# STATUS MESSAGE
# ============================================================

status_icon = (
    "🟢" if risk_level == "Low Risk"
    else "🟡" if risk_level == "Moderate Risk"
    else "🔴" if risk_level == "High Risk"
    else "⚪"
)

render_html(f"""
<div class="{performance_class}" style="margin-top:20px;">
    {status_icon}
    {performance_status}
    — AI Risk Level: {risk_level}
</div>
""")


# ============================================================
# ACADEMIC FACTORS
# ============================================================

render_html("""
<div class="section-title">
    📚 Academic Factors
</div>
""")

factor_columns = st.columns(4)

factors = [
    ("📅", "Absences", f"{absences_value:.0f}"),
    ("❌", "Previous Failures", f"{failures_value:.0f}"),
    ("📖", "Study Time", f"{studytime_value:.1f}"),
    ("🎯", "Attendance", f"{attendance_value:.1f}%")
]

for column, factor in zip(factor_columns, factors):

    icon, title, value = factor

    with column:
        render_html(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """)


# ============================================================
# DATASET SUMMARY
# ============================================================

render_html("""
<div class="section-title">
    🗂️ Dataset Summary
</div>
""")

d1, d2, d3, d4 = st.columns(4)

dataset_metrics = [
    ("👥", "Total Records", str(len(df))),
    ("📊", "Total Columns", str(len(df.columns))),
    (
        "🔢",
        "Numerical Columns",
        str(len(df.select_dtypes(include=np.number).columns))
    ),
    ("🎯", "Target Column", target_column or "Not Found")
]

for column, metric in zip(
    [d1, d2, d3, d4],
    dataset_metrics
):

    icon, title, value = metric

    with column:
        render_html(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """)


# ============================================================
# TARGET ANALYSIS
# ============================================================

render_html("""
<div class="section-title">
    📈 Target Analysis
</div>
""")

if target_column:

    target_values = pd.to_numeric(
        df[target_column],
        errors="coerce"
    ).dropna()

    if not target_values.empty:

        t1, t2, t3 = st.columns(3)

        t1.metric(
            "Average Score",
            f"{target_values.mean():.2f}"
        )

        t2.metric(
            "Highest Score",
            f"{target_values.max():.2f}"
        )

        t3.metric(
            "Lowest Score",
            f"{target_values.min():.2f}"
        )

        st.bar_chart(
            target_values.value_counts().sort_index()
        )

    else:
        st.info("No valid numeric target values were found.")

else:
    st.info(
        "No standard academic target column was detected."
    )


# ============================================================
# FEATURE INFORMATION
# ============================================================

render_html("""
<div class="section-title">
    🔍 Dataset Features
</div>
""")

feature_table = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": [
        str(df[col].dtype)
        for col in df.columns
    ],
    "Missing Values": [
        int(df[col].isna().sum())
        for col in df.columns
    ],
    "Unique Values": [
        int(df[col].nunique())
        for col in df.columns
    ]
})

st.dataframe(
    feature_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RAW DATA PREVIEW
# ============================================================

render_html("""
<div class="section-title">
    👀 Student Dataset Preview
</div>
""")

st.dataframe(
    df.head(10),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# WEAK AREAS AND EARLY WARNING
# ============================================================

render_html("""
<div class="section-title">
    ⚠️ Weak Areas & Early Warning
</div>
""")

weak_areas = []

if predicted_score is not None and predicted_score < 50:

    weak_areas.append(
        "Overall predicted academic performance"
    )

if studytime_value < 3:

    weak_areas.append(
        "Study consistency"
    )

if absences_value > 5:

    weak_areas.append(
        "Attendance / absence management"
    )

if failures_value > 0:

    weak_areas.append(
        "Previous academic failures"
    )

if not weak_areas:

    weak_areas.append(
        "No major academic weakness detected by these indicators"
    )


for area in weak_areas:

    render_html(f"""
    <div class="recommendation-card">
        <h4>⚠️ {html.escape(area)}</h4>
        <p>
            This area may require additional attention
            to improve academic performance.
        </p>
    </div>
    """)


# ============================================================
# PERSONALIZED RECOMMENDATIONS
# ============================================================

render_html("""
<div class="section-title">
    💡 Personalized Recommendations
</div>
""")

recommendations = []

if studytime_value < 3:

    recommendations.append(
        (
            "📖 Increase Study Time",
            "Maintain a consistent daily study schedule "
            "and gradually increase study time."
        )
    )

if absences_value > 5:

    recommendations.append(
        (
            "🏫 Improve Attendance",
            "Regular class attendance can help "
            "understand concepts and improve performance."
        )
    )

if failures_value > 0:

    recommendations.append(
        (
            "👨‍🏫 Academic Support",
            "Seek guidance from faculty members "
            "for subjects where previous difficulties "
            "were observed."
        )
    )

if predicted_score is not None and predicted_score < 50:

    recommendations.append(
        (
            "🎯 Focus on Weak Subjects",
            "Identify difficult subjects and spend "
            "additional practice time on them."
        )
    )

recommendations.append(
    (
        "📝 Regular Revision",
        "Revise important concepts every week "
        "instead of studying only before examinations."
    )
)

for title, description in recommendations:

    render_html(f"""
    <div class="recommendation-card">
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    """)


# ============================================================
# EXPLAINABLE AI — EXISTING SHAP REPORTS
# ============================================================

render_html("""
<div class="section-title">
    🔎 Explainable AI — Global Analysis
</div>
""")

st.info(
    "SHAP explains how input features influence model predictions. "
    "The following saved charts describe previous model analysis; "
    "they are not necessarily explanations for the selected student."
)

shap_global_path = os.path.join(
    BASE_DIR,
    "eda_graphs",
    "20_shap_global_importance.png"
)

shap_summary_path = os.path.join(
    BASE_DIR,
    "eda_graphs",
    "21_shap_summary.png"
)

shap_individual_path = os.path.join(
    BASE_DIR,
    "eda_graphs",
    "22_individual_student_explanation.png"
)


if os.path.isfile(shap_global_path):

    st.subheader("📊 Global Feature Importance")

    st.image(
        shap_global_path,
        use_container_width=True
    )

else:
    st.warning("Global SHAP importance chart was not found.")


if os.path.isfile(shap_summary_path):

    st.subheader("📈 SHAP Summary Plot")

    st.image(
        shap_summary_path,
        use_container_width=True
    )

else:
    st.warning("SHAP summary chart was not found.")


if os.path.isfile(shap_individual_path):

    st.subheader("👤 Previous Individual Explanation")

    st.caption(
        "This is a previously generated example, "
        "not necessarily the selected student."
    )

    st.image(
        shap_individual_path,
        use_container_width=True
    )

else:
    st.warning("Individual SHAP example chart was not found.")


# ============================================================
# EXISTING SHAP CSV
# ============================================================

shap_csv_path = os.path.join(
    BASE_DIR,
    "data",
    "individual_student_explanation.csv"
)

if os.path.isfile(shap_csv_path):

    try:
        shap_df = pd.read_csv(shap_csv_path)

        required_columns = {
            "Feature",
            "Feature_Value",
            "SHAP_Value",
            "Absolute_SHAP"
        }

        if required_columns.issubset(shap_df.columns):

            shap_df = shap_df.sort_values(
                "Absolute_SHAP",
                ascending=False
            )

            st.subheader(
                "📋 Previous Feature-Level Explanation"
            )

            st.dataframe(
                shap_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning(
                "The SHAP CSV does not contain the expected columns."
            )

    except Exception as error:
        st.warning(
            f"Unable to display the SHAP CSV: {error}"
        )


# ============================================================
# STUDENT-SPECIFIC SHAP EXPLANATION
# ============================================================

render_html("""
<div class="section-title">
    🔎 Explainable AI — Student-Specific Analysis
</div>
""")

st.info(
    "This section explains how academic features influence "
    "the selected student's predicted marks. SHAP values "
    "describe model behavior, not cause and effect."
)

if predicted_score is None or input_df is None:

    st.warning(
        "A valid student prediction is required "
        "before generating a student-specific SHAP explanation."
    )

elif model_available:

    try:
        import shap
        import matplotlib.pyplot as plt

        shap_preprocessor = model.named_steps["preprocessor"]
        shap_regressor = model.named_steps["model"]

        # Use the same validated input used for prediction.
        selected_input = input_df.copy()

        # Prepare background data from the database.
        background = df.copy()

        background.columns = (
            background.columns.astype(str)
            .str.strip()
            .str.lower()
        )

        missing_features = [
            feature
            for feature in MODEL_FEATURES
            if feature.lower() not in background.columns
        ]

        if missing_features:

            st.warning(
                "Missing model features in database: "
                f"{missing_features}"
            )

        else:

            background = background[
                [feature.lower() for feature in MODEL_FEATURES]
            ].copy()

            background.columns = MODEL_FEATURES

            # Convert numeric background fields.
            for column in NUMERIC_FEATURES:

                background[column] = pd.to_numeric(
                    background[column],
                    errors="coerce"
                )

            # Convert college marks to training scale.
            background["Internal_Marks"] = (
                background["Internal_Marks"] * 100 / 20
            )

            background["Assignment_Marks"] = (
                background["Assignment_Marks"] * 100 / 10
            )

            background["Quiz_Marks"] = (
                background["Quiz_Marks"] * 100 / 10
            )

            # Remove rows with missing model inputs.
            background = background.dropna(
                subset=MODEL_FEATURES
            )

            if background.empty:

                st.warning(
                    "There are not enough complete student "
                    "records to calculate SHAP."
                )

            else:

                # Use a small background sample.
                background = background.head(50)

                # Transform using the trained preprocessing pipeline.
                transformed_background = (
                    shap_preprocessor.transform(background)
                )

                transformed_student = (
                    shap_preprocessor.transform(selected_input)
                )

                feature_names = (
                    shap_preprocessor.get_feature_names_out()
                )

                # Convert sparse matrices if necessary.
                if hasattr(
                    transformed_background,
                    "toarray"
                ):
                    transformed_background = (
                        transformed_background.toarray()
                    )

                if hasattr(
                    transformed_student,
                    "toarray"
                ):
                    transformed_student = (
                        transformed_student.toarray()
                    )

                # Explain the trained estimator.
                explainer = shap.Explainer(
                    shap_regressor.predict,
                    transformed_background,
                    feature_names=feature_names
                )

                shap_values = explainer(
                    transformed_student,
                    max_evals=2 * len(feature_names) + 1
                )

                values = np.asarray(
                    shap_values.values
                ).reshape(-1)

                base_value = float(
                    np.asarray(
                        shap_values.base_values
                    ).reshape(-1)[0]
                )

                explanation_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP_Value": values,
                    "Absolute_SHAP": np.abs(values)
                }).sort_values(
                    "Absolute_SHAP",
                    ascending=False
                )

                st.metric(
                    "Selected Student's Predicted Marks",
                    f"{predicted_score:.2f}/100"
                )

                st.caption(
                    f"SHAP baseline prediction: {base_value:.2f}. "
                    "Positive SHAP values push the prediction "
                    "higher than the baseline; negative values "
                    "push it lower."
                )

                st.subheader(
                    "📊 Factors Influencing This Prediction"
                )

                top_features = (
                    explanation_df.head(10)
                    .sort_values("SHAP_Value")
                )

                fig, ax = plt.subplots(
                    figsize=(10, 6)
                )

                ax.barh(
                    top_features["Feature"],
                    top_features["SHAP_Value"]
                )

                ax.axvline(0, linewidth=1)

                ax.set_xlabel(
                    "SHAP value (effect on model output)"
                )

                ax.set_title(
                    "SHAP Explanation for Selected Student"
                )

                fig.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

                st.subheader("📋 Student SHAP Values")

                st.dataframe(
                    explanation_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "📥 Download Student SHAP Explanation",
                    data=explanation_df.to_csv(index=False),
                    file_name=f"shap_explanation_{student_id}.csv",
                    mime="text/csv"
                )

    except Exception as error:

        st.warning(
            "The student-specific SHAP explanation "
            "could not be generated."
        )

        st.code(str(error))


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">
    <p>
        Trekverse — Intelligent Academic Performance
        Monitoring and Recommendation System
    </p>
    <p>
        AI-Powered Student Academic Analytics
    </p>
</div>
""")
