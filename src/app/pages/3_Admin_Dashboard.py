
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Trekverse | Admin Dashboard",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from access_control import require_role, logout_button


user = require_role(["Admin"])
logout_button()

DATA_PATH = PROJECT_DIR / "data" / "student_academic_data.csv"
REPORT_PATH = PROJECT_DIR / "reports" / "early_warning_report.csv"
MODEL_PATH = PROJECT_DIR / "models" / "academic_performance_model.pkl"
CLUSTER_PATH = PROJECT_DIR / "data" / "cluster_profiles.csv"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .stApp {
        background: linear-gradient(
            135deg, #f8fbff 0%, #eef5ff 50%, #f8fbff 100%
        );
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg, #172554 0%, #1e3a8a 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        background: linear-gradient(
            135deg, #172554, #1d4ed8, #2563eb
        );
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 35px;
        margin: 0;
    }

    .hero p {
        color: #dbeafe;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-header {
        background: white;
        padding: 18px 22px;
        border-radius: 14px;
        margin-top: 24px;
        margin-bottom: 16px;
        border-left: 6px solid #2563eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
    }

    .section-header h2 {
        color: #172554;
        margin: 0;
        font-size: 23px;
    }

    .section-header p {
        color: #64748b;
        margin: 5px 0 0 0;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 28px;
        margin-top: 25px;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_student_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_risk_report():
    if not REPORT_PATH.exists():
        return None

    return pd.read_csv(REPORT_PATH)


def show_section(title, description):
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOAD CURRENT DATASET
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>👨‍💼 Trekverse Admin Dashboard</h1>
        <p>
            Intelligent Academic Performance Monitoring,
            Risk Analysis and Model Management
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if st.sidebar.button("🔄 Refresh Dashboard", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

try:
    if not DATA_PATH.exists():
        st.error(f"Student dataset not found: {DATA_PATH}")
        st.stop()

    df = load_student_data()

    if df.empty:
        st.error("The student dataset is empty.")
        st.stop()

except Exception as error:
    st.error("Unable to load the student dataset.")
    st.code(str(error))
    st.stop()

required_columns = [
    "Student_ID",
    "Department",
    "Semester",
    "Final_Marks"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Required columns are missing: {missing_columns}")
    st.stop()

# ============================================================
# DATA QUALITY
# ============================================================

show_section(
    "📊 Dataset Overview",
    "Summary and data quality of the current academic dataset"
)

total_students = len(df)
total_columns = len(df.columns)
missing_values = int(df.isnull().sum().sum())
duplicate_rows = int(df.duplicated().sum())

c1, c2, c3, c4 = st.columns(4)

c1.metric("👨‍🎓 Total Students", total_students)
c2.metric("📋 Total Columns", total_columns)
c3.metric("⚠️ Missing Values", missing_values)
c4.metric("🔁 Duplicate Rows", duplicate_rows)

# ============================================================
# RISK REPORT
# ============================================================

report = None
report_error = None

try:
    report = load_risk_report()
except Exception as error:
    report_error = str(error)

if report is not None:
    if "Student_ID" not in report.columns:
        report_error = "The report does not contain Student_ID."
        report = None
    elif "Risk_Level" not in report.columns:
        report_error = "The report does not contain Risk_Level."
        report = None
    else:
        report = report.drop_duplicates(subset=["Student_ID"])

if report is not None:
    report_columns = [
        column for column in report.columns
        if column == "Student_ID" or column not in df.columns
    ]

    try:
        dashboard_df = df.merge(
            report[report_columns],
            on="Student_ID",
            how="left",
            validate="one_to_one"
        )
    except Exception as error:
        st.error("Unable to connect student records with the risk report.")
        st.code(str(error))
        st.stop()
else:
    dashboard_df = df.copy()

if "Risk_Level" not in dashboard_df.columns:
    dashboard_df["Risk_Level"] = "Not Available"

dashboard_df["Risk_Level"] = dashboard_df["Risk_Level"].fillna(
    "Not Available"
)

if "Predicted_Final_Marks" in dashboard_df.columns:
    dashboard_df["Predicted_Final_Marks"] = pd.to_numeric(
        dashboard_df["Predicted_Final_Marks"],
        errors="coerce"
    )

# ============================================================
# RISK DISTRIBUTION
# ============================================================

show_section(
    "🚦 Student Risk Monitoring",
    "Risk categories generated by the Early Warning System"
)

risk_counts = (
    dashboard_df["Risk_Level"]
    .value_counts()
    .reindex(
        ["High Risk", "Moderate Risk", "Low Risk"],
        fill_value=0
    )
)

r1, r2, r3 = st.columns(3)

r1.metric("🔴 High Risk", int(risk_counts["High Risk"]))
r2.metric("🟡 Moderate Risk", int(risk_counts["Moderate Risk"]))
r3.metric("🟢 Low Risk", int(risk_counts["Low Risk"]))

if report is None:
    st.warning(
        "The early-warning report is not available. "
        "Generate it from the project terminal using:"
    )
    st.code("python src\\early_warning.py")

    if report_error:
        st.caption(f"Report loading detail: {report_error}")

else:
    risk_chart = risk_counts[
        ["High Risk", "Moderate Risk", "Low Risk"]
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(risk_chart.index, risk_chart.values)
    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Number of Students")
    ax.set_title("Student Risk Distribution")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Risk levels are estimates based on predicted final marks "
        "and the provisional thresholds configured in the project."
    )

# ============================================================
# DEPARTMENT ANALYSIS
# ============================================================

show_section(
    "🏫 Department Analysis",
    "Compare student numbers and recorded academic performance"
)

department_summary = (
    dashboard_df.groupby("Department", dropna=False)
    .agg(
        Total_Students=("Student_ID", "count"),
        Average_Final_Marks=("Final_Marks", "mean")
    )
    .reset_index()
)

department_summary["Average_Final_Marks"] = (
    department_summary["Average_Final_Marks"].round(2)
)

st.dataframe(
    department_summary,
    use_container_width=True,
    hide_index=True
)

if not department_summary.empty:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        department_summary["Department"].astype(str),
        department_summary["Total_Students"]
    )
    ax.set_xlabel("Department")
    ax.set_ylabel("Number of Students")
    ax.set_title("Students by Department")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# SEMESTER ANALYSIS
# ============================================================

show_section(
    "📚 Semester Analysis",
    "Review recorded final marks across semesters"
)

semester_summary = (
    dashboard_df.groupby("Semester", dropna=False)
    .agg(
        Total_Students=("Student_ID", "count"),
        Average_Final_Marks=("Final_Marks", "mean")
    )
    .reset_index()
)

semester_summary["Average_Final_Marks"] = (
    semester_summary["Average_Final_Marks"].round(2)
)

st.dataframe(
    semester_summary,
    use_container_width=True,
    hide_index=True
)

semester_plot = semester_summary.dropna(
    subset=["Average_Final_Marks"]
)

if not semester_plot.empty:
    semester_plot = semester_plot.sort_values("Semester")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        semester_plot["Semester"].astype(str),
        semester_plot["Average_Final_Marks"],
        marker="o"
    )
    ax.set_xlabel("Semester")
    ax.set_ylabel("Average Final Marks")
    ax.set_title("Average Final Marks by Semester")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# STUDENT RECORDS
# ============================================================

show_section(
    "🔎 Student Records",
    "Search and filter student academic information"
)

filter_col1, filter_col2 = st.columns(2)

departments = sorted(
    dashboard_df["Department"].dropna().astype(str).unique().tolist()
)

with filter_col1:
    selected_department = st.selectbox(
        "Filter by Department",
        ["All"] + departments
    )

with filter_col2:
    selected_risk = st.selectbox(
        "Filter by Risk Level",
        ["All", "High Risk", "Moderate Risk", "Low Risk", "Not Available"]
    )

filtered_df = dashboard_df.copy()

if selected_department != "All":
    filtered_df = filtered_df[
        filtered_df["Department"].astype(str) == selected_department
    ]

if selected_risk != "All":
    filtered_df = filtered_df[
        filtered_df["Risk_Level"] == selected_risk
    ]

search = st.text_input(
    "Search by student ID or name",
    placeholder="Example: STU111"
)

if search.strip():
    search_mask = (
        filtered_df["Student_ID"]
        .astype(str)
        .str.contains(search, case=False, na=False)
    )

    if "Name" in filtered_df.columns:
        search_mask |= (
            filtered_df["Name"]
            .astype(str)
            .str.contains(search, case=False, na=False)
        )

    filtered_df = filtered_df[search_mask]

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# SELECT AND VIEW INDIVIDUAL STUDENT
# ============================================================

if not filtered_df.empty:
    student_options = (
        filtered_df["Student_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_student_id = st.selectbox(
        "Select Student to View Details",
        student_options
    )

    selected_student = filtered_df[
        filtered_df["Student_ID"].astype(str) == selected_student_id
    ]

    if not selected_student.empty:
        st.subheader("Selected Student Details")
        st.dataframe(
            selected_student,
            use_container_width=True,
            hide_index=True
        )
else:
    st.warning("No student records found for the selected filters.")

# ============================================================
# ACADEMIC PERFORMANCE
# ============================================================

show_section(
    "📈 Academic Performance",
    "Distribution of recorded final marks"
)

valid_marks = pd.to_numeric(
    dashboard_df["Final_Marks"],
    errors="coerce"
).dropna()

if not valid_marks.empty:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(valid_marks, bins=10)
    ax.set_xlabel("Actual Final Marks")
    ax.set_ylabel("Number of Students")
    ax.set_title("Final Marks Distribution")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.metric("Overall Average Final Marks", f"{valid_marks.mean():.2f}")
else:
    st.info("No valid final marks are available.")

# ============================================================
# MODEL STATUS
# ============================================================

show_section(
    "🤖 Machine Learning Model Status",
    "Check the availability of the current academic model"
)

if MODEL_PATH.exists():
    st.success("✅ Academic performance model is available.")
    st.write(f"Model file: `{MODEL_PATH.relative_to(PROJECT_DIR)}`")
else:
    st.error("Academic performance model file was not found.")
    st.write(f"Expected location: `{MODEL_PATH}`")

st.info(
    "The current training workflow selected Ridge Regression "
    "for academic performance prediction. Use the saved evaluation "
    "results to report model metrics; do not assume other models "
    "are trained or deployed just because they are listed in the project."
)

results_files = [
    PROJECT_DIR / "final_model_test_results.csv",
    PROJECT_DIR / "new_model_results.csv",
    PROJECT_DIR / "models" / "final_model_test_results.csv",
    PROJECT_DIR / "models" / "new_model_results.csv"
]

found_results = [path for path in results_files if path.exists()]

if found_results:
    st.subheader("Available Model Evaluation Files")

    for result_path in found_results:
        st.write(f"✅ `{result_path.relative_to(PROJECT_DIR)}`")

        try:
            results_df = pd.read_csv(result_path)
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )
        except Exception:
            st.warning("This results file could not be displayed.")
else:
    st.info(
        "No known model evaluation CSV was found at the expected "
        "locations. Check the output location used by the training script."
    )

# ============================================================
# CLUSTER INFORMATION
# ============================================================

show_section(
    "🔬 Student Cluster Information",
    "Display cluster profiles if clustering results are available"
)

if CLUSTER_PATH.exists():
    try:
        cluster_df = pd.read_csv(CLUSTER_PATH)

        st.dataframe(
            cluster_df,
            use_container_width=True,
            hide_index=True
        )
    except Exception as error:
        st.warning("Unable to read the cluster profile file.")
        st.code(str(error))
else:
    st.info(
        "Cluster profiles are not currently available. "
        "This optional section will display data when the cluster "
        "profile file has been generated."
    )

# ============================================================
# DATA QUALITY DETAILS
# ============================================================

show_section(
    "🧹 Data Quality Details",
    "Review missing data and duplicate records"
)

quality_col1, quality_col2 = st.columns(2)

with quality_col1:
    st.subheader("Missing Values by Column")
    missing_by_column = (
        df.isnull()
        .sum()
        .rename("Missing Values")
        .reset_index()
    )
    missing_by_column.columns = ["Column", "Missing Values"]

    st.dataframe(
        missing_by_column,
        use_container_width=True,
        hide_index=True
    )

with quality_col2:
    st.subheader("Dataset Details")
    st.write(f"**Dataset:** `{DATA_PATH.relative_to(PROJECT_DIR)}`")
    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")
    st.write(f"**Duplicate rows:** {duplicate_rows}")
    st.subheader("Download Data Quality Report")

    quality_report = missing_by_column.copy()

    quality_report["Data Quality Status"] = (
        quality_report["Missing Values"]
        .apply(
            lambda count: (
                "Needs Review" if count > 0 else "Complete"
            )
        )
    )

    st.download_button(
        "Download Data Quality Report",
        data=quality_report.to_csv(index=False).encode("utf-8"),
        file_name="trekverse_data_quality_report.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================
# DOWNLOAD REPORTS
# ============================================================

show_section(
    "📥 Download Reports",
    "Export student records and the early-warning report"
)

download_col1, download_col2 = st.columns(2)

with download_col1:
    st.download_button(
        "⬇️ Download Filtered Student Data",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="trekverse_student_data.csv",
        mime="text/csv",
        use_container_width=True
    )

with download_col2:
    if report is not None:
        st.download_button(
            "⬇️ Download Early-Warning Report",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="trekverse_early_warning_report.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.button(
            "Early-Warning Report Unavailable",
            disabled=True,
            use_container_width=True
        )

# ============================================================
# PROJECT SUMMARY
# ============================================================

show_section(
    "📝 Project Summary",
    "Overview of the Trekverse academic monitoring system"
)

st.info(
    """
    Trekverse is an Intelligent Academic Performance Monitoring
    and Early-Warning System. It uses student academic information
    to estimate final marks, categorize provisional academic risk,
    and provide faculty with information for student support.

    The project also includes SHAP-based explainability to help
    inspect factors influencing model predictions.

    Predictions and risk categories should support human review,
    not replace faculty judgment. The dataset is limited in size,
    and risk thresholds should be validated against institutional
    academic policies before practical deployment.
    """
)
# ============================================================
# ATTENDANCE ANALYTICS
# ============================================================

show_section(
    "📅 Attendance Analytics",
    "Monitor attendance and identify students who may need support"
)

attendance = pd.to_numeric(
    dashboard_df["Attendance"], errors="coerce"
)

valid_attendance = attendance.dropna()

avg_attendance = (
    valid_attendance.mean()
    if not valid_attendance.empty else 0
)

low_attendance_count = int(
    (valid_attendance < 75).sum()
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Attendance",
    f"{avg_attendance:.2f}%" if not valid_attendance.empty else "N/A"
)

col2.metric(
    "Students Below 75%",
    low_attendance_count
)

col3.metric(
    "Students With Attendance Data",
    len(valid_attendance)
)

if not valid_attendance.empty:
    attendance_chart = pd.DataFrame({
        "Attendance Category": [
            "Below 75%",
            "75% or Above"
        ],
        "Students": [
            int((valid_attendance < 75).sum()),
            int((valid_attendance >= 75).sum())
        ]
    })

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        attendance_chart["Attendance Category"],
        attendance_chart["Students"]
    )

    ax.set_title("Student Attendance Overview")
    ax.set_ylabel("Number of Students")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# CGPA ANALYTICS
# ============================================================

show_section(
    "🎓 CGPA Analytics",
    "Review current CGPA and compare academic performance by department"
)

cgpa = pd.to_numeric(
    dashboard_df["Current_CGPA"],
    errors="coerce"
)

valid_cgpa = cgpa.dropna()

if not valid_cgpa.empty:
    cgpa_col1, cgpa_col2 = st.columns(2)

    cgpa_col1.metric(
        "Average Current CGPA",
        f"{valid_cgpa.mean():.2f}"
    )

    cgpa_col2.metric(
        "Students With CGPA Data",
        len(valid_cgpa)
    )

    cgpa_data = dashboard_df.copy()

    cgpa_data["Current_CGPA"] = pd.to_numeric(
        cgpa_data["Current_CGPA"],
        errors="coerce"
    )

    department_cgpa = (
        cgpa_data.groupby("Department", dropna=False)
        .agg(
            Average_CGPA=("Current_CGPA", "mean"),
            Students=("Student_ID", "count")
        )
        .reset_index()
    )

    department_cgpa["Average_CGPA"] = (
        department_cgpa["Average_CGPA"].round(2)
    )

    st.subheader("Department-wise Average CGPA")

    st.dataframe(
        department_cgpa,
        use_container_width=True,
        hide_index=True
    )

    chart_data = department_cgpa.dropna(
        subset=["Average_CGPA"]
    )

    if not chart_data.empty:
        fig, ax = plt.subplots(figsize=(9, 4))

        ax.bar(
            chart_data["Department"].astype(str),
            chart_data["Average_CGPA"]
        )

        ax.set_title("Average CGPA by Department")
        ax.set_xlabel("Department")
        ax.set_ylabel("Average CGPA")

        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

else:
    st.info("No valid Current_CGPA values are available.")

# ============================================================
# STUDENT PERFORMANCE ALERTS
# ============================================================

show_section(
    "🚨 Student Performance Alerts",
    "Identify students who may benefit from academic support"
)

alerts_df = dashboard_df.copy()

alerts_df["Attendance"] = pd.to_numeric(
    alerts_df["Attendance"], errors="coerce"
)

alerts_df["Final_Marks"] = pd.to_numeric(
    alerts_df["Final_Marks"], errors="coerce"
)

alerts_df["Backlogs"] = pd.to_numeric(
    alerts_df["Backlogs"], errors="coerce"
)

# Standardize risk labels for comparison.
# The dataset may store High, Moderate, Low rather than
# High Risk, Moderate Risk, Low Risk.
risk_normalized = (
    alerts_df["Risk_Level"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s*risk$", "", regex=True)
)

high_risk = risk_normalized.eq("high")
moderate_risk = risk_normalized.eq("moderate")
low_attendance = alerts_df["Attendance"] < 75
low_marks = alerts_df["Final_Marks"] < 40
has_backlogs = alerts_df["Backlogs"] > 0

# Generate reasons for review.
alerts_df["Alert Reasons"] = ""

alerts_df.loc[
    high_risk,
    "Alert Reasons"
] += "High recorded risk; "

alerts_df.loc[
    moderate_risk,
    "Alert Reasons"
] += "Moderate recorded risk; "

alerts_df.loc[
    low_attendance.fillna(False),
    "Alert Reasons"
] += "Attendance below 75%; "

alerts_df.loc[
    low_marks.fillna(False),
    "Alert Reasons"
] += "Final marks below 40; "

alerts_df.loc[
    has_backlogs.fillna(False),
    "Alert Reasons"
] += "Pending backlogs; "

# Include students with at least one alert reason.
has_alert = alerts_df["Alert Reasons"].str.strip().ne("")

alert_columns = [
    "Student_ID",
    "Name",
    "Department",
    "Semester",
    "Attendance",
    "Final_Marks",
    "Current_CGPA",
    "Backlogs",
    "Risk_Level",
    "Alert Reasons"
]

available_alert_columns = [
    column for column in alert_columns
    if column in alerts_df.columns
]

student_alerts = alerts_df.loc[
    has_alert,
    available_alert_columns
].copy()
# Assign priority based on alert conditions.
student_alerts["Alert Priority"] = "Low"

student_alerts.loc[
    student_alerts["Alert Reasons"].str.contains(
        "Moderate recorded risk", na=False
    ),
    "Alert Priority"
] = "Medium"

student_alerts.loc[
    student_alerts["Alert Reasons"].str.contains(
        "High recorded risk", na=False
    ),
    "Alert Priority"
] = "High"

# Show High priority alerts first.
priority_order = {
    "High": 0,
    "Medium": 1,
    "Low": 2
}

student_alerts["_Priority_Order"] = (
    student_alerts["Alert Priority"].map(priority_order)
)

student_alerts = (
    student_alerts
    .sort_values("_Priority_Order")
    .drop(columns="_Priority_Order")
)
if student_alerts.empty:
    st.success(
        "No students matched the current alert rules."
    )
else:
    st.warning(
        f"{len(student_alerts)} student record(s) need review "
        "under the current alert rules."
    )

    st.dataframe(
        student_alerts,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Student Performance Alerts",
        data=student_alerts.to_csv(index=False).encode("utf-8"),
        file_name="trekverse_student_performance_alerts.csv",
        mime="text/csv",
        use_container_width=True
    )

st.caption(
    "Alert rules are preliminary screening criteria, not definitive "
    "judgments about students. Faculty should review the context "
    "before taking action."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        👨‍💼 Trekverse Admin Dashboard
        <br>
        Machine Learning • Explainable AI • Early Warning
        <br><br>
        Intelligent Academic Performance Monitoring System
    </div>
    """,
    unsafe_allow_html=True
)
# ============================================================
# ADMIN USER MANAGEMENT
# ============================================================

