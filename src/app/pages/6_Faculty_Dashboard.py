import sys
from pathlib import Path
from io import BytesIO
from xml.sax.saxutils import escape as xml_escape

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Trekverse | Faculty Dashboard",
    page_icon="👨‍🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_PATH = PROJECT_DIR / "data" / "student_academic_data.csv"
REPORT_PATH = PROJECT_DIR / "reports" / "early_warning_report.csv"

# ============================================================
# ACCESS CONTROL
# ============================================================

from access_control import require_role, logout_button

user = require_role(["Faculty", "Admin"])
logout_button()

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
        background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 50%, #f8fbff 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #172554 0%, #1e3a8a 100%);
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    .hero {
        background: linear-gradient(135deg, #172554, #1d4ed8, #2563eb);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 24px;
    }
    .hero h1 {
        color: white;
        font-size: 36px;
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
        font-size: 24px;
    }
    .section-header p {
        color: #64748b;
        margin: 5px 0 0 0;
        font-size: 14px;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        min-height: 125px;
    }
    .metric-icon {
        font-size: 26px;
    }
    .metric-title {
        color: #64748b;
        font-size: 14px;
    }
    .metric-value {
        color: #172554;
        font-size: 29px;
        font-weight: bold;
    }
    .metric-subtitle {
        color: #94a3b8;
        font-size: 12px;
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
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=30)
def load_student_data():
    """Load student records from SQLite first, with CSV as a fallback."""
    try:
        from database import get_students

        records = get_students()
        if records:
            students_df = pd.DataFrame([dict(record) for record in records])
            # Convert database field names to the column names used by this dashboard.
            column_map = {
                "student_id": "Student_ID",
                "student_name": "Name",
                "department": "Department",
                "semester": "Semester",
                "attendance": "Attendance",
                "internal_marks": "Internal_Marks",
                "assignment_marks": "Assignment_Marks",
                "quiz_marks": "Quiz_Marks",
                "study_hours": "Study_Hours",
                "previous_cgpa": "Previous_CGPA",
                "current_cgpa": "Current_CGPA",
                "backlogs": "Backlogs",
                "assignment_submission_rate": "Assignment_Submission_Rate",
                "learning_activity": "Learning_Activity",
                "final_marks": "Final_Marks",
                "mathematics_marks": "Mathematics_Marks",
                "programming_marks": "Programming_Marks",
                "dbms_marks": "DBMS_Marks",
                "computer_networks_marks": "Computer_Networks_Marks",
                "electronics_marks": "Electronics_Marks",
                "semester_1_cgpa": "Semester_1_CGPA",
                "semester_2_cgpa": "Semester_2_CGPA",
                "semester_3_cgpa": "Semester_3_CGPA",
                "semester_4_cgpa": "Semester_4_CGPA",
                "risk_level": "Risk_Level",
                "predicted_score": "Predicted_Final_Marks",
                "performance_level": "Performance_Level",
                "early_warning": "Early_Warning",
            }
            students_df = students_df.rename(columns=column_map)
            # Prefer the dedicated Final_Marks field when available; use legacy predicted score only if needed.
            if "Final_Marks" not in students_df.columns:
                students_df["Final_Marks"] = pd.NA
            if "Student_ID" in students_df.columns:
                students_df["Student_ID"] = students_df["Student_ID"].astype(str).str.strip()
            return students_df

    except Exception as database_error:
        # Continue with the CSV fallback so the dashboard can still open.
        st.warning(f"Could not read student records from SQLite; using CSV fallback. Details: {database_error}")

    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


@st.cache_data
def load_early_warning_report():
    if not REPORT_PATH.exists():
        return None
    return pd.read_csv(REPORT_PATH)


try:
    students = load_student_data()
    report = load_early_warning_report()

except Exception as error:
    st.error("Unable to load the student data.")
    st.code(str(error))
    st.stop()

if students.empty:
    st.warning("The student dataset contains no records.")
    st.stop()

# ============================================================
# REQUIRED COLUMN VALIDATION
# ============================================================

required_columns = ["Student_ID", "Department", "Semester", "Final_Marks"]
missing_columns = [column for column in required_columns if column not in students.columns]

if missing_columns:
    st.error(f"Required columns are missing from the dataset: {missing_columns}")
    st.stop()

# ============================================================
# MERGE DATA WITH EARLY-WARNING REPORT
# ============================================================

if report is not None and "Student_ID" in report.columns:
    report = report.drop_duplicates(subset=["Student_ID"])

    # Dataset values take precedence if a column already exists there.
    report_columns = [
        column for column in report.columns
        if column == "Student_ID" or column not in students.columns
    ]

    try:
        df = students.merge(
            report[report_columns],
            on="Student_ID",
            how="left",
            validate="one_to_one",
        )
    except Exception as error:
        st.error("Unable to merge the student dataset with the early-warning report.")
        st.code(str(error))
        st.stop()
else:
    df = students.copy()

# Ensure columns used later exist, even if the report is not available.
if "Predicted_Final_Marks" not in df.columns:
    df["Predicted_Final_Marks"] = pd.NA

if "Risk_Level" not in df.columns:
    st.warning(
        "Risk_Level is missing from the dataset and early-warning report. "
        "Run `python src\\early_warning.py` and refresh this page."
    )
    st.stop()

df["Predicted_Final_Marks"] = pd.to_numeric(
    df["Predicted_Final_Marks"], errors="coerce"
)
df["Final_Marks"] = pd.to_numeric(df["Final_Marks"], errors="coerce")
df["Risk_Level"] = df["Risk_Level"].fillna("Not Available").astype(str)

# Normalize risk labels for dashboard calculations and filtering.
# Supports both "High" and "High Risk" style labels.
risk_label_map = {
    "high": "High Risk",
    "high risk": "High Risk",
    "moderate": "Moderate Risk",
    "moderate risk": "Moderate Risk",
    "medium": "Moderate Risk",
    "medium risk": "Moderate Risk",
    "low": "Low Risk",
    "low risk": "Low Risk",
}
df["Risk_Level"] = df["Risk_Level"].str.strip().str.lower().map(
    risk_label_map
).fillna(df["Risk_Level"])

# ============================================================
# PDF GENERATION FUNCTION
# ============================================================

def create_student_pdf(student_record):
    """Create a downloadable PDF report for one student."""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TrekverseReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1d4ed8"),
        spaceAfter=12,
    )

    story = [
        Paragraph("TREKVERSE", title_style),
        Paragraph("Student Academic Performance Report", styles["Heading2"]),
        Spacer(1, 12),
        Paragraph("Student Profile", styles["Heading2"]),
    ]

    def printable_value(value):
        if pd.isna(value):
            return "Not available"
        return str(value)

    profile_data = [
        ["Student ID", printable_value(student_record.get("Student_ID", "N/A"))],
        ["Name", printable_value(student_record.get("Name", "N/A"))],
        ["Department", printable_value(student_record.get("Department", "N/A"))],
        ["Semester", printable_value(student_record.get("Semester", "N/A"))],
    ]

    profile_table = Table(profile_data, colWidths=[2 * inch, 4 * inch])
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#dbeafe")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([profile_table, Spacer(1, 18)])

    academic_fields = [
        "Attendance", "Internal_Marks", "Assignment_Marks", "Quiz_Marks",
        "Study_Hours", "Previous_CGPA", "Current_CGPA", "Backlogs",
        "Assignment_Submission_Rate", "Final_Marks",
        "Predicted_Final_Marks", "Risk_Level",
    ]
    academic_data = [["Academic Factor", "Value"]]
    for field in academic_fields:
        if field in student_record.index:
            academic_data.append([
                field.replace("_", " "),
                printable_value(student_record[field]),
            ])

    story.append(Paragraph("Academic Details", styles["Heading2"]))
    academic_table = Table(academic_data, colWidths=[3 * inch, 3 * inch], repeatRows=1)
    academic_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([academic_table, Spacer(1, 18)])

    subject_fields = [
        "Mathematics_Marks", "Programming_Marks", "DBMS_Marks",
        "Computer_Networks_Marks", "Electronics_Marks",
    ]
    available_subjects = [field for field in subject_fields if field in student_record.index]
    if available_subjects:
        story.append(Paragraph("Subject-wise Marks", styles["Heading2"]))
        subject_data = [["Subject", "Marks"]]
        for field in available_subjects:
            subject_data.append([
                field.replace("_Marks", "").replace("_", " "),
                printable_value(student_record[field]),
            ])
        subject_table = Table(subject_data, colWidths=[3 * inch, 3 * inch], repeatRows=1)
        subject_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([subject_table, Spacer(1, 18)])

    history_fields = [
        "Semester_1_CGPA", "Semester_2_CGPA",
        "Semester_3_CGPA", "Semester_4_CGPA",
    ]
    available_history = [field for field in history_fields if field in student_record.index]
    if available_history:
        story.append(Paragraph("Semester-wise CGPA History", styles["Heading2"]))
        history_data = [["Semester", "CGPA"]]
        for field in available_history:
            history_data.append([
                field.replace("_CGPA", "").replace("_", " "),
                printable_value(student_record[field]),
            ])
        history_table = Table(history_data, colWidths=[3 * inch, 3 * inch])
        history_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([history_table, Spacer(1, 18)])

    if "Recommendation" in student_record.index and pd.notna(student_record["Recommendation"]):
        story.append(Paragraph("Academic Recommendation", styles["Heading2"]))
        safe_recommendation = xml_escape(str(student_record["Recommendation"]))
        story.extend([Paragraph(safe_recommendation, styles["BodyText"]), Spacer(1, 12)])

    story.extend([
        Spacer(1, 12),
        Paragraph(
            "This report supports academic monitoring. Predictions may be inaccurate "
            "and should be reviewed by authorized faculty. Risk classifications are "
            "provisional and should not replace human judgment.",
            styles["Italic"],
        ),
    ])

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>👨‍🏫 Faculty Dashboard</h1>
        <p>Trekverse Intelligent Academic Performance Monitoring and Early-Warning System</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR FILTERS
# ============================================================

with st.sidebar:
    st.markdown("## 👨‍🏫 Faculty Portal")
    st.markdown("---")
    st.markdown("### 🔎 Student Filters")

    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    departments = sorted(df["Department"].dropna().astype(str).unique().tolist())
    selected_department = st.selectbox("Department", ["All"] + departments)

    semesters = sorted(pd.to_numeric(df["Semester"], errors="coerce").dropna().unique().tolist())
    selected_semester = st.selectbox("Semester", ["All"] + semesters)

    risk_options = ["All", "High Risk", "Moderate Risk", "Low Risk"]
    selected_risk = st.selectbox("Risk Level", risk_options)

    st.markdown("---")
    st.info(
        "Use these filters to review student performance and identify "
        "students who may need academic support."
    )

# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[filtered_df["Department"].astype(str) == selected_department]

if selected_semester != "All":
    filtered_df = filtered_df[
        pd.to_numeric(filtered_df["Semester"], errors="coerce") == selected_semester
    ]

if selected_risk != "All":
    filtered_df = filtered_df[filtered_df["Risk_Level"] == selected_risk]

# ============================================================
# METRICS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📊 Academic Overview</h2>
        <p>Summary of the currently selected students</p>
    </div>
    """,
    unsafe_allow_html=True,
)

total_students = len(filtered_df)
average_marks = filtered_df["Final_Marks"].mean() if total_students else 0
high_risk_count = int((filtered_df["Risk_Level"] == "High Risk").sum())
moderate_risk_count = int((filtered_df["Risk_Level"] == "Moderate Risk").sum())
low_risk_count = int((filtered_df["Risk_Level"] == "Low Risk").sum())


def metric_card(icon, title, value, subtitle):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("👨‍🎓", "Total Students", total_students, "Students in selected group")
with c2:
    metric_card("📚", "Average Final Marks", f"{average_marks:.2f}", "Out of 100")
with c3:
    metric_card("🔴", "High Risk", high_risk_count, "Priority faculty review")
with c4:
    metric_card("🟡", "Moderate Risk", moderate_risk_count, "Additional monitoring")

# ============================================================
# RISK SUMMARY
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>🚦 Early-Warning Summary</h2>
        <p>Risk categories based on predicted final marks</p>
    </div>
    """,
    unsafe_allow_html=True,
)

r1, r2, r3 = st.columns(3)
with r1:
    st.metric("🔴 High Risk", high_risk_count)
with r2:
    st.metric("🟡 Moderate Risk", moderate_risk_count)
with r3:
    st.metric("🟢 Low Risk", low_risk_count)

if high_risk_count:
    st.warning(
        f"{high_risk_count} student(s) are classified as High Risk. "
        "Review their records and consider appropriate academic support."
    )
elif moderate_risk_count:
    st.info(
        "No High-Risk students were found in this selection. "
        "Continue monitoring students classified as Moderate Risk."
    )
else:
    st.success(
        "No High-Risk or Moderate-Risk students were found in this selection."
    )

# ============================================================
# RISK DISTRIBUTION CHART
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📈 Student Risk Distribution</h2>
        <p>Number of students in each risk category</p>
    </div>
    """,
    unsafe_allow_html=True,
)

risk_counts = filtered_df["Risk_Level"].value_counts().reindex(
    ["High Risk", "Moderate Risk", "Low Risk"], fill_value=0
)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(risk_counts.index, risk_counts.values)
ax.set_ylabel("Number of Students")
ax.set_xlabel("Risk Category")
ax.set_title("Early-Warning Risk Distribution")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ============================================================
# MARKS DISTRIBUTION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📊 Final Marks Distribution</h2>
        <p>Distribution of recorded final marks</p>
    </div>
    """,
    unsafe_allow_html=True,
)

valid_marks = filtered_df["Final_Marks"].dropna()
if not valid_marks.empty:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(valid_marks, bins=10)
    ax.set_xlabel("Actual Final Marks")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Actual Final Marks")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Actual final marks are unavailable for this selection.")

# ============================================================
# ATTENDANCE ANALYSIS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>🏫 Attendance Analysis</h2>
        <p>Relationship between attendance and final marks</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "Attendance" in filtered_df.columns:
    attendance_values = pd.to_numeric(filtered_df["Attendance"], errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(attendance_values, filtered_df["Final_Marks"], alpha=0.7)
    ax.set_xlabel("Attendance")
    ax.set_ylabel("Actual Final Marks")
    ax.set_title("Attendance vs Final Marks")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Attendance information is unavailable.")

# ============================================================
# STUDENTS REQUIRING SUPPORT
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>⚠️ Students Requiring Academic Support</h2>
        <p>Students classified as High Risk or Moderate Risk</p>
    </div>
    """,
    unsafe_allow_html=True,
)

support_df = filtered_df[
    filtered_df["Risk_Level"].isin(["High Risk", "Moderate Risk"])
].copy()

if not support_df.empty:
    risk_order = {"High Risk": 0, "Moderate Risk": 1, "Low Risk": 2}
    support_df["_order"] = support_df["Risk_Level"].map(risk_order)
    support_df = support_df.sort_values(
        ["_order", "Predicted_Final_Marks"], ascending=[True, True], na_position="last"
    )
    support_columns = [
        "Student_ID", "Name", "Department", "Semester",
        "Predicted_Final_Marks", "Risk_Level", "Recommendation",
    ]
    available_support_columns = [c for c in support_columns if c in support_df.columns]
    st.dataframe(support_df[available_support_columns], use_container_width=True, hide_index=True)
else:
    st.success(
        "No students in this selection are currently classified as High Risk or Moderate Risk."
    )

# ============================================================
# STUDENT SEARCH
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>🔍 Search Student</h2>
        <p>Search by student ID or name</p>
    </div>
    """,
    unsafe_allow_html=True,
)

search_text = st.text_input("Enter Student ID or Name", placeholder="Example: STU111 or Student 111")
if search_text.strip():
    search_mask = filtered_df["Student_ID"].astype(str).str.contains(
        search_text, case=False, na=False
    )
    if "Name" in filtered_df.columns:
        search_mask = search_mask | filtered_df["Name"].astype(str).str.contains(
            search_text, case=False, na=False
        )
    search_results = filtered_df[search_mask]
    if not search_results.empty:
        st.dataframe(search_results, use_container_width=True, hide_index=True)
    else:
        st.info("No matching student was found in the selected filters.")

# ============================================================
# INDIVIDUAL STUDENT PROFILE + CSV/PDF DOWNLOAD
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>👤 Individual Student Profile</h2>
        <p>Select a student to view their complete academic details</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not filtered_df.empty:
    profile_options = filtered_df.copy()
    profile_options["Student_ID"] = profile_options["Student_ID"].astype(str).str.strip()

    if "Name" in profile_options.columns:
        profile_options["Display_Name"] = (
            profile_options["Student_ID"] + " - " +
            profile_options["Name"].fillna("Unknown").astype(str)
        )
    else:
        profile_options["Display_Name"] = profile_options["Student_ID"]

    selected_profile = st.selectbox(
        "Choose a student",
        profile_options["Display_Name"].tolist(),
        key="faculty_student_profile",
    )
    selected_student = profile_options[
        profile_options["Display_Name"] == selected_profile
    ].iloc[0]

    st.subheader(f"🎓 {selected_profile}")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("Student ID", selected_student["Student_ID"])
    with p2:
        st.metric("Department", str(selected_student.get("Department", "N/A")))
    with p3:
        st.metric("Semester", str(selected_student.get("Semester", "N/A")))

    st.markdown("#### 📊 Academic Performance")
    detail_columns = [
        "Attendance", "Internal_Marks", "Assignment_Marks", "Quiz_Marks",
        "Study_Hours", "Previous_CGPA", "Current_CGPA", "Backlogs",
        "Assignment_Submission_Rate", "Final_Marks",
        "Predicted_Final_Marks", "Risk_Level",
    ]
    available_details = [c for c in detail_columns if c in profile_options.columns]
    details_table = pd.DataFrame({
        "Academic Factor": available_details,
        "Value": [selected_student[c] for c in available_details],
    })
    st.dataframe(details_table, use_container_width=True, hide_index=True)

    subject_columns = [
        c for c in [
            "Mathematics_Marks", "Programming_Marks", "DBMS_Marks",
            "Computer_Networks_Marks", "Electronics_Marks",
        ] if c in profile_options.columns
    ]
    if subject_columns:
        st.markdown("#### 📚 Subject-wise Marks")
        subject_table = pd.DataFrame({
            "Subject": subject_columns,
            "Marks": [selected_student[c] for c in subject_columns],
        })
        st.dataframe(subject_table, use_container_width=True, hide_index=True)

    history_columns = [
        c for c in [
            "Semester_1_CGPA", "Semester_2_CGPA",
            "Semester_3_CGPA", "Semester_4_CGPA",
        ] if c in profile_options.columns
    ]
    if history_columns:
        st.markdown("#### 📈 Semester-wise CGPA")
        history_values = pd.to_numeric(selected_student[history_columns], errors="coerce")
        history_df = pd.DataFrame({
            "Semester": [c.replace("_CGPA", "").replace("_", " ") for c in history_columns],
            "CGPA": history_values.values,
        }).dropna()
        if not history_df.empty:
            st.line_chart(history_df.set_index("Semester"))

    if "Recommendation" in profile_options.columns:
        recommendation = selected_student["Recommendation"]
        if pd.notna(recommendation):
            st.markdown("#### 💡 Academic Recommendation")
            st.info(str(recommendation))

    csv_record = selected_student.drop(labels=["Display_Name"], errors="ignore")
    st.download_button(
        label="📥 Download Selected Student Record (CSV)",
        data=pd.DataFrame([csv_record]).to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_student['Student_ID']}_academic_record.csv",
        mime="text/csv",
        key="download_faculty_student_record",
    )

    st.markdown("#### 📄 Download Academic PDF Report")
    try:
        pdf_data = create_student_pdf(selected_student)
        st.download_button(
            label="📥 Download Student PDF Report",
            data=pdf_data,
            file_name=f"{selected_student['Student_ID']}_academic_report.pdf",
            mime="application/pdf",
            key="download_faculty_student_pdf",
        )
    except Exception as error:
        st.error("Unable to generate the PDF report.")
        st.code(str(error))
else:
    st.info("No students are available for the selected filters.")

# ============================================================
# TOP PERFORMING STUDENTS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>🌟 Top Performing Students</h2>
        <p>Students with the highest recorded final marks</p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_students = filtered_df.sort_values(
    "Final_Marks", ascending=False, na_position="last"
).head(10)
top_columns = [
    "Student_ID", "Name", "Department", "Semester",
    "Final_Marks", "Predicted_Final_Marks", "Risk_Level",
]
available_top_columns = [c for c in top_columns if c in top_students.columns]
st.dataframe(top_students[available_top_columns], use_container_width=True, hide_index=True)

# ============================================================
# ACADEMIC FACTORS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📚 Academic Factors</h2>
        <p>Average values of important student indicators</p>
    </div>
    """,
    unsafe_allow_html=True,
)

factor_columns = [
    "Attendance", "Internal_Marks", "Assignment_Marks", "Quiz_Marks",
    "Study_Hours", "Previous_CGPA", "Backlogs", "Assignment_Submission_Rate",
]
available_factors = [c for c in factor_columns if c in filtered_df.columns]

if available_factors and not filtered_df.empty:
    numeric_factors = filtered_df[available_factors].apply(pd.to_numeric, errors="coerce")
    factor_means = numeric_factors.mean(numeric_only=True).dropna().sort_values()
    if not factor_means.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(factor_means.index, factor_means.values)
        ax.set_xlabel("Average Value")
        ax.set_title("Average Academic Indicators")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("No academic factor data is available.")

# ============================================================
# FACULTY RECOMMENDATIONS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>💡 Faculty Recommendations</h2>
        <p>Suggested actions based on the current student selection</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if total_students == 0:
    st.info("Select different filters to view student recommendations.")
else:
    if high_risk_count > 0:
        st.warning(
            f"🔴 Review the {high_risk_count} High-Risk student(s) individually. "
            "Discuss their academic difficulties and prepare appropriate support plans."
        )
    if moderate_risk_count > 0:
        st.info(
            f"🟡 Monitor the {moderate_risk_count} Moderate-Risk student(s). "
            "Consider additional practice, assignment support, and regular progress reviews."
        )
    if "Attendance" in filtered_df.columns:
        average_attendance = pd.to_numeric(filtered_df["Attendance"], errors="coerce").mean()
        if pd.notna(average_attendance) and average_attendance < 75:
            st.warning(
                f"🏫 Average attendance is {average_attendance:.1f}. "
                "Review attendance records and investigate whether students need support."
            )
    if "Study_Hours" in filtered_df.columns:
        average_study_hours = pd.to_numeric(filtered_df["Study_Hours"], errors="coerce").mean()
        if pd.notna(average_study_hours) and average_study_hours < 2:
            st.info(
                "📖 The selected group's average study hours are below two hours. "
                "Consider discussing study routines and time management with students."
            )
    if high_risk_count == 0 and moderate_risk_count == 0:
        st.success("All students in this selection are classified as Low Risk. Continue routine academic monitoring.")

# ============================================================
# DOWNLOAD FILTERED REPORT
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📥 Download Student Report</h2>
        <p>Export the currently selected student records</p>
    </div>
    """,
    unsafe_allow_html=True,
)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Early-Warning Report (CSV)",
    data=csv_data,
    file_name="trekverse_early_warning_report.csv",
    mime="text/csv",
    use_container_width=True,
)

# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("🤖 About the Prediction System"):
    st.write(
        """
        Trekverse uses a trained machine-learning pipeline to estimate
        students' final marks using available academic indicators.
        The Early Warning System assigns provisional risk categories
        based on the predicted marks.

        Risk thresholds:
        - High Risk: predicted marks below 55
        - Moderate Risk: predicted marks from 55 to below 65
        - Low Risk: predicted marks of 65 or above

        These thresholds should be confirmed against institutional
        academic policies. Predictions support faculty review and do
        not replace human judgment.
        """
    )
    st.write("Academic performance model: models/academic_performance_model.pkl")
    st.write("Early-warning report: reports/early_warning_report.csv")

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        👨‍🏫 Trekverse Faculty Dashboard
        <br>
        Machine Learning • Explainable AI • Academic Analytics
        <br><br>
        Intelligent Academic Performance Monitoring System
    </div>
    """,
    unsafe_allow_html=True,
)
# ============================================================
# ACADEMIC PERFORMANCE TREND ANALYSIS
# ============================================================

st.markdown("---")
st.header("📈 Academic Performance Trend Analysis")

st.write(
    "Compare average CGPA across semesters for the currently "
    "selected students."
)

cgpa_columns = [
    "Semester_1_CGPA",
    "Semester_2_CGPA",
    "Semester_3_CGPA",
    "Semester_4_CGPA",
]

available_cgpa = [
    column for column in cgpa_columns
    if column in filtered_df.columns
]

if len(available_cgpa) >= 2:

    cgpa_data = filtered_df[available_cgpa].apply(
        pd.to_numeric,
        errors="coerce"
    )

    average_cgpa = cgpa_data.mean().dropna()

    if len(average_cgpa) >= 2:

        trend_df = pd.DataFrame({
            "Semester": [
                column.replace("_CGPA", "").replace("_", " ")
                for column in average_cgpa.index
            ],
            "Average CGPA": average_cgpa.values
        })

        st.line_chart(
            trend_df.set_index("Semester")
        )

        first_cgpa = average_cgpa.iloc[0]
        latest_cgpa = average_cgpa.iloc[-1]
        change = latest_cgpa - first_cgpa

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "First Available Semester CGPA",
            f"{first_cgpa:.2f}"
        )

        col2.metric(
            "Latest Available Semester CGPA",
            f"{latest_cgpa:.2f}"
        )

        col3.metric(
            "CGPA Change",
            f"{change:+.2f}"
        )

        if change < -0.5:
            st.warning(
                "The group's average CGPA has declined. "
                "Review individual student records to identify "
                "students who may need academic support."
            )

        elif change > 0.5:
            st.success(
                "The group's average CGPA has improved."
            )

        else:
            st.info(
                "The group's average CGPA has remained relatively stable."
            )

    else:
        st.info(
            "Not enough CGPA data is available to display a trend."
        )

else:
    st.info(
        "At least two semester CGPA columns are required "
        "for trend analysis."
    )