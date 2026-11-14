import os
import sys
from io import BytesIO
from datetime import datetime

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether
)

# Make the src folder importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

st.set_page_config(page_title="Automatic PDF Student Report", page_icon="📄", layout="wide")
try:
    from access_control import require_login
    require_login()
except ImportError as e:
    st.error(f"Could not import the login function: {e}")
    st.stop()

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "student_academic_data.csv")


@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def value_or_na(value):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def build_pdf(student, recommendations, plan):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Trekverse Student Academic Report",
        author="Trekverse",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=19,
        leading=24,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="SubTitleCenter",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.HexColor("#555555"),
        spaceAfter=14,
    ))
    story = [
        Paragraph("TREKVERSE", styles["ReportTitle"]),
        Paragraph("Student Academic Performance Report", styles["SubTitleCenter"]),
        Paragraph("1. Student Details", styles["Heading2"]),
    ]

    details = [
        ["Student ID", value_or_na(student.get("Student_ID", "N/A")),
         "Student Name", value_or_na(student.get("Name", "N/A"))],
        ["Department", value_or_na(student.get("Department", "N/A")),
         "Semester", value_or_na(student.get("Semester", "N/A"))],
        ["Report Generated", datetime.now().strftime("%d %b %Y, %I:%M %p"),
         "Risk Level", value_or_na(student.get("Risk_Level", "Not available"))],
    ]
    detail_table = Table(details, colWidths=[3.0*cm, 5.0*cm, 3.0*cm, 5.0*cm])
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F8")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EAF0F8")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B8C4D6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [detail_table, Spacer(1, 14), Paragraph("2. Academic Performance", styles["Heading2"])]

    academic_fields = [
        ("Attendance (%)", "Attendance"),
        ("Internal Marks", "Internal_Marks"),
        ("Assignment Marks", "Assignment_Marks"),
        ("Quiz Marks", "Quiz_Marks"),
        ("Study Hours", "Study_Hours"),
        ("Assignment Submission Rate (%)", "Assignment_Submission_Rate"),
        ("Previous CGPA", "Previous_CGPA"),
        ("Current CGPA", "Current_CGPA"),
        ("Backlogs", "Backlogs"),
        ("Final Marks", "Final_Marks"),
    ]
    rows = [["Measure", "Value"]]
    for label, key in academic_fields:
        if key in student.index:
            rows.append([label, value_or_na(student[key])])
    academic_table = Table(rows, colWidths=[9.5*cm, 6.5*cm], repeatRows=1)
    academic_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C4D6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [academic_table, Spacer(1, 14), Paragraph("3. Personalized Recommendations", styles["Heading2"])]

    if recommendations:
        for item in recommendations:
            story.append(Paragraph("• " + item, styles["BodyText"]))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No recommendations were generated.", styles["BodyText"]))

    story += [Spacer(1, 8), Paragraph("4. Suggested Weekly Improvement Plan", styles["Heading2"])]
    for day, task in plan:
        story.append(Paragraph(f"<b>{day}:</b> {task}", styles["BodyText"]))
        story.append(Spacer(1, 4))

    story += [
        Spacer(1, 16),
        Paragraph(
            "Note: This report is intended to support academic guidance. "
            "Risk labels and recommendations should be reviewed by faculty and are not a substitute "
            "for professional academic judgement.",
            styles["Italic"],
        ),
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


st.title("📄 Automatic PDF Student Report")
st.caption("Generate a downloadable academic report using the Trekverse student dataset.")

df = load_data(DATA_PATH)
if df.empty:
    st.error(f"Could not load the dataset at: {DATA_PATH}")
    st.info("Confirm that data/student_academic_data.csv exists in your project folder.")
    st.stop()

if "Student_ID" not in df.columns:
    st.error("The dataset must contain a Student_ID column.")
    st.stop()

student_ids = df["Student_ID"].dropna().astype(str).unique().tolist()
if not student_ids:
    st.error("No student IDs were found in the dataset.")
    st.stop()

selected_id = st.selectbox("Select a student", student_ids)
selected_rows = df[df["Student_ID"].astype(str) == selected_id]
if selected_rows.empty:
    st.error("Student record not found.")
    st.stop()
student = selected_rows.iloc[0]

attendance = pd.to_numeric(student.get("Attendance", 0), errors="coerce")
internal = pd.to_numeric(student.get("Internal_Marks", 0), errors="coerce")
assignment = pd.to_numeric(student.get("Assignment_Marks", 0), errors="coerce")
quiz = pd.to_numeric(student.get("Quiz_Marks", 0), errors="coerce")
study_hours = pd.to_numeric(student.get("Study_Hours", 0), errors="coerce")
submission = pd.to_numeric(student.get("Assignment_Submission_Rate", 0), errors="coerce")
backlogs = pd.to_numeric(student.get("Backlogs", 0), errors="coerce")

recommendations = []
if pd.notna(attendance) and attendance < 75:
    recommendations.append("Improve attendance and aim to attend at least 75% of classes, following institutional rules.")
if pd.notna(internal) and internal < 60:
    recommendations.append("Revise class notes regularly and practise questions related to internal assessments.")
if pd.notna(assignment) and assignment < 60:
    recommendations.append("Break assignments into smaller tasks and submit them before the deadline.")
if pd.notna(quiz) and quiz < 60:
    recommendations.append("Use short daily quizzes and flashcards to strengthen topic recall.")
if pd.notna(study_hours) and study_hours < 2:
    recommendations.append("Build a consistent study routine, starting with focused 25–30 minute sessions.")
if pd.notna(submission) and submission < 80:
    recommendations.append("Track assignment deadlines in a calendar and submit work on time.")
if pd.notna(backlogs) and backlogs > 0:
    recommendations.append("Create a backlog-clearing timetable and ask faculty for help with difficult topics.")
if not recommendations:
    recommendations.append("Maintain your current habits, review progress weekly, and keep practising important topics.")

plan = [
    ("Monday", "Review lecture notes and list the topics that need improvement."),
    ("Tuesday", "Practise problems or questions from one difficult subject."),
    ("Wednesday", "Complete pending assignments and check submission deadlines."),
    ("Thursday", "Take a short self-test and review incorrect answers."),
    ("Friday", "Ask a faculty member or peer for help with unresolved doubts."),
    ("Saturday", "Revise the week's topics and update your progress checklist."),
    ("Sunday", "Plan next week's study sessions and take adequate rest."),
]

c1, c2, c3 = st.columns(3)
c1.metric("Student ID", str(student.get("Student_ID", "N/A")))
c2.metric("Risk Level", str(student.get("Risk_Level", "Not available")))
c3.metric("Final Marks", value_or_na(student.get("Final_Marks", "N/A")))

st.subheader("Report preview")
st.write(f"**Student:** {value_or_na(student.get('Name', 'N/A'))}")
st.write(f"**Department / Semester:** {value_or_na(student.get('Department', 'N/A'))} / {value_or_na(student.get('Semester', 'N/A'))}")
st.write("**Recommendations:**")
for item in recommendations:
    st.write(f"- {item}")

pdf_bytes = build_pdf(student, recommendations, plan)
safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in selected_id)
st.download_button(
    label="⬇️ Download Student Report (PDF)",
    data=pdf_bytes,
    file_name=f"Trekverse_Student_Report_{safe_id}.pdf",
    mime="application/pdf",
    use_container_width=True,
)

st.caption("Demo note: the student selector is for project testing. For production, bind Student access to the authenticated user's own Student_ID.")
