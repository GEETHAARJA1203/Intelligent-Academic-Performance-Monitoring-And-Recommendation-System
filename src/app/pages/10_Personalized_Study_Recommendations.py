"""
Trekverse - Personalized Study Recommendations
Rule-based recommendations from the selected student's academic indicators.
Run through the main Streamlit app; this page reads the current CSV dataset.
"""
import os
import sys
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

st.set_page_config(
    page_title="Personalized Study Plan | Trekverse",
    page_icon="📚",
    layout="wide",
)

# Use the project's access-control functions when available.
try:
    from access_control import require_role, logout_button
    require_role(["Student", "Faculty", "Admin"])
    try:
        logout_button()
    except Exception:
        pass
except ImportError as exc:
    st.error("Could not import the project's access-control module.")
    st.code(str(exc))
    st.stop()
except Exception as exc:
    st.error("Access control could not be initialized.")
    st.code(str(exc))
    st.stop()

st.title("📚 Personalized Study Recommendations")
st.write(
    "Review academic indicators and get a practical study plan. "
    "Recommendations use transparent rules based on the selected student's data."
)

DATA_PATH = os.path.join(BASE_DIR, "data", "student_academic_data.csv")
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as exc:
    st.error("Unable to read the academic dataset.")
    st.code(str(exc))
    st.stop()

required = [
    "Student_ID", "Name", "Attendance", "Internal_Marks",
    "Assignment_Marks", "Quiz_Marks", "Study_Hours",
    "Assignment_Submission_Rate", "Backlogs", "Risk_Level",
]
missing = [col for col in required if col not in df.columns]
if missing:
    st.error("The dataset is missing required columns:")
    st.write(missing)
    st.stop()

ids = df["Student_ID"].dropna().astype(str).unique().tolist()
if not ids:
    st.warning("No student records were found.")
    st.stop()

# Demo selector only. In production, bind this to the authenticated user's Student_ID.
student_id = st.selectbox("Select Student ID (demo/testing)", ids)
matches = df[df["Student_ID"].astype(str) == student_id]
if matches.empty:
    st.error("Student record not found.")
    st.stop()
student = matches.iloc[0]

def numeric_value(column, default=0.0):
    value = pd.to_numeric(pd.Series([student[column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float(default)

name = str(student["Name"]) if pd.notna(student["Name"]) else "Student"
attendance = numeric_value("Attendance")
internal = numeric_value("Internal_Marks")
assignment = numeric_value("Assignment_Marks")
quiz = numeric_value("Quiz_Marks")
study_hours = numeric_value("Study_Hours")
submission_rate = numeric_value("Assignment_Submission_Rate")
backlogs = numeric_value("Backlogs")
risk = str(student["Risk_Level"]) if pd.notna(student["Risk_Level"]) else "Unknown"

st.subheader(f"Student overview: {name}")
a, b, c, d = st.columns(4)
a.metric("Attendance", f"{attendance:.1f}%")
b.metric("Study hours/day", f"{study_hours:.1f}")
c.metric("Assignment submission", f"{submission_rate:.1f}%")
d.metric("Risk level", risk)

recommendations = []

# Thresholds are illustrative configurable rules, not official college policy.
if attendance < 75:
    recommendations.append({
        "priority": "High",
        "area": "Attendance",
        "finding": f"Attendance is {attendance:.1f}%.",
        "action": "Attend scheduled classes regularly and plan how to make up missed topics. Confirm the official attendance requirement with faculty.",
        "weekly_goal": "Attend every scheduled class this week."
    })

if internal < 60:
    recommendations.append({
        "priority": "High",
        "area": "Internal marks",
        "finding": f"Internal marks are {internal:.1f}.",
        "action": "Review low-scoring topics, solve practice questions, and ask faculty to clarify mistakes.",
        "weekly_goal": "Complete three focused revision sessions and one practice test."
    })

if assignment < 60:
    recommendations.append({
        "priority": "Medium",
        "area": "Assignments",
        "finding": f"Assignment marks are {assignment:.1f}.",
        "action": "Break each assignment into smaller tasks, check the rubric, and review feedback before submission.",
        "weekly_goal": "Finish the next assignment before its deadline and check it against the rubric."
    })

if submission_rate < 80:
    recommendations.append({
        "priority": "Medium",
        "area": "Submission habits",
        "finding": f"Assignment submission rate is {submission_rate:.1f}%.",
        "action": "Record deadlines in a calendar and start assignments early enough to review them.",
        "weekly_goal": "Submit all assignments due this week on time."
    })

if quiz < 60:
    recommendations.append({
        "priority": "Medium",
        "area": "Quiz preparation",
        "finding": f"Quiz marks are {quiz:.1f}.",
        "action": "Use short retrieval-practice sessions, flashcards, and timed topic quizzes; review incorrect answers.",
        "weekly_goal": "Complete two short practice quizzes and review every incorrect answer."
    })

if study_hours < 2:
    recommendations.append({
        "priority": "Medium",
        "area": "Study routine",
        "finding": f"Reported study time is {study_hours:.1f} hours/day.",
        "action": "Build a manageable routine with focused study blocks and short breaks. Adjust it to your timetable and wellbeing.",
        "weekly_goal": "Try one focused 25–40 minute study block on most days."
    })

if backlogs > 0:
    recommendations.append({
        "priority": "High",
        "area": "Backlog planning",
        "finding": f"Recorded backlogs: {backlogs:.0f}.",
        "action": "List backlog subjects, prioritize upcoming exam topics, and discuss a realistic revision schedule with faculty.",
        "weekly_goal": "Create a topic checklist for one backlog subject and complete the first two topics."
    })

if not recommendations:
    recommendations.append({
        "priority": "Maintain",
        "area": "Consistent progress",
        "finding": "The selected indicators do not trigger the configured improvement rules.",
        "action": "Maintain regular attendance, submit work on time, review difficult topics weekly, and monitor results.",
        "weekly_goal": "Complete a weekly review and prepare a plan for the next week."
    })

st.markdown("---")
st.subheader("Recommended actions")
for rec in recommendations:
    with st.container(border=True):
        st.markdown(f"### {rec['area']}  ·  {rec['priority']} priority")
        st.write(f"**Observation:** {rec['finding']}")
        st.write(f"**Suggested action:** {rec['action']}")
        st.write(f"**This week's goal:** {rec['weekly_goal']}")

st.markdown("---")
st.subheader("Simple weekly study plan")
plan = pd.DataFrame([
    {"Day": "Monday", "Activity": "Review the weakest topic and write down questions."},
    {"Day": "Tuesday", "Activity": "Practice problems for 30–45 minutes."},
    {"Day": "Wednesday", "Activity": "Work on assignments and check deadlines."},
    {"Day": "Thursday", "Activity": "Take a short quiz and review mistakes."},
    {"Day": "Friday", "Activity": "Revise class notes and ask for help on difficult concepts."},
    {"Day": "Saturday", "Activity": "Complete a longer practice session or backlog topic."},
    {"Day": "Sunday", "Activity": "Review progress and plan next week's priorities."},
])
st.dataframe(plan, use_container_width=True, hide_index=True)

csv_bytes = pd.DataFrame(recommendations).to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download recommendations as CSV",
    data=csv_bytes,
    file_name=f"study_recommendations_{student_id}.csv",
    mime="text/csv",
)

st.caption(
    "These are transparent rule-based suggestions, not medical, disciplinary, "
    "or guaranteed academic outcomes. Thresholds are illustrative and should be "
    "reviewed by the institution. The student selector is for demo/testing; "
    "production access must restrict students to their own records."
)
