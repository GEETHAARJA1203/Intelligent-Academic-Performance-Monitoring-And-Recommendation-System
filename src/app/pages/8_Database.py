import os
import sys
import pandas as pd
import streamlit as st

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from access_control import require_role, logout_button
from database import (
    add_student, get_students, get_student, delete_student,
    get_statistics, import_students_from_csv, DEFAULT_CSV_PATH
)

st.set_page_config(page_title="Student Database", page_icon="🗄️", layout="wide")
user = require_role(["Admin"])
logout_button()

st.markdown("""
<style>
.block-container {padding-top:1.5rem;padding-bottom:2rem}
.stApp {background:linear-gradient(135deg,#f8fbff 0%,#eef5ff 50%,#f8fbff 100%)}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#172554,#1e3a8a)}
section[data-testid="stSidebar"] * {color:white !important}
.hero {background:linear-gradient(135deg,#172554,#1d4ed8,#2563eb);padding:30px;border-radius:22px;margin-bottom:24px}
.hero h1 {color:white;font-size:36px;margin:0}
.hero p {color:#dbeafe;font-size:16px;margin-bottom:0}
.section {background:white;padding:18px 22px;border-radius:15px;margin:20px 0 16px;border-left:6px solid #2563eb;box-shadow:0 5px 20px #0f172a12}
.section h2 {margin:0;color:#172554;font-size:24px}
.section p {margin:5px 0 0;color:#64748b;font-size:14px}
.footer {text-align:center;padding:28px;color:#64748b;font-size:13px;margin-top:30px}
</style>
<div class="hero"><h1>🗄️ Student Academic Database</h1>
<p>Manage Trekverse academic records, CGPA, marks, and early-warning risk levels.</p></div>
""", unsafe_allow_html=True)

def section(title, subtitle):
    st.markdown(f'<div class="section"><h2>{title}</h2><p>{subtitle}</p></div>', unsafe_allow_html=True)

def all_records():
    rows = get_students()
    return pd.DataFrame([dict(row) for row in rows]) if rows else pd.DataFrame()

def visible_columns(df):
    preferred = [
        "student_id", "student_name", "department", "semester", "attendance",
        "internal_marks", "assignment_marks", "quiz_marks", "study_hours",
        "previous_cgpa", "current_cgpa", "backlogs", "assignment_submission_rate",
        "learning_activity", "final_marks", "risk_level", "created_at"
    ]
    return [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred and c not in ("id",)
    ]

with st.sidebar:
    st.markdown("## 🗄️ Student Database")
    menu = st.radio("Select Operation", [
        "📊 Database Overview", "🔄 Import Academic CSV",
        "➕ Add Student", "🔍 Search Student", "🗑️ Delete Student"
    ])
    st.caption("Records are stored locally in SQLite.")

if menu == "📊 Database Overview":
    section("📊 Database Overview", "Summary of academic records stored in SQLite.")
    stats = get_statistics()
    c1, c2, c3 = st.columns(3)
    c1.metric("👨‍🎓 Total Students", stats["total_students"] or 0)
    c2.metric("🎯 Average Final Marks",
              f'{stats["average_score"]:.2f}' if stats["average_score"] is not None else "N/A")
    c3.metric("📅 Average Absences",
              f'{stats["average_absences"]:.1f}' if stats["average_absences"] is not None else "N/A")
    df = all_records()
    section("👨‍🎓 Student Academic Records", "Academic data currently stored in the database.")
    if df.empty:
        st.info("No records found. Use Import Academic CSV to load the dataset.")
    else:
        if "risk_level" in df.columns:
            st.write("**Risk-level distribution**")
            st.bar_chart(df["risk_level"].fillna("Unknown").value_counts())
        cols = visible_columns(df)
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Database Records (CSV)",
            df[cols].to_csv(index=False).encode("utf-8"),
            "trekverse_student_database.csv", "text/csv", use_container_width=True)

elif menu == "🔄 Import Academic CSV":
    section("🔄 Import Academic Dataset", "Import or refresh student_academic_data.csv.")
    st.write("Default dataset path:")
    st.code(DEFAULT_CSV_PATH)
    st.warning("Student_ID is used to match records. Matching IDs are updated; new IDs are inserted. Other records are not deleted.")
    uploaded = st.file_uploader("Optional: upload another CSV", type=["csv"])
    confirmed = st.checkbox("I understand that matching Student IDs will be updated.")
    if st.button("📥 Import / Refresh Student Records", type="primary", use_container_width=True):
        if not confirmed:
            st.warning("Please confirm the import first.")
        else:
            temp_path = None
            try:
                if uploaded is not None:
                    os.makedirs(os.path.dirname(DEFAULT_CSV_PATH), exist_ok=True)
                    temp_path = os.path.join(os.path.dirname(DEFAULT_CSV_PATH), "_uploaded_academic_import.csv")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                    result = import_students_from_csv(temp_path)
                else:
                    result = import_students_from_csv(DEFAULT_CSV_PATH)
                st.success("Import completed.")
                a, b, c, d = st.columns(4)
                a.metric("Inserted", result["inserted"])
                b.metric("Updated", result["updated"])
                c.metric("Skipped", result["skipped"])
                d.metric("Rows in CSV", result["total_in_csv"])
            except Exception as exc:
                st.error(f"Import failed: {exc}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

elif menu == "➕ Add Student":
    section("➕ Add Student", "Add a student using Trekverse academic fields.")
    with st.form("academic_student_form"):
        a, b, c = st.columns(3)
        with a:
            sid = st.text_input("Student ID", placeholder="Example: STU121")
            name = st.text_input("Student Name")
            dept = st.text_input("Department", value="Information Technology")
            semester = st.number_input("Semester", 1, 12, 1)
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, 75.0)
        with b:
            internal = st.number_input("Internal Marks", 0.0, 100.0, 0.0)
            assignment = st.number_input("Assignment Marks", 0.0, 100.0, 0.0)
            quiz = st.number_input("Quiz Marks", 0.0, 100.0, 0.0)
            study = st.number_input("Study Hours", 0.0, 24.0, 2.0)
            submission = st.number_input("Assignment Submission Rate (%)", 0.0, 100.0, 80.0)
        with c:
            previous = st.number_input("Previous CGPA", 0.0, 10.0, 0.0, step=0.1)
            current = st.number_input("Current CGPA", 0.0, 10.0, 0.0, step=0.1)
            backlogs = st.number_input("Backlogs", 0, 30, 0)
            activity = st.selectbox("Learning Activity", ["Low", "Moderate", "High"])
            final_marks = st.number_input("Final Marks", 0.0, 100.0, 0.0)
        risk = st.selectbox("Risk Level", ["Low", "Moderate", "High"])
        submitted = st.form_submit_button("💾 Save Student", type="primary", use_container_width=True)
    if submitted:
        if not sid.strip() or not name.strip():
            st.error("Please enter both Student ID and Student Name.")
        else:
            success, message = add_student(
                student_id=sid.strip(), student_name=name.strip(), department=dept.strip(),
                semester=int(semester), attendance=float(attendance),
                internal_marks=float(internal), assignment_marks=float(assignment),
                quiz_marks=float(quiz), study_hours=float(study),
                previous_cgpa=float(previous), current_cgpa=float(current),
                backlogs=int(backlogs), assignment_submission_rate=float(submission),
                learning_activity=activity, final_marks=float(final_marks), risk_level=risk,
                predicted_score=float(final_marks),
                performance_level=("Excellent" if final_marks >= 85 else "Good" if final_marks >= 70 else "Average" if final_marks >= 50 else "Needs Attention"),
                early_warning=("Attention Required" if risk == "High" else "Monitor" if risk == "Moderate" else "On Track")
            )
            st.success("Student record saved successfully.") if success else st.error(message)

elif menu == "🔍 Search Student":
    section("🔍 Search Student", "Find a student by Student ID.")
    sid = st.text_input("Enter Student ID", placeholder="Example: STU001")
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not sid.strip():
            st.warning("Please enter a Student ID.")
        else:
            row = get_student(sid.strip())
            if row is None:
                st.error("Student ID not found.")
            else:
                data = dict(row)
                st.success("Student found.")
                fields = [
                    ("Student ID","student_id"),("Student Name","student_name"),
                    ("Department","department"),("Semester","semester"),
                    ("Attendance (%)","attendance"),("Internal Marks","internal_marks"),
                    ("Assignment Marks","assignment_marks"),("Quiz Marks","quiz_marks"),
                    ("Study Hours","study_hours"),("Previous CGPA","previous_cgpa"),
                    ("Current CGPA","current_cgpa"),("Backlogs","backlogs"),
                    ("Assignment Submission Rate","assignment_submission_rate"),
                    ("Learning Activity","learning_activity"),("Final Marks","final_marks"),
                    ("Risk Level","risk_level"),("Early Warning","early_warning"),("Created At","created_at")
                ]
                left, right = st.columns(2)
                for i, (label, key) in enumerate(fields):
                    value = data.get(key)
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        value = "Not available"
                    (left if i % 2 == 0 else right).write(f"**{label}:** {value}")
                st.download_button("⬇️ Download This Student Record (CSV)",
                    pd.DataFrame([data]).to_csv(index=False).encode("utf-8"),
                    f'{data.get("student_id", "student")}_record.csv', "text/csv",
                    use_container_width=True)

elif menu == "🗑️ Delete Student":
    section("🗑️ Delete Student", "Remove one student record from SQLite.")
    sid = st.text_input("Student ID to delete", placeholder="Example: STU121")
    confirmed = st.checkbox("I understand this permanently deletes this database record.")
    if st.button("🗑️ Delete Student Record", type="primary", use_container_width=True):
        if not sid.strip():
            st.warning("Please enter a Student ID.")
        elif not confirmed:
            st.warning("Please confirm before deleting.")
        elif delete_student(sid.strip()):
            st.success("Student record deleted.")
        else:
            st.error("Student ID not found.")

st.markdown('<div class="footer">🗄️ Trekverse Student Academic Database<br>SQLite • Python • Streamlit</div>', unsafe_allow_html=True)
