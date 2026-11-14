import os
import sqlite3
from datetime import datetime
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATABASE_PATH = os.path.join(BASE_DIR, "student_database.db")
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "data", "student_academic_data.csv")

ACADEMIC_COLUMNS = {
    "semester": "INTEGER",
    "attendance": "REAL",
    "internal_marks": "REAL",
    "assignment_marks": "REAL",
    "quiz_marks": "REAL",
    "study_hours": "REAL",
    "previous_cgpa": "REAL",
    "current_cgpa": "REAL",
    "backlogs": "INTEGER",
    "assignment_submission_rate": "REAL",
    "learning_activity": "TEXT",
    "final_marks": "REAL",
    "mathematics_marks": "REAL",
    "programming_marks": "REAL",
    "dbms_marks": "REAL",
    "computer_networks_marks": "REAL",
    "electronics_marks": "REAL",
    "semester_1_cgpa": "REAL",
    "semester_2_cgpa": "REAL",
    "semester_3_cgpa": "REAL",
    "semester_4_cgpa": "REAL",
    "risk_level": "TEXT",
}

def get_connection():
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection

def create_tables():
    """Create the existing table and add academic columns without deleting records."""
    connection = get_connection()
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                student_name TEXT NOT NULL,
                department TEXT,
                study_time INTEGER,
                absences INTEGER,
                failures INTEGER,
                goout INTEGER,
                freetime INTEGER,
                health INTEGER,
                famrel INTEGER,
                traveltime INTEGER,
                predicted_score REAL,
                performance_level TEXT,
                early_warning TEXT,
                created_at TEXT
            )
        """)
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(students)")}
        for name, kind in ACADEMIC_COLUMNS.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE students ADD COLUMN {name} {kind}")
        connection.commit()
    finally:
        connection.close()

def _clean(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value

def add_student(student_id, student_name, department, study_time=None, absences=None,
                failures=None, goout=None, freetime=None, health=None, famrel=None,
                traveltime=None, predicted_score=None, performance_level=None,
                early_warning=None, **academic_fields):
    """Add a record while remaining compatible with the existing Database page."""
    record = {
        "student_id": str(student_id).strip(),
        "student_name": str(student_name).strip(),
        "department": department,
        "study_time": study_time,
        "absences": absences,
        "failures": failures,
        "goout": goout,
        "freetime": freetime,
        "health": health,
        "famrel": famrel,
        "traveltime": traveltime,
        "predicted_score": predicted_score,
        "performance_level": performance_level,
        "early_warning": early_warning,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for key, value in academic_fields.items():
        if key in ACADEMIC_COLUMNS:
            record[key] = _clean(value)

    columns = list(record)
    placeholders = ", ".join("?" for _ in columns)
    connection = get_connection()
    try:
        connection.execute(
            f"INSERT INTO students ({', '.join(columns)}) VALUES ({placeholders})",
            [_clean(record[column]) for column in columns],
        )
        connection.commit()
        return True, "Student saved successfully."
    except sqlite3.IntegrityError:
        return False, "Student ID already exists."
    except Exception as exc:
        return False, str(exc)
    finally:
        connection.close()
def import_students_from_dataframe(df):
    """
    Import validated and mapped student records into SQLite.
    Existing Student IDs are updated; new IDs are inserted.
    """
    if df is None or df.empty:
        raise ValueError("No student records available for import.")

    df = df.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]

    required_columns = ["student_id", "name"]

    # Support either 'name' or the database field 'student_name'.
    if "student_name" not in df.columns and "name" in df.columns:
        df = df.rename(columns={"name": "student_name"})

    if "student_id" not in df.columns or "student_name" not in df.columns:
        raise ValueError("Student ID and Student Name columns are required.")

    fields = ["student_name", "department"] + list(ACADEMIC_COLUMNS)
    fields = [field for field in fields if field in df.columns]

    inserted = 0
    updated = 0
    skipped = 0

    connection = get_connection()

    try:
        for _, row in df.iterrows():
            sid = _clean(row.get("student_id"))
            name = _clean(row.get("student_name"))

            if (
                sid is None or name is None
                or not str(sid).strip()
                or not str(name).strip()
            ):
                skipped += 1
                continue

            sid = str(sid).strip()
            name = str(name).strip()

            values = {
                field: _clean(row.get(field))
                for field in fields
            }
            values["student_name"] = name

            existing = connection.execute(
                "SELECT id FROM students WHERE student_id = ?",
                (sid,)
            ).fetchone()

            if existing:
                assignments = ", ".join(
                    f"{field} = ?" for field in fields
                )

                connection.execute(
                    f"UPDATE students SET {assignments} "
                    "WHERE student_id = ?",
                    [values[field] for field in fields] + [sid]
                )
                updated += 1

            else:
                values["student_id"] = sid
                values["created_at"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                columns = list(values)

                connection.execute(
                    f"INSERT INTO students ({', '.join(columns)}) "
                    f"VALUES ({', '.join('?' for _ in columns)})",
                    [values[column] for column in columns]
                )
                inserted += 1

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return {
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "total_in_csv": len(df)
    }        

def import_students_from_csv(csv_path=DEFAULT_CSV_PATH):
    """Import the academic CSV; update matching IDs and insert new IDs."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    aliases = {
        "Student_ID": "student_id", "Name": "student_name",
        "Department": "department", "Semester": "semester",
        "Attendance": "attendance", "Internal_Marks": "internal_marks",
        "Assignment_Marks": "assignment_marks", "Quiz_Marks": "quiz_marks",
        "Study_Hours": "study_hours", "Previous_CGPA": "previous_cgpa",
        "Current_CGPA": "current_cgpa", "Backlogs": "backlogs",
        "Assignment_Submission_Rate": "assignment_submission_rate",
        "Learning_Activity": "learning_activity", "Final_Marks": "final_marks",
        "Mathematics_Marks": "mathematics_marks",
        "Programming_Marks": "programming_marks", "DBMS_Marks": "dbms_marks",
        "Computer_Networks_Marks": "computer_networks_marks",
        "Electronics_Marks": "electronics_marks",
        "Semester_1_CGPA": "semester_1_cgpa", "Semester_2_CGPA": "semester_2_cgpa",
        "Semester_3_CGPA": "semester_3_cgpa", "Semester_4_CGPA": "semester_4_cgpa",
        "Risk_Level": "risk_level",
    }
    df = df.rename(columns=aliases)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "student_id" not in df or "student_name" not in df:
        raise ValueError("CSV must contain Student_ID and Name columns.")

    fields = ["student_name", "department"] + list(ACADEMIC_COLUMNS)
    fields = [field for field in fields if field in df.columns]
    inserted = updated = skipped = 0
    connection = get_connection()
    try:
        for _, row in df.iterrows():
            sid, name = _clean(row.get("student_id")), _clean(row.get("student_name"))
            if sid is None or name is None or not str(sid).strip() or not str(name).strip():
                skipped += 1
                continue
            sid, name = str(sid).strip(), str(name).strip()
            existing = connection.execute(
                "SELECT id FROM students WHERE student_id = ?", (sid,)
            ).fetchone()
            values = {field: _clean(row.get(field)) for field in fields}
            values["student_name"] = name

            if existing:
                assignments = ", ".join(f"{field} = ?" for field in fields)
                if assignments:
                    connection.execute(
                        f"UPDATE students SET {assignments} WHERE student_id = ?",
                        [values[field] for field in fields] + [sid],
                    )
                updated += 1
            else:
                values.update({
                    "student_id": sid,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                columns = list(values)
                connection.execute(
                    f"INSERT INTO students ({', '.join(columns)}) VALUES "
                    f"({', '.join('?' for _ in columns)})",
                    [values[column] for column in columns],
                )
                inserted += 1
        connection.commit()
    finally:
        connection.close()
    return {"inserted": inserted, "updated": updated, "skipped": skipped, "total_in_csv": len(df)}

def get_students():
    connection = get_connection()
    try:
        return connection.execute(
            "SELECT * FROM students ORDER BY created_at DESC, student_id"
        ).fetchall()
    finally:
        connection.close()

def get_student(student_id):
    connection = get_connection()
    try:
        return connection.execute(
            "SELECT * FROM students WHERE student_id = ?", (str(student_id).strip(),)
        ).fetchone()
    finally:
        connection.close()

def delete_student(student_id):
    connection = get_connection()
    try:
        cursor = connection.execute(
            "DELETE FROM students WHERE student_id = ?", (str(student_id).strip(),)
        )
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()

def get_statistics():
    connection = get_connection()
    try:
        return connection.execute("""
            SELECT COUNT(*) AS total_students,
                   AVG(COALESCE(final_marks, predicted_score)) AS average_score,
                   AVG(absences) AS average_absences
            FROM students
        """).fetchone()
    finally:
        connection.close()

create_tables()
