"""
Trekverse - What-If Simulator
Uses student_academic_data.csv and models/what_if_model.pkl
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PATH CONFIGURATION
# ============================================================

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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="What-If Simulator | Trekverse",
    page_icon="🎯",
    layout="wide",
)


# ============================================================
# ACCESS CONTROL
# ============================================================

try:
    from access_control import require_role, logout_button

    require_role(["Student", "Faculty", "Admin"])

except ImportError as exc:
    st.error("Could not import the access-control module.")
    st.code(str(exc))
    st.stop()

except Exception as exc:
    st.error("Access control could not be initialized.")
    st.code(str(exc))
    st.stop()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🎯 What-If Simulator")

st.write(
    """
    Explore how hypothetical changes in academic activities
    affect the marks estimated by the machine-learning model.
    """
)

try:
    logout_button()
except Exception:
    pass


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "student_academic_data.csv",
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "what_if_model.pkl",
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "Attendance",
    "Internal_Marks",
    "Assignment_Marks",
    "Quiz_Marks",
    "Study_Hours",
    "Previous_CGPA",
    "Backlogs",
    "Assignment_Submission_Rate",
    "Learning_Activity",
    "Department",
    "Semester",
]

TARGET = "Final_Marks"


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(DATA_PATH):
    st.error("Student academic dataset was not found.")
    st.code(DATA_PATH)
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.warning("The What-If model has not been trained yet.")

    st.info(
        "Open the VS Code terminal and run:"
    )

    st.code("python src\\train_what_if_model.py")

    st.stop()


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

try:
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

except Exception as exc:
    st.error("Unable to load the dataset or model.")
    st.code(str(exc))
    st.stop()


# ============================================================
# VALIDATE DATASET
# ============================================================

required_columns = FEATURES + [
    TARGET,
    "Student_ID",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error("Required columns are missing from the dataset.")

    st.write("Missing columns:")

    st.write(missing_columns)

    st.stop()


if df.empty:
    st.warning("No student records are available.")
    st.stop()


# ============================================================
# STUDENT SELECTION
# ============================================================

# This selector is for development and demonstration.
# In production, students must only access their own records.

student_ids = (
    df["Student_ID"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if not student_ids:
    st.error("No student IDs are available.")
    st.stop()


selected_id = st.selectbox(
    "Select Student ID",
    options=student_ids,
)

selected_rows = df[
    df["Student_ID"].astype(str) == selected_id
]

if selected_rows.empty:
    st.error("Student record not found.")
    st.stop()


student = selected_rows.iloc[0]


# ============================================================
# BASELINE AND SCENARIO DATA
# ============================================================

baseline = student[FEATURES].to_frame().T.copy()

scenario = baseline.copy()


st.subheader("📊 Current Academic Details")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Attendance",
    f"{float(student['Attendance']):.1f}%"
    if pd.notna(student["Attendance"])
    else "N/A",
)

col2.metric(
    "Study Hours",
    f"{float(student['Study_Hours']):.1f}"
    if pd.notna(student["Study_Hours"])
    else "N/A",
)

col3.metric(
    "Previous CGPA",
    f"{float(student['Previous_CGPA']):.2f}"
    if pd.notna(student["Previous_CGPA"])
    else "N/A",
)


# ============================================================
# HELPER FUNCTION FOR SLIDERS
# ============================================================

def get_numeric_bounds(column, default_min=0, default_max=100):
    """
    Return valid numeric slider limits based on dataset values.
    """

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return float(default_min), float(default_max)

    minimum = float(values.min())
    maximum = float(values.max())

    if maximum <= minimum:
        maximum = minimum + 1.0

    return minimum, maximum


def get_current_value(row, column, default=0.0):
    """
    Return a numeric value from a student record.
    """

    value = pd.to_numeric(
        pd.Series([row[column]]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(value):
        return float(default)

    return float(value)


# ============================================================
# WHAT-IF INPUTS
# ============================================================

st.markdown("---")

st.subheader("🎯 Adjust Academic Factors")

st.write(
    "Change the values below to explore a hypothetical scenario."
)

left_column, right_column = st.columns(2)


# ------------------------------------------------------------
# ATTENDANCE AND STUDY HOURS
# ------------------------------------------------------------

with left_column:

    attendance_min, attendance_max = get_numeric_bounds(
        "Attendance",
        0,
        100,
    )

    current_attendance = get_current_value(
        student,
        "Attendance",
        attendance_min,
    )

    attendance_value = st.slider(
        "Attendance (%)",
        min_value=attendance_min,
        max_value=attendance_max,
        value=float(
            np.clip(
                current_attendance,
                attendance_min,
                attendance_max,
            )
        ),
        step=1.0,
    )

    scenario.loc[0, "Attendance"] = attendance_value


    study_min, study_max = get_numeric_bounds(
        "Study_Hours",
        0,
        12,
    )

    study_min = max(0.0, study_min)
    study_max = max(study_max, study_min + 0.5)

    current_study_hours = get_current_value(
        student,
        "Study_Hours",
        study_min,
    )

    study_value = st.slider(
        "Study Hours Per Day",
        min_value=study_min,
        max_value=study_max,
        value=float(
            np.clip(
                current_study_hours,
                study_min,
                study_max,
            )
        ),
        step=0.5,
    )

    scenario.loc[0, "Study_Hours"] = study_value


# ------------------------------------------------------------
# INTERNAL, ASSIGNMENT AND QUIZ MARKS
# ------------------------------------------------------------

with right_column:

    mark_columns = [
        ("Internal_Marks", "Internal Marks"),
        ("Assignment_Marks", "Assignment Marks"),
        ("Quiz_Marks", "Quiz Marks"),
    ]

    for column, label in mark_columns:

        minimum, maximum = get_numeric_bounds(
            column,
            0,
            100,
        )

        current_value = get_current_value(
            student,
            column,
            minimum,
        )

        slider_value = st.slider(
            label,
            min_value=minimum,
            max_value=maximum,
            value=float(
                np.clip(
                    current_value,
                    minimum,
                    maximum,
                )
            ),
            step=1.0,
        )

        scenario.loc[0, column] = slider_value


# ============================================================
# PREDICTION
# ============================================================

st.markdown("---")

if st.button(
    "🔍 Compare Baseline and What-If Scenario",
    type="primary",
    use_container_width=True,
):

    try:

        # The saved pipeline performs preprocessing internally.
        baseline_prediction = float(
            model.predict(baseline)[0]
        )

        scenario_prediction = float(
            model.predict(scenario)[0]
        )

        difference = (
            scenario_prediction - baseline_prediction
        )

        # ----------------------------------------------------
        # DISPLAY METRICS
        # ----------------------------------------------------

        st.subheader("📈 Prediction Results")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Baseline Estimate",
            f"{baseline_prediction:.2f}",
        )

        metric2.metric(
            "What-If Estimate",
            f"{scenario_prediction:.2f}",
        )

        metric3.metric(
            "Estimated Change",
            f"{difference:+.2f}",
        )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        if difference > 0.01:

            st.success(
                "The model estimates a higher score "
                "for this hypothetical scenario."
            )

        elif difference < -0.01:

            st.warning(
                "The model estimates a lower score "
                "for this hypothetical scenario."
            )

        else:

            st.info(
                "The model estimates little or no change "
                "for this hypothetical scenario."
            )

        # ----------------------------------------------------
        # COMPARISON CHART
        # ----------------------------------------------------

        st.subheader("📊 Baseline vs What-If Comparison")

        comparison = pd.DataFrame(
            {
                "Scenario": [
                    "Baseline",
                    "What-If Scenario",
                ],
                "Estimated Final Marks": [
                    baseline_prediction,
                    scenario_prediction,
                ],
            }
        )

        st.bar_chart(
            comparison.set_index("Scenario")
        )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        st.info(
            """
            These values are estimates generated by a machine-learning
            model. They are not guaranteed future marks.

            The comparison does not prove that changing an individual
            academic factor will cause the predicted change.
            """
        )

    except Exception as exc:

        st.error(
            "The model could not calculate this scenario. "
            "Check whether the model was trained using the "
            "current academic dataset and expected features."
        )

        st.code(str(exc))


# ============================================================
# VIEW SCENARIO INPUTS
# ============================================================

with st.expander("View Academic Values Used in the Scenario"):

    st.dataframe(
        scenario,
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "TREKVERSE | AI-Powered Student Academic Management "
    "& Early-Warning System"
)