
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# TREKVERSE - EARLY WARNING SYSTEM
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "student_academic_data.csv"
MODEL_PATH = BASE_DIR / "models" / "academic_performance_model.pkl"
REPORT_DIR = BASE_DIR / "reports"
REPORT_PATH = REPORT_DIR / "early_warning_report.csv"

FEATURE_COLUMNS = [
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
    "Learning_Activity",
]


def get_risk_level(predicted_marks):
    """Assign risk using the project's current provisional thresholds."""
    if predicted_marks < 55:
        return "High Risk"
    elif predicted_marks < 65:
        return "Moderate Risk"
    else:
        return "Low Risk"


def get_recommendation(risk_level):
    """Generate an initial academic support recommendation."""

    recommendations = {
        "High Risk": (
            "Priority faculty review; prepare an individual "
            "study plan and monitor progress."
        ),
        "Moderate Risk": (
            "Provide additional practice, monitor attendance, "
            "and review assignment completion."
        ),
        "Low Risk": (
            "Continue current study habits and maintain "
            "regular academic progress."
        ),
    }

    return recommendations[risk_level]


def generate_early_warning_report():
    # Check required files
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    # Load dataset and trained model
    data = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    if data.empty:
        raise ValueError("The student dataset is empty.")

    # Validate required model features
    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Prepare model input
    features = data[FEATURE_COLUMNS].copy()

    # Predict final marks for every student
    predictions = model.predict(features)

    # Prepare report
    report = pd.DataFrame()

    report["Student_ID"] = (
        data["Student_ID"]
        if "Student_ID" in data.columns
        else [f"STU{i:03d}" for i in range(1, len(data) + 1)]
    )

    if "Name" in data.columns:
        report["Name"] = data["Name"]

    if "Department" in data.columns:
        report["Department"] = data["Department"]

    if "Semester" in data.columns:
        report["Semester"] = data["Semester"]

    # Include actual marks when available for comparison
    if "Final_Marks" in data.columns:
        report["Actual_Final_Marks"] = data["Final_Marks"]

    report["Predicted_Final_Marks"] = [
        round(max(0.0, min(100.0, float(mark))), 2)
        for mark in predictions
    ]

    report["Risk_Level"] = report[
        "Predicted_Final_Marks"
    ].apply(get_risk_level)

    report["Recommendation"] = report[
        "Risk_Level"
    ].apply(get_recommendation)

    # Show students needing attention first
    risk_order = {
        "High Risk": 0,
        "Moderate Risk": 1,
        "Low Risk": 2,
    }

    report["_Risk_Order"] = report["Risk_Level"].map(risk_order)

    report = report.sort_values(
        by=["_Risk_Order", "Predicted_Final_Marks"],
        ascending=[True, True],
    )

    report = report.drop(columns=["_Risk_Order"])

    # Create reports folder if it does not exist
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save report to CSV
    report.to_csv(REPORT_PATH, index=False)

    # Display summary
    risk_counts = report["Risk_Level"].value_counts()

    high_risk = int(risk_counts.get("High Risk", 0))
    moderate_risk = int(risk_counts.get("Moderate Risk", 0))
    low_risk = int(risk_counts.get("Low Risk", 0))

    print("\n" + "=" * 55)
    print("       TREKVERSE EARLY WARNING SYSTEM")
    print("=" * 55)

    print(f"Total Students Analysed : {len(report)}")
    print(f"High-Risk Students      : {high_risk}")
    print(f"Moderate-Risk Students  : {moderate_risk}")
    print(f"Low-Risk Students       : {low_risk}")

    print("-" * 55)
    print("STUDENTS REQUIRING ACADEMIC SUPPORT")
    print("-" * 55)

    students_needing_support = report[
        report["Risk_Level"].isin(["High Risk", "Moderate Risk"])
    ]

    display_columns = [
        "Student_ID",
        "Predicted_Final_Marks",
        "Risk_Level",
    ]

    if "Name" in report.columns:
        display_columns.insert(1, "Name")

    if students_needing_support.empty:
        print("No students were flagged for support.")
    else:
        print(
            students_needing_support[display_columns].to_string(
                index=False
            )
        )

    print("-" * 55)
    print(f"Report saved to:\n{REPORT_PATH}")
    print("=" * 55)

    return report


if __name__ == "__main__":
    generate_early_warning_report()
