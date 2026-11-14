"""
Train a What-If Simulator model using Trekverse's current academic dataset.
Run from the project root:
    python src/train_what_if_model.py
"""
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_academic_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "what_if_model.pkl")

# These features are available before the final outcome and include the fields
# that the What-If Simulator lets the user change.
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

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing)
            + f"\nAvailable columns: {list(df.columns)}"
        )

    data = df[FEATURES + [TARGET]].copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    if len(data) < 10:
        raise ValueError("Not enough valid rows to train and test the model.")

    X = data[FEATURES]
    y = data[TARGET]

    numeric_features = [
        "Attendance", "Internal_Marks", "Assignment_Marks", "Quiz_Marks",
        "Study_Hours", "Previous_CGPA", "Backlogs",
        "Assignment_Submission_Rate", "Semester",
    ]
    categorical_features = ["Learning_Activity", "Department"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print("\nTREKVERSE WHAT-IF MODEL EVALUATION")
    print(f"Rows used: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"MAE:  {mean_absolute_error(y_test, predictions):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.3f}")
    print(f"R²:   {r2_score(y_test, predictions):.3f}")
    print("\nNote: This is a small dataset. Treat these metrics as an initial check,")
    print("not proof that predictions will generalize to other colleges or students.")

    # Retrain the final pipeline on all available rows after evaluating the split.
    pipeline.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved model pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
