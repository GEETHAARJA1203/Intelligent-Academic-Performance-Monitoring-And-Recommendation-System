import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(
    BASE_DIR, "data", "student_academic_data.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

print("\n" + "=" * 65)
print("TREKVERSE - MODEL VALIDATION AND TRAINING")
print("=" * 65)

df = pd.read_csv(DATASET_PATH)

required_columns = [
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
    "Final_Marks"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["Final_Marks"] = pd.to_numeric(
    df["Final_Marks"], errors="coerce"
)

df = df.dropna(subset=["Final_Marks"]).copy()

feature_columns = [
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

X = df[feature_columns].copy()
y = df["Final_Marks"].copy()

print(f"Dataset records: {len(df)}")
print(f"Input features: {len(feature_columns)}")
print("Prediction target: Final_Marks")


# --------------------------------------------------
# 3. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")


# --------------------------------------------------
# 4. PREPROCESSING PIPELINE
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# --------------------------------------------------
# 5. DEFINE MODELS
# --------------------------------------------------

models = {
    "Mean Baseline": DummyRegressor(strategy="mean"),

    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(),

    "Lasso Regression": Lasso(max_iter=10000),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Support Vector Regression": SVR(),

    "XGBoost": XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        verbosity=-1,
        random_state=42
    ),

    "CatBoost": CatBoostRegressor(
        iterations=150,
        learning_rate=0.05,
        depth=4,
        verbose=0,
        random_seed=42
    )
}


# --------------------------------------------------
# 6. FIVE-FOLD CROSS-VALIDATION
# --------------------------------------------------

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}

results = {}
successful_models = {}

for name, estimator in models.items():

    print(f"\nEvaluating {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    try:
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise"
        )

        mean_mae = -scores["test_MAE"].mean()
        mean_rmse = -scores["test_RMSE"].mean()
        mean_r2 = scores["test_R2"].mean()

        results[name] = {
            "Model": name,
            "CV_MAE": mean_mae,
            "CV_RMSE": mean_rmse,
            "CV_R2": mean_r2
        }

        successful_models[name] = estimator

        print(
            f"CV MAE: {mean_mae:.3f} | "
            f"CV RMSE: {mean_rmse:.3f} | "
            f"CV R2: {mean_r2:.3f}"
        )

    except Exception as error:
        print(f"Could not evaluate {name}: {error}")


# --------------------------------------------------
# 7. SELECT MODEL USING CROSS-VALIDATION
# --------------------------------------------------

if not results:
    raise RuntimeError(
        "No models completed cross-validation successfully."
    )

results_df = pd.DataFrame(
    list(results.values())
).sort_values(
    by="CV_RMSE",
    ascending=True
)

best_name = results_df.iloc[0]["Model"]
best_estimator = successful_models[best_name]

print("\n" + "=" * 65)
print("CROSS-VALIDATION MODEL COMPARISON")
print("=" * 65)

print(
    results_df.round(4).to_string(index=False)
)

print(f"\nSelected model: {best_name}")


# --------------------------------------------------
# 8. TRAIN SELECTED MODEL
# --------------------------------------------------

best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_estimator)
])

best_pipeline.fit(X_train, y_train)


# --------------------------------------------------
# 9. FINAL HOLDOUT TEST
# --------------------------------------------------

test_predictions = best_pipeline.predict(X_test)

test_mae = mean_absolute_error(
    y_test, test_predictions
)

test_rmse = mean_squared_error(
    y_test, test_predictions
) ** 0.5

test_r2 = r2_score(
    y_test, test_predictions
)

print("\n" + "=" * 65)
print("FINAL HOLDOUT TEST RESULTS")
print("=" * 65)

print(f"Selected model: {best_name}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R2: {test_r2:.4f}")


# --------------------------------------------------
# 10. SAVE MODEL AND RESULTS
# --------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "academic_performance_model.pkl"
)

results_path = os.path.join(
    BASE_DIR,
    "new_model_results.csv"
)

joblib.dump(best_pipeline, model_path)

results_df.to_csv(
    results_path,
    index=False
)

# Save final test metrics separately.
test_results = pd.DataFrame([{
    "Selected_Model": best_name,
    "Test_MAE": test_mae,
    "Test_RMSE": test_rmse,
    "Test_R2": test_r2,
    "Training_Records": len(X_train),
    "Test_Records": len(X_test)
}])

test_results_path = os.path.join(
    BASE_DIR,
    "final_model_test_results.csv"
)

test_results.to_csv(
    test_results_path,
    index=False
)

print("\n" + "=" * 65)
print("TRAINING COMPLETED")
print("=" * 65)

print(f"Model saved: {model_path}")
print(f"CV results saved: {results_path}")
print(f"Final test results saved: {test_results_path}")