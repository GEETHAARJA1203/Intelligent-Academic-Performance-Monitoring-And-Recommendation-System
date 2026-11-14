
import os
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Dataset path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_academic_data.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Select target column
target_column = "Final_Marks"

if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

# Use numeric columns only and exclude target and identifier columns
numeric_df = df.select_dtypes(include=["number"]).copy()

X = numeric_df.drop(columns=[target_column], errors="ignore")
y = numeric_df[target_column]

if X.empty:
    raise ValueError("No numeric input features found.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models for comparison
models = {
    "Baseline (Mean)": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Test-set predictions
    predictions = pipeline.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    # Cross-validation on training data
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()

    results.append({
        "Model": name,
        "Test MAE": round(mae, 3),
        "Test RMSE": round(rmse, 3),
        "Test R2": round(r2, 3),
        "5-Fold CV MAE": round(cv_mae, 3)
    })

# Display results
results_df = pd.DataFrame(results).sort_values(by="Test MAE")

print("\nTREKVERSE - MODEL VALIDATION RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))
print("=" * 60)
print("Lower MAE/RMSE is better. Higher R2 is generally better.")
