import os
import joblib
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(
    BASE_DIR, "data", "student_academic_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "academic_performance_model.pkl"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "eda_graphs")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("=" * 65)
print("TREKVERSE - EXPLAINABLE AI ANALYSIS")
print("=" * 65)

df = pd.read_csv(DATASET_PATH)

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

target_column = "Final_Marks"

required_columns = feature_columns + [target_column]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing columns: {missing_columns}"
    )

df[target_column] = pd.to_numeric(
    df[target_column], errors="coerce"
)

df = df.dropna(subset=[target_column]).copy()

X = df[feature_columns].copy()
y = df[target_column].copy()

print("Dataset records:", len(df))
print("Input features:", len(feature_columns))


# --------------------------------------------------
# 3. RECREATE THE TRAIN/TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Test records:", len(X_test))


# --------------------------------------------------
# 4. LOAD THE NEW TRAINED PIPELINE
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Trained model not found. Run src/new_models.py first."
    )

model = joblib.load(MODEL_PATH)

print("\nSaved academic performance pipeline loaded!")

# Confirm that the model accepts the expected input columns.
expected_features = getattr(
    model, "feature_names_in_", None
)

if expected_features is not None:
    if list(expected_features) != feature_columns:
        raise ValueError(
            "The saved model's expected features do not "
            "match the current feature list."
        )


# --------------------------------------------------
# 5. CREATE SHAP EXPLAINER
# --------------------------------------------------

print("\nCreating SHAP explanations...")

# Use training records as the background data.
# The explainer calls the complete saved pipeline,
# including preprocessing and Ridge Regression.

background = X_train.sample(
    n=min(30, len(X_train)),
    random_state=42
)

# Limit the explanation set to keep computation manageable.
X_explain = X_test.iloc[
    :min(20, len(X_test))
].copy()

# Create a background dataset
background = X_train.sample(
    n=min(20, len(X_train)),
    random_state=42
).copy()

# Select test records to explain
X_explain = X_test.head(min(10, len(X_test))).copy()

# Convert categorical columns to numeric codes consistently
combined = pd.concat(
    [background, X_explain],
    axis=0
)

combined_encoded = pd.get_dummies(
    combined,
    columns=combined.select_dtypes(
        include=["object", "category"]
    ).columns.tolist(),
    dtype=float
)

# Split the encoded data back into background and test rows
background_encoded = combined_encoded.iloc[
    :len(background)
].copy()

X_explain_encoded = combined_encoded.iloc[
    len(background):
].copy()

# Use the same encoded columns for both datasets
background_encoded, X_explain_encoded = (
    background_encoded.align(
        X_explain_encoded,
        join="left",
        axis=1,
        fill_value=0
    )
)

# Explain the prediction function using encoded features.
# The wrapper converts encoded data back to the original input format
# expected by the saved preprocessing pipeline.
categorical_columns = background.select_dtypes(
    include=["object", "category"]
).columns.tolist()

encoded_columns = background_encoded.columns.tolist()


def predict_from_encoded(encoded_data):
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_columns
    )

    original_df = pd.DataFrame(
        index=encoded_df.index,
        columns=background.columns
    )

    for column in background.columns:
        if column not in categorical_columns:
            original_df[column] = encoded_df[column]
        else:
            matching_columns = [
                col for col in encoded_columns
                if col.startswith(column + "_")
            ]

            if matching_columns:
                selected = encoded_df[matching_columns].idxmax(axis=1)
                original_df[column] = selected.str[
                    len(column) + 1:
                ]
            else:
                original_df[column] = background[column].mode().iloc[0]

    return model.predict(original_df)


explainer = shap.Explainer(
    predict_from_encoded,
    background_encoded,
    algorithm="permutation"
)

print("Generating SHAP explanations...")

shap_explanation = explainer(
    X_explain_encoded,
    max_evals=2 * len(encoded_columns) + 1
)

print("SHAP explanations generated successfully!")

# Create a background dataset for SHAP
background = X_train.sample(
    n=min(30, len(X_train)),
    random_state=42
)

# Select a small set of test records
X_explain = X_test.head(min(20, len(X_test))).copy()

# Create SHAP explainer using the model pipeline
explainer = shap.Explainer(
    model.predict,
    background,
    algorithm="permutation"
)

print("Generating SHAP explanations...")

# Generate explanations

print("SHAP explanations generated successfully!")
print("SHAP values calculated successfully!")


# --------------------------------------------------
# 6. GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------

print("\nCreating global feature importance plot...")

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_explanation.values,
    X_explain_encoded,
    feature_names=X_explain_encoded.columns.tolist(),
    show=False
)

plt.title("Global Feature Importance - SHAP")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "20_shap_global_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Global importance plot saved!")


# --------------------------------------------------
# 7. SHAP SUMMARY PLOT
# --------------------------------------------------

print("\nCreating SHAP summary plot...")

plt.figure(figsize=(10, 7))

print("Creating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_explanation.values,
    features=X_explain_encoded.to_numpy(),
    feature_names=X_explain_encoded.columns.tolist(),
    show=False
)

plt.savefig(
    os.path.join(
        BASE_DIR,
        "eda_graphs",
        "21_shap_summary.png"
    ),
    bbox_inches="tight",
    dpi=300
)

plt.close()

print("SHAP summary plot saved!")

plt.title("SHAP Summary Plot")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "21_shap_summary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP summary plot saved!")


# --------------------------------------------------
# 8. INDIVIDUAL STUDENT EXPLANATION
# --------------------------------------------------

student_index = 0

student_features = X_explain.iloc[
    student_index
]

student_shap_values = shap_explanation.values[
    student_index
]

student_input = X_explain.iloc[
    [student_index]
]

prediction = model.predict(student_input)[0]

print("\n" + "=" * 65)
print("INDIVIDUAL STUDENT PREDICTION EXPLANATION")
print("=" * 65)

print("Test record index:", student_index)
print(f"Predicted Final Marks: {prediction:.2f}")

print(
    "Actual Final Marks:",
    float(y_test.iloc[student_index])
)


# --------------------------------------------------
# 9. CREATE EXPLANATION TABLE
# --------------------------------------------------

individual_explanation = pd.DataFrame({
    "Feature": X_explain_encoded.columns.tolist(),
    "Feature_Value": X_explain_encoded.iloc[0].values,
    "SHAP_Value": shap_explanation.values[0]
})

individual_explanation["Absolute_SHAP"] = (
    individual_explanation["SHAP_Value"].abs()
)

individual_explanation = individual_explanation.sort_values(
    "Absolute_SHAP",
    ascending=False
)

print("\nTop factors influencing this prediction:")

print(
    individual_explanation[
        ["Feature", "Feature_Value", "SHAP_Value"]
    ].head(10).to_string(index=False)
)


# --------------------------------------------------
# 10. SAVE EXPLANATION TABLE
# --------------------------------------------------

explanation_path = os.path.join(
    DATA_DIR,
    "individual_student_explanation.csv"
)

individual_explanation.to_csv(
    explanation_path,
    index=False
)

print("\nIndividual explanation saved:")
print(explanation_path)


# --------------------------------------------------
# 11. INDIVIDUAL EXPLANATION GRAPH
# --------------------------------------------------

top_features = (
    individual_explanation
    .head(10)
    .sort_values("SHAP_Value")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["SHAP_Value"]
)

plt.axvline(0, linewidth=1)

plt.xlabel("SHAP Value")
plt.ylabel("Feature")
plt.title("Individual Student Prediction Explanation")

plt.tight_layout()

individual_plot_path = os.path.join(
    OUTPUT_DIR,
    "22_individual_student_explanation.png"
)

plt.savefig(
    individual_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Individual explanation graph saved:")
print(individual_plot_path)


# --------------------------------------------------
# 12. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("EXPLAINABLE AI ANALYSIS COMPLETED")
print("=" * 65)