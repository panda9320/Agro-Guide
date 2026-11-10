from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Change this to "fertilizer" to train the fertilizer model.
TASK = "fertilizer"

# Configure the actual target column for your fertilizer dataset.
FERTILIZER_TARGET = "fertilizer"

if TASK == "Crop":
    DATA_FILE = DATA_DIR / "crop_dataset_clean.csv"
    TARGET = "Crop"
else:
    DATA_FILE = DATA_DIR / "crop_fertilizer_recommendation_clean.csv"
    TARGET = FERTILIZER_TARGET


# 1. Load and validate the dataset
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.drop_duplicates()
df = df.dropna(subset=[TARGET])

X = df.drop(columns=[TARGET])
y = df[TARGET]

if X.empty or y.nunique() < 2:
    raise ValueError("Dataset needs features and at least two classes.")

print("Dataset shape:", df.shape)
print("Number of classes:", y.nunique())
print("Target distribution:\n", y.value_counts())


# 2. Split before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Identify numerical and categorical input columns.
numeric_cols = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    exclude="number"
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])


# 3. Define the three candidate models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    ),
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced",
    ),
    "Gaussian Naive Bayes": GaussianNB(),
}

# Use the same folds to compare models fairly.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = []
best_name = None
best_cv_score = -1
best_pipeline = None

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model),
    ])

    # Cross-validation uses training data only.
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        error_score="raise",
    )

    cv_score = scores.mean()

    # Fit on all training data, then evaluate on held-out test data.
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results.append({
        "Model": name,
        "CV Macro F1": cv_score,
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test Macro F1": f1,
    })

    print(f"\n--- {name} ---")
    print(f"Mean CV Macro F1: {cv_score:.6f}")
    print(f"Test Accuracy:    {accuracy:.6f}")
    print(f"Test Precision:   {precision:.6f}")
    print(f"Test Recall:      {recall:.6f}")
    print(f"Test Macro F1:    {f1:.6f}")

    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    print("Training Accuracy:",
        accuracy_score(y_train, train_predictions))

    print("Testing Accuracy:",
        accuracy_score(y_test, test_predictions))
    
    if cv_score > best_cv_score:
        best_cv_score = cv_score
        best_name = name
        best_pipeline = pipeline


# 4. Print a comparison table
results_df = pd.DataFrame(results)
print("\nModel comparison:")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    MODEL_DIR / f"{TASK}_model_comparison.csv",
    index=False,
)


# 5. Save the best model and its expected input columns
model_path = MODEL_DIR / f"{TASK}_model.pkl"
joblib.dump(
    {
        "model": best_pipeline,
        "feature_columns": X.columns.tolist(),
        "target_column": TARGET,
        "task": TASK,
    },
    model_path,
)

print(f"\nBest model by CV Macro F1: {best_name}")
print(f"Saved model to: {model_path}")


# 6. Detailed evaluation of the selected model
final_predictions = best_pipeline.predict(X_test)

print("\nClassification report:")
print(classification_report(
    y_test,
    final_predictions,
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(
    y_test,
    final_predictions,
))
