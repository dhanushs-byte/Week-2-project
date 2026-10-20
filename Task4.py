
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# 1. Load dataset
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "loan_default.csv")

# 2. Recreate Task 2 / Task 3 features
rng = np.random.default_rng(42)

df["income"] = 30000 + np.abs(df["income"]) * 10000
df["loan_amount"] = 5000 + np.abs(df["loan_amount"]) * 5000

df["employment_type"] = rng.choice(
    ["Salaried", "Self-employed", "Business"], len(df)
)
df["credit_history"] = rng.choice(
    ["poor", "fair", "good"], len(df)
)

df["application_date"] = pd.date_range(
    start="2025-01-01", periods=len(df), freq="D"
)

df["application_year"] = df["application_date"].dt.year
df["application_month"] = df["application_date"].dt.month
df["income_loan_ratio"] = (
    df["income"] / df["loan_amount"].replace(0, np.nan)
)
df["log_loan_amount"] = np.log1p(df["loan_amount"])

# Target and features
X = df.drop(columns=["default", "application_date"])
y = df["default"]

numeric_features = [
    col for col in X.select_dtypes(include=np.number).columns
]
categorical_features = ["employment_type"]
ordinal_features = ["credit_history"]

# 3. Split into 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

# 4. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[["poor", "fair", "good"]]
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
    ("ord", ordinal_pipeline, ordinal_features)
])

# 5. Four classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )
}

results = []

# 6. Train and evaluate on validation data
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(y_val, predictions, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    cm = confusion_matrix(y_val, predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Default", "Default"]
    )
    display.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

# 7. Compare all models
results_df = pd.DataFrame(results)
print("\nClassification Results")
print(results_df.to_string(index=False))

results_df.to_csv(
    BASE_DIR / "task4_classification_results.csv",
    index=False
)
print("\nResults saved to task4_classification_results.csv")


# Compare three classifiers using ROC curves
roc_models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]

plt.figure(figsize=(8, 6))

for name in roc_models:
    model = models[name]

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    # Probability of class 1 (Default)
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_val, probabilities)
    auc_score = roc_auc_score(y_val, probabilities)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc_score:.3f})"
    )

# Random guessing reference line
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curves - Classification Models")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(BASE_DIR / "task4_roc_curves.png")
plt.show()




# Train XGBoost on training data only
xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    ))
])

xgb_pipeline.fit(X_train, y_train)

# Get probabilities for the positive class
y_prob = xgb_pipeline.predict_proba(X_val)[:, 1]

# Calculate recall and precision at different thresholds
thresholds = np.arange(0.05, 1.00, 0.01)
valid_thresholds = []

for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)

    recall = recall_score(
        y_val, y_pred_threshold, zero_division=0
    )
    precision = precision_score(
        y_val, y_pred_threshold, zero_division=0
    )

    if recall >= 0.75:
        valid_thresholds.append(
            (threshold, precision, recall)
        )

if valid_thresholds:
    # Choose the qualifying threshold with the highest precision
    best_threshold, best_precision, best_recall = max(
        valid_thresholds, key=lambda item: item[1]
    )

    print("\nThreshold Selection - XGBoost")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")
else:
    print("No tested threshold achieved 75% recall.")

# ---------------------------------
# A. Random Forest - GridSearchCV
# ---------------------------------

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42
    ))
])

rf_parameters = {
    "classifier__n_estimators": [100, 150, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5, 10]
}

rf_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_parameters,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train, y_train)

print("\nRandom Forest - GridSearchCV")
print("Best parameters:", rf_search.best_params_)
print("Best CV F1-score:", round(rf_search.best_score_, 4))

rf_val_predictions = rf_search.best_estimator_.predict(X_val)
print(
    "Validation F1-score:",
    round(f1_score(y_val, rf_val_predictions), 4)
)

# ---------------------------------
# B. XGBoost - RandomizedSearchCV
# ---------------------------------

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ))
])

xgb_parameters = {
    "classifier__n_estimators": [100, 150, 200, 300],
    "classifier__max_depth": [2, 3, 4, 5, 6],
    "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_parameters,
    n_iter=15,
    scoring="f1",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)

print("\nXGBoost - RandomizedSearchCV")
print("Best parameters:", xgb_search.best_params_)
print("Best CV F1-score:", round(xgb_search.best_score_, 4))

xgb_val_predictions = xgb_search.best_estimator_.predict(X_val)
print(
    "Validation F1-score:",
    round(f1_score(y_val, xgb_val_predictions), 4)

)

# ==========================================
# A. Regression: predict loan amount
# ==========================================

# Avoid target leakage: remove loan_amount and
# features calculated directly from loan_amount.
regression_features = df.drop(columns=[
    "default",
    "application_date",
    "loan_amount",
    "log_loan_amount",
    "income_loan_ratio"
])

y_reg = df["loan_amount"]

# Use the same row splits as classification
X_reg_train = regression_features.loc[X_train.index]
X_reg_val = regression_features.loc[X_val.index]

y_reg_train = y_reg.loc[X_train.index]
y_reg_val = y_reg.loc[X_val.index]

reg_numeric = list(
    X_reg_train.select_dtypes(include=np.number).columns
)
reg_categorical = ["employment_type"]
reg_ordinal = ["credit_history"]

reg_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), reg_numeric),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), reg_categorical),

    ("ord", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["poor", "fair", "good"]]
        ))
    ]), reg_ordinal)
])

regression_model = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("regressor", LinearRegression())
])

regression_model.fit(X_reg_train, y_reg_train)
reg_predictions = regression_model.predict(X_reg_val)

# Mean baseline: always predict training mean
mean_prediction = np.full(
    len(y_reg_val), y_reg_train.mean()
)

def show_regression_metrics(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

regression_results = []

regression_results.append(show_regression_metrics(
    "Linear Regression", y_reg_val, reg_predictions
))

regression_results.append(show_regression_metrics(
    "Mean Baseline", y_reg_val, mean_prediction
))

pd.DataFrame(regression_results).to_csv(
    BASE_DIR / "task4_regression_results.csv",
    index=False
)

# ==========================================
# B. Final test-set evaluation - once
# ==========================================

# Select the tuned XGBoost model based on validation F1.
# Do not compare models or tune parameters on the test set.
best_pipeline = xgb_search.best_estimator_

final_test_predictions = best_pipeline.predict(X_test)

print("\nFINAL TEST RESULTS - TUNED XGBOOST")
print("Accuracy:",
      round(accuracy_score(y_test, final_test_predictions), 4))
print("Precision:",
      round(precision_score(
          y_test, final_test_predictions, zero_division=0
      ), 4))
print("Recall:",
      round(recall_score(
          y_test, final_test_predictions, zero_division=0
      ), 4))
print("F1-score:",
      round(f1_score(
          y_test, final_test_predictions, zero_division=0
      ), 4))

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, final_test_predictions))

print("\nTest Classification Report:")
print(classification_report(
    y_test,
    final_test_predictions,
    target_names=["No Default", "Default"],
    zero_division=0
))
