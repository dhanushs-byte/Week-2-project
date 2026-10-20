
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans

from xgboost import XGBClassifier

# =========================================
# STEP 1: LOAD AND PREPARE DATA
# =========================================

df = pd.read_csv("loan_default.csv")

rng = np.random.default_rng(42)

# Convert synthetic features into positive illustrative values
df["income"] = 30000 + np.abs(df["income"]) * 10000
df["loan_amount"] = 5000 + np.abs(df["loan_amount"]) * 5000

df["employment_type"] = rng.choice(
    ["Salaried", "Self-employed", "Business"],
    len(df)
)

df["credit_history"] = rng.choice(
    ["poor", "fair", "good"],
    len(df)
)

df["application_date"] = pd.date_range(
    "2025-01-01", periods=len(df), freq="D"
)

df.loc[0:19, "income"] = np.nan
df.loc[20:34, "employment_type"] = np.nan
df.loc[35:44, "credit_history"] = np.nan

df["application_year"] = df["application_date"].dt.year
df["application_month"] = df["application_date"].dt.month
df["income_loan_ratio"] = df["income"] / df["loan_amount"]
df["log_loan_amount"] = np.log1p(df["loan_amount"])

df.drop(columns=["application_date"], inplace=True)

# =========================================
# STEP 2: TRAIN / VALIDATION / TEST SPLIT
# =========================================

X = df.drop(columns=["default"])
y = df["default"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

numeric_features = [
    "income", "loan_amount", "credit_score",
    "debt_to_income", "employment_years", "late_payments",
    "application_year", "application_month",
    "income_loan_ratio", "log_loan_amount"
]

categorical_features = ["employment_type"]
ordinal_features = ["credit_history"]

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
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
    ("ordinal", ordinal_pipeline, ordinal_features)
])

# =========================================
# STEP 3: TRAIN THREE CLASSIFIERS
# =========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        oob_score=True,
        bootstrap=True,
        random_state=42
    )
}

results = {}
fitted_models = {}

for name, classifier in models.items():

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)

    results[name] = accuracy
    fitted_models[name] = model

    print(f"\n{name}")
    print("Validation accuracy:", round(accuracy, 4))

    if name == "Random Forest":
        print("OOB score:", round(
            model.named_steps["classifier"].oob_score_, 4
        ))

# =========================================
# STEP 4: XGBOOST WITH EARLY STOPPING
# =========================================

# Fit preprocessing on training data only
xgb_preprocessor = preprocessor

X_train_processed = xgb_preprocessor.fit_transform(X_train)
X_val_processed = xgb_preprocessor.transform(X_val)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="logloss",
    early_stopping_rounds=10,
    random_state=42
)

xgb_model.fit(
    X_train_processed,
    y_train,
    eval_set=[(X_val_processed, y_val)],
    verbose=False
)

xgb_predictions = xgb_model.predict(X_val_processed)

results["XGBoost"] = accuracy_score(y_val, xgb_predictions)
fitted_models["XGBoost"] = xgb_model

print("\nXGBoost")
print("Validation accuracy:", round(results["XGBoost"], 4))
print("Best iteration:", xgb_model.best_iteration)

# =========================================
# STEP 5: PREDICT PROBABILITIES
# =========================================

logistic_model = fitted_models["Logistic Regression"]

probabilities = logistic_model.predict_proba(X_val.iloc[:5])
predictions = logistic_model.predict(X_val.iloc[:5])

print("\n--- Logistic Regression: First 5 Rows ---")
print("Predicted classes:", predictions)
print("Probabilities [class 0, class 1]:")
print(probabilities)

# =========================================
# STEP 6: FEATURE IMPORTANCE
# =========================================

feature_names = (
    numeric_features
    + list(
        fitted_models["Random Forest"]
        .named_steps["preprocessing"]
        .named_transformers_["categorical"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_features)
    )
    + ordinal_features
)

forest = fitted_models["Random Forest"].named_steps["classifier"]

forest_importance = pd.Series(
    forest.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

xgb_importance = pd.Series(
    xgb_model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nTop 5 Random Forest Features:")
print(forest_importance.head(5))

print("\nTop 5 XGBoost Features:")
print(xgb_importance.head(5))

# =========================================
# STEP 7: REGRESSION MODELS
# =========================================

# Predict loan amount without using loan_amount or
# features derived from loan_amount as input.
regression_features = [
    "income", "credit_score", "debt_to_income",
    "employment_years", "late_payments",
    "application_year", "application_month"
]

X_reg = df[regression_features]
y_reg = df["loan_amount"]

Xr_train, Xr_val, yr_train, yr_val = train_test_split(
    X_reg, y_reg,
    test_size=0.20,
    random_state=42
)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0, max_iter=10000)
}

regression_results = {}

for name, reg_model in regression_models.items():

    reg_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", reg_model)
    ])

    reg_pipeline.fit(Xr_train, yr_train)
    score = reg_pipeline.score(Xr_val, yr_val)

    regression_results[name] = reg_pipeline

    print(f"\n{name}")
    print("Validation R-squared:", round(score, 4))
    print("Coefficients:",
          reg_pipeline.named_steps["model"].coef_)

lasso_model = regression_results["Lasso"].named_steps["model"]

print("Lasso zero coefficients:",
      np.sum(np.isclose(lasso_model.coef_, 0)))

# =========================================
# STEP 8: K-MEANS CLUSTERING
# =========================================

cluster_features = [
    "income", "loan_amount", "credit_score"
]

cluster_data = SimpleImputer(strategy="median").fit_transform(
    df[cluster_features]
)

cluster_data = StandardScaler().fit_transform(cluster_data)

k_values = range(2, 9)
inertias = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(cluster_data)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(
        silhouette_score(cluster_data, labels)
    )

    print(
        f"k={k}, inertia={kmeans.inertia_:.2f}, "
        f"silhouette={silhouette_scores[-1]:.4f}"
    )

# Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("kmeans_elbow.png", dpi=300)
plt.show()

# Silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Scores")
plt.grid(True)
plt.tight_layout()
plt.savefig("kmeans_silhouette.png", dpi=300)
plt.show()

# =========================================
# STEP 9: RESULTS TABLE
# =========================================

results_table = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Validation Accuracy"]
)

print("\n--- Classification Results ---")
print(results_table)

results_table.to_csv(
    "task3_classification_results.csv", index=False
)

print("\nTask 3 experiments completed.")
