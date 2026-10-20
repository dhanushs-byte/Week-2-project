
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer
)
from sklearn.linear_model import LogisticRegression

# -----------------------------------------
# STEP 1: Load the Task 1 dataset
# -----------------------------------------

df = pd.read_csv("loan_default.csv")

# Convert synthetic numeric features into positive,
# illustrative income and loan amount values.
df["income"] = 30000 + np.abs(df["income"]) * 10000
df["loan_amount"] = 5000 + np.abs(df["loan_amount"]) * 5000

# Add categorical features
rng = np.random.default_rng(42)

df["employment_type"] = rng.choice(
    ["Salaried", "Self-employed", "Business"],
    size=len(df)
)

df["credit_history"] = rng.choice(
    ["poor", "fair", "good"],
    size=len(df)
)

# Add an application date
df["application_date"] = pd.date_range(
    start="2025-01-01",
    periods=len(df),
    freq="D"
)

# Introduce some missing values for the imputation exercise
df.loc[0:19, "income"] = np.nan
df.loc[20:34, "employment_type"] = np.nan
df.loc[35:44, "credit_history"] = np.nan

# -----------------------------------------
# STEP 2: Feature engineering
# -----------------------------------------

# Feature 1 and 2: extract year and month from date
df["application_year"] = df["application_date"].dt.year
df["application_month"] = df["application_date"].dt.month

# Feature 3: income-to-loan ratio
df["income_loan_ratio"] = (
    df["income"] / df["loan_amount"]
)

# Feature 4: log transformation of loan amount
df["log_loan_amount"] = np.log1p(df["loan_amount"])

# Remove original date because the model uses date parts
df = df.drop(columns=["application_date"])

print("--- Dataset Preview ---")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset shape:", df.shape)



# -----------------------------------------
# STEP 3: Experiment with KNNImputer
# -----------------------------------------

numeric_columns = [
    "income",
    "loan_amount",
    "credit_score",
    "debt_to_income",
    "employment_years",
    "late_payments",
    "application_year",
    "application_month",
    "income_loan_ratio",
    "log_loan_amount"
]

# Keep the original numeric data for comparison
numeric_data = df[numeric_columns].copy()

# KNN imputation experiment
knn_imputer = KNNImputer(n_neighbors=5)
numeric_knn_result = knn_imputer.fit_transform(numeric_data)

print("\n--- KNN Imputer Experiment ---")
print("Missing values before:", numeric_data.isnull().sum().sum())
print("Missing values after:", np.isnan(numeric_knn_result).sum())



# -----------------------------------------
# STEP 4: Prepare input features and target
# -----------------------------------------

X = df.drop(columns=["default"])
y = df["default"]

# Numeric columns used by the final model
numeric_features = numeric_columns

# Nominal category: no natural ordering
categorical_features = ["employment_type"]

# Ordered category: poor < fair < good
ordinal_features = ["credit_history"]

# -----------------------------------------
# STEP 5: Numeric preprocessing
# -----------------------------------------

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# -----------------------------------------
# STEP 6: Categorical preprocessing
# -----------------------------------------

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

ordinal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(
        categories=[["poor", "fair", "good"]]
    ))
])

# -----------------------------------------
# STEP 7: Combine preprocessing
# -----------------------------------------

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
    ("ordinal", ordinal_pipeline, ordinal_features)
])

# -----------------------------------------
# STEP 8: Complete ML pipeline
# -----------------------------------------

model_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("\nPipeline created successfully!")
print(model_pipeline)



# -----------------------------------------
# STEP 9: Cross-validation
# -----------------------------------------

cv_scores = cross_val_score(
    model_pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n--- Pipeline Cross-Validation ---")
print("Fold scores:", cv_scores)
print(f"Mean accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")



# -----------------------------------------
# STEP 10: Fit and save the pipeline
# -----------------------------------------

# Keep the final test data out of model fitting.
# Split before fitting the final pipeline.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model_pipeline.fit(X_train, y_train)

# Save the complete fitted pipeline
joblib.dump(model_pipeline, "loan_default_pipeline.joblib")

print("\nPipeline saved successfully!")

# -----------------------------------------
# STEP 11: Load the saved pipeline
# -----------------------------------------

loaded_pipeline = joblib.load("loan_default_pipeline.joblib")

# Three new RAW rows: no manual imputation,
# encoding or scaling is needed.
new_applicants = pd.DataFrame([
    {
        "income": 45000,
        "loan_amount": 12000,
        "credit_score": 0.5,
        "debt_to_income": -0.3,
        "employment_years": 1.2,
        "late_payments": -0.4,
        "employment_type": "Salaried",
        "credit_history": "good",
        "application_year": 2026,
        "application_month": 10,
        "income_loan_ratio": 45000 / 12000,
        "log_loan_amount": np.log1p(12000)
    },
    {
        "income": 28000,
        "loan_amount": 18000,
        "credit_score": -0.6,
        "debt_to_income": 1.0,
        "employment_years": -0.5,
        "late_payments": 1.1,
        "employment_type": "Self-employed",
        "credit_history": "poor",
        "application_year": 2026,
        "application_month": 10,
        "income_loan_ratio": 28000 / 18000,
        "log_loan_amount": np.log1p(18000)
    },
    {
        "income": 60000,
        "loan_amount": 10000,
        "credit_score": 0.8,
        "debt_to_income": -0.7,
        "employment_years": 2.0,
        "late_payments": -0.8,
        "employment_type": "Business",
        "credit_history": "fair",
        "application_year": 2026,
        "application_month": 10,
        "income_loan_ratio": 60000 / 10000,
        "log_loan_amount": np.log1p(10000)
    }
])

predictions = loaded_pipeline.predict(new_applicants)

print(f"Pipeline CV accuracy: {cv_scores.mean():.2f}")
print("Predictions for 3 New Rows:", predictions)
