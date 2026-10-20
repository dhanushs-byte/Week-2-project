
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Step 1: Generate a sample loan-default dataset
X, y = make_classification(
    n_samples=1000,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    weights=[0.75, 0.25],
    random_state=42
)

# Step 2: Give the features meaningful names
df = pd.DataFrame(X, columns=[
    "income",
    "loan_amount",
    "credit_score",
    "debt_to_income",
    "employment_years",
    "late_payments"
])

# Step 3: Add the target column
# 0 = No default, 1 = Default
df["default"] = y

# Step 4: Save the dataset
df.to_csv("loan_default.csv", index=False)

print("Dataset created successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(df["default"].value_counts())


# Step 5: Separate input features and target
X = df.drop("default", axis=1)
y = df["default"]

# Step 6: Split off 20% for the final test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 7: Split the remaining 80% into 60% train and 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

# Display the split sizes
print("\n--- Dataset Split ---")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

print("\n--- Split Percentages ---")
print("Training:", len(X_train) / len(df) * 100, "%")
print("Validation:", len(X_val) / len(df) * 100, "%")
print("Test:", len(X_test) / len(df) * 100, "%")

print("\n--- Target Distribution ---")
print("Training:\n", y_train.value_counts(normalize=True))
print("Validation:\n", y_val.value_counts(normalize=True))
print("Test:\n", y_test.value_counts(normalize=True))


# TASK 1: Decision Tree Overfitting Experiment



# Different tree depths to compare
depths = [1, 3, 5, 10, None]

train_accuracies = []
val_accuracies = []

print("\n--- Decision Tree Overfitting Results ---")

for depth in depths:

    # Create the decision tree
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    # Train using training data only
    model.fit(X_train, y_train)

    # Predict on training and validation data
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    # Calculate accuracy
    train_acc = accuracy_score(y_train, train_predictions)
    val_acc = accuracy_score(y_val, val_predictions)

    # Save results for plotting
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(
        f"Depth: {depth}, "
        f"Train Accuracy: {train_acc:.4f}, "
        f"Validation Accuracy: {val_acc:.4f}"
    )

# Plot the results
depth_labels = [str(depth) for depth in depths]

plt.figure(figsize=(9, 5))

plt.plot(
    depth_labels,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    depth_labels,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Training vs Validation Accuracy")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("overfitting_plot.png", dpi=300)
plt.show()


# TASK 1: 5-Fold Cross-Validation
# Create a pipeline: scaling + Logistic Regression
logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# Perform 5-fold cross-validation on training data only
cv_scores = cross_val_score(
    logistic_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

# Calculate mean and standard deviation
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print("\n--- Logistic Regression: 5-Fold Cross-Validation ---")
print("Individual fold scores:", cv_scores)
print(f"Mean accuracy: {cv_mean:.4f}")
print(f"Standard deviation: {cv_std:.4f}")
print(f"Result: {cv_mean:.4f} +/- {cv_std:.4f}")


# TASK 1: Demonstrate Data Leakage

# ----------------------------------
# WRONG APPROACH: Scale before split
# ----------------------------------

wrong_scaler = StandardScaler()

# Leakage: learns statistics from ALL rows
X_scaled_wrong = pd.DataFrame(
    wrong_scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Select the already-defined splits
X_train_wrong = X_scaled_wrong.loc[X_train.index]
X_val_wrong = X_scaled_wrong.loc[X_val.index]

wrong_model = LogisticRegression(max_iter=1000)
wrong_model.fit(X_train_wrong, y_train)

wrong_predictions = wrong_model.predict(X_val_wrong)
wrong_accuracy = accuracy_score(y_val, wrong_predictions)


# ----------------------------------
# CORRECT APPROACH: Scale train only
# ----------------------------------

correct_scaler = StandardScaler()

# Learn scaling statistics from training rows only
X_train_scaled = pd.DataFrame(
    correct_scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

# Transform validation using the same fitted scaler
X_val_scaled = pd.DataFrame(
    correct_scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index
)

correct_model = LogisticRegression(max_iter=1000)
correct_model.fit(X_train_scaled, y_train)

correct_predictions = correct_model.predict(X_val_scaled)
correct_accuracy = accuracy_score(y_val, correct_predictions)


# ----------------------------------
# Compare the results
# ----------------------------------

print("\n--- Data Leakage Demonstration ---")

print("Wrong scaling - validation accuracy:",
      round(wrong_accuracy, 4))

print("Correct scaling - validation accuracy:",
      round(correct_accuracy, 4))

print("\nMean learned for income:")
print("All data:", round(wrong_scaler.mean_[0], 4))
print("Training only:", round(correct_scaler.mean_[0], 4))
