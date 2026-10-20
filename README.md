# Week 2 Machine Learning Assessment

## Overview

This project demonstrates a machine learning workflow using Python and Scikit-learn. It covers data splitting, preprocessing, feature engineering, classification, regression, clustering, model evaluation, and hyperparameter tuning.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib

## Project Tasks

### Task 1: Machine Learning Fundamentals
- Split the dataset into training, validation, and test sets.
- Compare Decision Tree models with different depths.
- Perform five-fold cross-validation with Logistic Regression.
- Demonstrate data leakage and explain the machine learning workflow.

### Task 2: Data Preprocessing and Pipelines
- Handle missing numerical and categorical values.
- Apply one-hot and ordinal encoding.
- Perform feature engineering.
- Build a preprocessing and classification pipeline.
- Save and reload the fitted pipeline using Joblib.
- Generate predictions for new input records.

### Task 3: Model Training and Comparison
- Train Logistic Regression, Decision Tree, Random Forest, and XGBoost classifiers.
- Compare classification performance.
- Examine feature importance.
- Compare Linear Regression, Ridge, and Lasso.
- Apply K-Means clustering and evaluate different cluster counts.

### Task 4: Model Evaluation and Tuning
- Calculate accuracy, precision, recall, and F1-score.
- Plot confusion matrices and ROC curves.
- Compare AUC scores.
- Select a classification threshold to achieve at least 75% recall on validation data.
- Tune Random Forest with GridSearchCV.
- Tune XGBoost with RandomizedSearchCV.
- Compare regression metrics against a mean baseline.
- Evaluate the selected classifier on the held-out test set.

## Key Results

### Classification

The initial validation results were:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.800 | 0.6923 | 0.3600 | 0.4737 |
| Decision Tree | 0.840 | 0.6957 | 0.6400 | 0.6667 |
| Random Forest | 0.855 | 0.8000 | 0.5600 | 0.6588 |
| XGBoost | 0.860 | 0.7391 | 0.6800 | 0.7083 |

### ROC-AUC

| Model | AUC |
|---|---:|
| Logistic Regression | 0.786 |
| Random Forest | 0.921 |
| XGBoost | 0.916 |

### Threshold Selection

The XGBoost validation threshold was set to 0.38, achieving 75% precision and 84% recall.

### Hyperparameter Tuning

- Random Forest best cross-validation F1-score: 0.6992.
- Tuned XGBoost best cross-validation F1-score: 0.7806.
- Tuned XGBoost validation F1-score: 0.8000.

### Final Test Evaluation

The tuned XGBoost model achieved:

- Accuracy: 0.89
- Precision: 0.85
- Recall: 0.68
- F1-score: 0.7556

### Regression

Linear Regression achieved an MAE of 2930.6047, RMSE of 3859.8346, and R² of 0.2334. The mean baseline performed worse on these validation metrics.

## How to Run

1. Install Python and open this project folder in VS Code.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run each task:

   ```bash
   python task1.py
   python task2.py
   python task3.py
   python task4.py
   ```

Run the scripts from the project folder and ensure `loan_default.csv` is present.

## Notes

- Results depend on the dataset and software versions.
- The dataset used in this assessment is illustrative; results should not be interpreted as a real-world credit-risk assessment.
- The test set is reserved for final evaluation and should not be used for tuning.
