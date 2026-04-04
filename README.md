 # Diabetes Prediction using Machine Learning
 
-## Problem Statement
-The goal of this project is to predict whether a patient has diabetes based on diagnostic medical features.
+A reproducible machine-learning workflow for predicting diabetes using the Pima Indians Diabetes dataset.
 
-## Dataset
-Pima Indians Diabetes Dataset  
-768 rows, 9 columns.
-
-## Workflow Followed
-1. Data Loading
-2. Data Cleaning (handling impossible zero values)
-3. Exploratory Data Analysis (EDA)
-4. Feature Preparation
-5. Logistic Regression Model
-6. Model Evaluation
-
-## Results
-- Accuracy: **75%**
-- Logistic Regression performed reasonably well for this dataset.
-
-## Visualizations
-
-### Outcome Count
-![Outcome Count](images/outcome_count.png)
+## What this project improves
+- Prevents **data leakage** by fitting imputers/scalers only on training folds.
+- Uses **Stratified 5-Fold Cross-Validation** for robust model comparison.
+- Compares multiple models (Logistic Regression and Random Forest).
+- Optimizes a prediction threshold with an **F2 score** focus (higher recall).
+- Reports medical-relevant metrics: ROC-AUC, PR-AUC, Recall, and Specificity.
+- Saves artifacts (metrics, plots, and trained model) for reproducibility.
 
-### Correlation Heatmap
-![Heatmap](images/correlation_heatmap.png)
-
-## Tools Used
-- Python
-- Pandas
-- Seaborn
-- Scikit-learn
-- VS Code
-
-## Conclusion
-Basic medical attributes can be used to reasonably predict diabetes using a simple machine learning model.
-This project analyzes diabetes data using Python.
+## Dataset
+- Pima Indians Diabetes Dataset
+- Expected local file: `diabetes.csv`
+- Typical shape: 768 rows, 9 columns
 
-## Files
-- `diabetes_analysis.py`: Main analysis script
-- `diabetes.csv`: Dataset containing diabetes information
-- `diabetes_project`.
+## Project Structure
+- `diabetes_analysis.py` – main training/evaluation script
+- `requirements.txt` – Python dependencies
+- `tests/` – unit tests for core helper logic
+- `images/` – legacy static EDA images
+- `artifacts/` – generated outputs after running script
 
 ## Setup
-1. Ensure Python is installed.
-2. Install required packages.
-3. Run the analysis: `python diabetes_analysis.py`
-
-Dataset: Pima Indians Diabetes Dataset (Kaggle)
+```bash
+python -m venv .venv
+source .venv/bin/activate
+pip install -r requirements.txt
+```
+
+## Run
+```bash
+python diabetes_analysis.py --data diabetes.csv --output-dir artifacts
+```
+
+## Outputs generated
+The script creates:
+- `artifacts/cv_summary.csv`
+- `artifacts/holdout_metrics.csv`
+- `artifacts/classification_report.txt`
+- `artifacts/confusion_matrix.png`
+- `artifacts/roc_curve.png`
+- `artifacts/outcome_count.png`
+- `artifacts/correlation_heatmap.png`
+- `artifacts/best_model.joblib`
+
+## Model Evaluation Notes
+- Uses stratified train/test split to preserve class ratio.
+- Tunes classification threshold from training probabilities with F2 focus.
+- Emphasizes reducing false negatives through recall-oriented thresholding.
+
+## Future Work
+- Add probability calibration.
+- Add SHAP-based explainability.
+- Add CI workflow for linting/tests.
+- Add fairness checks by demographic groups if available.
