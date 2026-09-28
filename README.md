# IBM HR Employee Attrition Experiment

This project compares Logistic Regression, Random Forest, XGBoost, and a stacked ensemble on the IBM HR Analytics Employee Attrition dataset, before and after applying SMOTE to the training data.

## Requirements

Python 3.10 or newer. Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Dataset

Download the IBM HR Analytics Employee Attrition dataset from Kaggle and place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in a local folder. The raw CSV is not included in this repository.

## Run

From this directory, run:

```powershell
python employee_attrition_experiment.py "C:\path\to\WA_Fn-UseC_-HR-Employee-Attrition.csv" --output employee_attrition_results.csv
```

The experiment uses a stratified 80/20 split with random seed 42. SMOTE is applied to training data only; metrics are evaluated on the unchanged test set. The output CSV contains accuracy, precision, recall, and F1-score for each model and condition. The script also prints confusion matrices.

## Results

In the included run, Logistic Regression after SMOTE achieved the highest recall (0.787) and F1-score (0.525). XGBoost achieved the highest post-SMOTE accuracy (0.850). Results are from one fixed holdout split and should not be treated as cross-validation estimates.