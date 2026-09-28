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

## Comparison with Published Results

The comparison below uses Logistic Regression before SMOTE from the included experiment (87.41%, rounded to 87.4%). The difference is calculated as experiment accuracy minus paper accuracy, in percentage points; a positive value means this experiment's accuracy is higher.

| Research paper | Method/result | Paper accuracy | Experiment accuracy | Difference (pp) | Higher accuracy |
| --- | --- | ---: | ---: | ---: | --- |
| Melon et al. (2026) | XGBoost + SHAP ensemble | 83.0% | 87.4% | +4.4 | Experiment |
| Li et al. (2023) | Transformer-based deep learning | 85.07% | 87.4% | +2.3 | Experiment |
| Habous et al. (2021) | Logistic Regression | 86.0% | 87.4% | +1.4 | Experiment |
| Nandal et al. (2024) | Stacking (best classical) | 89.9% | 87.4% | -2.5 | Paper |
| Nandal et al. (2024) | Feed-forward neural network (FNN) | 97.5% | 87.4% | -10.1 | Paper |
| Alsheref et al. (2022) | Automated ensemble framework | 98.8% | 87.4% | -11.4 | Paper |
| Konar et al. (2025) | Stacked model + Bayesian optimization | 98.8% | 87.4% | -11.4 | Paper |

The paper accuracies and abbreviated citations above were transcribed from the supplied comparison and study summaries; verify them against the original publications before citing them. These are reported results from different experimental setups and may use different splits, preprocessing, or evaluation protocols, so they are not a like-for-like benchmark.