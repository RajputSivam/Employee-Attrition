import argparse
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


RANDOM_STATE = 42
DROP_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


def build_models():
    random_forest = RandomForestClassifier(random_state=RANDOM_STATE)
    xgboost = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": random_forest,
        "XGBoost": xgboost,
        "Hybrid (Stacked)": StackingClassifier(
            estimators=[("rf", random_forest), ("xgb", xgboost)],
            final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            cv=5,
            n_jobs=-1,
        ),
    }


def run_experiment(data_path: Path, output_path: Path) -> None:
    data = pd.read_csv(data_path)
    initial_rows = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    data = data.drop(columns=[column for column in DROP_COLUMNS if column in data.columns])

    target = data.pop("Attrition").map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("Attrition contains values other than 'No' and 'Yes'.")

    categorical_columns = data.select_dtypes(include=["str"]).columns.tolist()
    numeric_columns = data.columns.difference(categorical_columns).tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_columns),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        target,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_smote, y_train_smote = SMOTE(random_state=RANDOM_STATE).fit_resample(x_train, y_train)
    results = []
    confusion_matrices = []
    for condition, features, labels in [
        ("Before SMOTE", x_train, y_train),
        ("After SMOTE", x_train_smote, y_train_smote),
    ]:
        for model_name, model in build_models().items():
            model.fit(features, labels)
            predicted = model.predict(x_test)
            results.append(
                {
                    "Condition": condition,
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, predicted),
                    "Precision": precision_score(y_test, predicted, zero_division=0),
                    "Recall": recall_score(y_test, predicted, zero_division=0),
                    "F1-score": f1_score(y_test, predicted, zero_division=0),
                }
            )
            tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
            confusion_matrices.append(
                {"Condition": condition, "Model": model_name, "TN": tn, "FP": fp, "FN": fn, "TP": tp}
            )

    metrics = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)

    print(f"Rows loaded: {initial_rows}; exact duplicates removed: {initial_rows - len(data)}")
    print(f"Full class counts: {target.value_counts().sort_index().to_dict()} (0=No, 1=Yes)")
    print(f"Train/test rows: {len(y_train)}/{len(y_test)}")
    print(f"Training counts before SMOTE: {y_train.value_counts().sort_index().to_dict()} (0=No, 1=Yes)")
    print(f"Training counts after SMOTE:  {pd.Series(y_train_smote).value_counts().sort_index().to_dict()} (0=No, 1=Yes)")
    print("\nMetrics:")
    formatters = {column: "{:.3f}".format for column in ["Accuracy", "Precision", "Recall", "F1-score"]}
    print(metrics.to_string(index=False, formatters=formatters))
    print("\nConfusion matrices (TN, FP, FN, TP):")
    print(pd.DataFrame(confusion_matrices).to_string(index=False))
    print(f"\nSaved metrics: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IBM HR employee attrition experiments.")
    parser.add_argument("data", type=Path, help="Path to the IBM attrition CSV file")
    parser.add_argument("--output", type=Path, default=Path("employee_attrition_results.csv"))
    arguments = parser.parse_args()
    run_experiment(arguments.data, arguments.output)