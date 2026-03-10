import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score # <-- Added roc_auc_score
from pathlib import Path

BASE_DIR = Path(__file__).parent

def evaluate(run_id):
    test_data = pd.read_csv(BASE_DIR / "test.csv")

    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-model")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc) 

    print(f"Evaluation completed for run {run_id}:")
    print(f"  - Accuracy = {accuracy:.3f}")
    print(f"  - ROC AUC  = {roc_auc:.3f}")

    return accuracy, roc_auc

if __name__ == "__main__":
    evaluate()