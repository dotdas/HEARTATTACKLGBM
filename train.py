import pandas as pd
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from pathlib import Path

BASE_DIR = Path(__file__).parent
TRAIN_FILE = BASE_DIR / "train.csv"
EXPERIMENT_NAME = "Heart-Attack-Prediction"
MODEL_NAME = "HeartAttackLGBMModel"

best_params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "class_weight": None,
    "colsample_bytree": 0.7516967431077703,
    "learning_rate": 0.0319067187566505,
    "max_depth": 11,
    "min_child_samples": 5,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "n_estimators": 730,
    "num_leaves": 244,
    "reg_alpha": 0.008109218983074139,
    "reg_lambda": 0.030949457025478713,
    "subsample": 0.6604675555847034,
    "subsample_for_bin": 200000,
    "subsample_freq": 0,
    "importance_type": "split",
    "random_state": 50303776,
    "n_jobs": -1
}

def train():
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow experiment set to: '{EXPERIMENT_NAME}'")

    train_data = pd.read_csv(TRAIN_FILE)
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]
    print(f"Training data loaded from {TRAIN_FILE}")

    with mlflow.start_run() as run:
        model = LGBMClassifier(**best_params)
        model.fit(X_train, y_train)
        print("Model training complete.")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="lightgbm-model",
            registered_model_name=MODEL_NAME
        )
        print(f"Model logged and registered as '{MODEL_NAME}' in MLflow.")

        return run.info.run_id

if __name__ == "__main__":
    train()