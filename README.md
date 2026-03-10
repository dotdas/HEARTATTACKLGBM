# Identity

- Frederick Allensius - 2802405030
- Winston Octavianus Sugih - 2802403776

# Heart Attack Prediction MLOps Pipeline

This project builds a complete machine learning pipeline to predict the likelihood of a heart attack based on patient clinical data. It demonstrates MLOps principles by automating data ingestion, preprocessing, model training, evaluation, and registration using **MLflow**.

The final model is an **LightGBM Classifier** that is trained and evaluated automatically. If it meets a predefined accuracy threshold, it is approved for deployment.

## Project Structure

```
heart-attack-prediction/
├── data_ingestion.py          # Script to load the raw dataset.
├── preprocessing.py           # Script to clean, scale, and split data.
├── train.py                   # Script to train the LGBM model and log it to MLflow.
├── evaluation.py              # Script to evaluate the trained model.
├── pipeline.py                # Main orchestrator script that runs the entire pipeline.
├── Exploration.ipynb          # Jupyter notebook for extensive data analysis and model comparison.
├── requirements.txt           # List of Python dependencies.
├── README.md                  # This file.
└── Heart Attack Data Set.csv  # The dataset being worked on  
```

## Setup and Installation

Follow these steps to set up the project environment on your local machine.

1.  **Install Dependencies**
    The required packages are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the MLflow Tracking Server (Optional but Recommended)**
    To visualize experiments, metrics, and models, run the MLflow UI in a separate terminal:
    ```bash
    mlflow ui
    ```

## Running the Pipeline

To execute the entire end-to-end pipeline, simply run the main script:

```bash
python pipeline.py
```

This command will sequentially execute the following steps:
1.  **Data Ingestion**: Loads the `Heart Attack Data Set.csv` file.
2.  **Preprocessing**: Cleans the data, handles missing values, scales features, and splits it into training and testing sets.
3.  **Training**: Trains a LightGBM model on the processed data, logging parameters and artifacts to MLflow.
4.  **Evaluation**: Evaluates the model on the test set. If accuracy is **≥ 80%**, the model is approved; otherwise, it is rejected.

## Model Details

The selected model is an **LightGBM Classifier**, chosen after a comparative analysis with other algorithms (Logistic Regression, Random Forest, LightGBM) as documented in `Exploration.ipynb`.

- **Algorithm**: `LGBMClassifier`
- **MLflow Experiment Name**: `Heart-Attack-Prediction`
- **Registered Model Name**: `HeartAttackLGBMModel`
- **Best Hyperparameters**:
    ```python
    {
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
    ```
- **Performance**: The model is automatically evaluated. The pipeline aims for an accuracy of at least 80% on the hold-out test set.


## ⚙️ Key Configuration

- **Random Seed**: `50303776` (used across all scripts for reproducibility)
- **Accuracy Threshold**: `0.8` (minimum accuracy for model deployment approval)
- **Test Size**: `0.2` (20% of data held out for testing)

## 📝 Requirements

```
pandas
numpy
scikit-learn
xgboost
mlflow
pathlib
optuna
lightgbm
matplotlib
seaborn
jupyter
```
