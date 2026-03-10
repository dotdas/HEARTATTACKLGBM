import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

SEED = 50303776

BASE_DIR = Path(__file__).parent

INGESTED_DIR = BASE_DIR / "ingested"
INPUT_FILE = INGESTED_DIR / "Heart Attack Data Set.csv"

TRAIN_FILE = BASE_DIR / "train.csv"
TEST_FILE = BASE_DIR / "test.csv"

def preprocess():
    df = pd.read_csv(INPUT_FILE)
    df = df.drop_duplicates()
    print(f"Data loaded from {INPUT_FILE}")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print("Data split into train and test sets.")

    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    print("Converted mismatched data types to missing values in train and test sets.")

    train_medians = X_train.median()

    X_train.fillna(train_medians, inplace=True)
    X_test.fillna(train_medians, inplace=True)
    print("Filled missing values with medians from the training set.")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    print("Fitted scaler on training data and applied scaling.")

    X_test_scaled = scaler.transform(X_test)
    print("Applied scaling to test data.")

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    train_df = pd.concat([X_train_scaled_df, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled_df, y_test.reset_index(drop=True)], axis=1)

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    joblib.dump(train_medians, BASE_DIR / "medians.pkl")
    joblib.dump(scaler, BASE_DIR / "scaler.pkl")
    print("Preprocessing artifacts (medians, scaler) saved for deployment.")

    print("Processed data saved:")
    print(f"Train set: {TRAIN_FILE} (Shape: {train_df.shape})")
    print(f"Test set: {TEST_FILE} (Shape: {test_df.shape})")

if __name__ == "__main__":
    preprocess()