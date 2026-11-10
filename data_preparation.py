from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

TARGET_COLUMN = "Diabetes_012"
PREDIABETES_LABEL = 1


def load_diabetes_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    if dataset_path is None:
        dataset_path = Path(__file__).resolve().with_name("diabetes_dataset.csv")
    return pd.read_csv(dataset_path)


def remove_prediabetes(
    diabetes: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    prediabetes_label: int = PREDIABETES_LABEL,
) -> pd.DataFrame:
    filtered = diabetes.loc[diabetes[target_column] != prediabetes_label].copy()
    print(f"Removed {(diabetes[target_column] == prediabetes_label).sum()} prediabetes rows.")
    return filtered


def balance_dataset(
    diabetes: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    samples_per_class: int = 4500,
) -> pd.DataFrame:
    class_counts = diabetes[target_column].value_counts()
    if class_counts.min() < samples_per_class:
        raise ValueError(
            f"Cannot sample {samples_per_class} rows per class; "
            f"available counts are {class_counts.sort_index().to_dict()}."
        )

    balanced = (
        diabetes.groupby(target_column, group_keys=False)
        .sample(n=samples_per_class, random_state=42)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    print(
        "Balanced dataset class counts:",
        balanced[target_column].value_counts().sort_index().to_dict(),
    )
    return balanced


def split_dataset(diabetes: pd.DataFrame, target_column: str = TARGET_COLUMN):
    X = diabetes.drop(columns=[target_column])
    y = diabetes[target_column]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
    )
    return X, y, X_train, X_val, X_test, y_train, y_val, y_test


def identify_feature_types(X_train: pd.DataFrame):
    ordinal_columns = [
        column for column in ("Age", "GenHlth", "Education", "Income")
        if column in X_train.columns
    ]
    binary_columns = [
        column for column in X_train.columns if X_train[column].nunique() == 2
    ]
    categorical_columns = list(dict.fromkeys(binary_columns + ordinal_columns))
    log_columns = X_train.columns[
        (X_train.nunique() > 2) & ~X_train.columns.isin(ordinal_columns)
    ]
    return categorical_columns, list(log_columns)


def transform_log_features(X_train, X_val, X_test, log_columns):
    X_train_log = X_train.copy()
    X_val_log = X_val.copy()
    X_test_log = X_test.copy()
    X_train_log[log_columns] = np.log1p(X_train[log_columns])
    X_val_log[log_columns] = np.log1p(X_val[log_columns])
    X_test_log[log_columns] = np.log1p(X_test[log_columns])
    return X_train_log, X_val_log, X_test_log


def scale_features(X_train_log, X_val_log, X_test_log):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_log)
    X_val_scaled = scaler.transform(X_val_log)
    X_test_scaled = scaler.transform(X_test_log)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def oversample_minority_classes(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    X_train: pd.DataFrame,
    categorical_columns,
):
    class_counts = y_train.value_counts()
    if class_counts.nunique() > 1:
        if not categorical_columns:
            raise ValueError("SMOTENC requires at least one categorical feature.")
        if class_counts.min() < 2:
            raise ValueError(
                "SMOTENC requires at least two training examples in every class."
            )

        smote = SMOTENC(
            categorical_features=[
                X_train.columns.get_loc(column) for column in categorical_columns
            ],
            categorical_encoder=OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            random_state=42,
            k_neighbors=min(5, int(class_counts.min()) - 1),
        )
        X_train_scaled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
        y_train = pd.Series(y_resampled, name=y_train.name)
        print("SMOTENC applied to training data.")
    else:
        print("Skipping SMOTENC because the training classes are already balanced.")

    return X_train_scaled, y_train
