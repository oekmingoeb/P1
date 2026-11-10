from __future__ import annotations

from time import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def make_knn():
    return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)


def make_forest():
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def compute_feature_selection(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    feature_names: np.ndarray,
):
    ridge = RidgeClassifierCV(alphas=np.logspace(-6, 6, num=5)).fit(
        X_train_scaled, y_train
    )
    importance = np.mean(np.abs(ridge.coef_), axis=0)
    threshold = np.sort(importance)[-min(3, len(importance))]

    sfm = SelectFromModel(ridge, threshold=threshold, prefit=True)
    sample_idx = np.random.RandomState(42).choice(
        len(X_train_scaled), size=min(10000, len(X_train_scaled)), replace=False
    )
    X_train_sample = X_train_scaled[sample_idx]
    y_train_sample = y_train.iloc[sample_idx]

    sfs_forward = SequentialFeatureSelector(
        ridge,
        n_features_to_select=min(5, X_train_scaled.shape[1]),
        direction="forward",
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    ).fit(X_train_sample, y_train_sample)

    feature_sets = {
        "Alle features": np.ones(len(feature_names), dtype=bool),
        "SelectFromModel": sfm.get_support(),
        "Forward (SFS)": sfs_forward.get_support(),
    }

    print(f"Features selected by SelectFromModel: {feature_names[sfm.get_support()]}")
    print(
        f"Features selected by forward sequential selection: "
        f"{feature_names[sfs_forward.get_support()]}"
    )
    return ridge, importance, feature_sets


def build_hyperparameter_searches():
    search_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    return [
        (
            "LogisticRegression GridSearch",
            GridSearchCV(
                LogisticRegression(solver="lbfgs", max_iter=1000),
                {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "class_weight": [None, "balanced"],
                },
                scoring="f1_micro",
                cv=search_cv,
                n_jobs=-1,
            ),
        ),
        (
            "LogisticRegression RandomSearch",
            RandomizedSearchCV(
                LogisticRegression(solver="lbfgs", max_iter=1000),
                {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "class_weight": [None, "balanced"],
                },
                n_iter=8,
                scoring="f1_micro",
                cv=search_cv,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "KNN GridSearch",
            GridSearchCV(
                KNeighborsClassifier(n_jobs=1),
                {
                    "n_neighbors": [3, 5, 9, 15],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                },
                scoring="f1_micro",
                cv=search_cv,
                n_jobs=-1,
            ),
        ),
        (
            "KNN RandomSearch",
            RandomizedSearchCV(
                KNeighborsClassifier(n_jobs=1),
                {
                    "n_neighbors": [3, 5, 7, 9, 11, 15, 19, 21],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                },
                n_iter=8,
                scoring="f1_micro",
                cv=search_cv,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "RandomForest GridSearch",
            GridSearchCV(
                RandomForestClassifier(
                    class_weight="balanced", n_jobs=1, random_state=42
                ),
                {
                    "n_estimators": [100, 150, 200, 250, 300],
                    "max_depth": [None, 15, 20, 30, 40, 50],
                    "min_samples_leaf": [10, 20, 30, 40, 50],
                },
                scoring="f1_micro",
                cv=search_cv,
                n_jobs=-1,
            ),
        ),
        (
            "RandomForest RandomSearch",
            RandomizedSearchCV(
                RandomForestClassifier(
                    class_weight="balanced", n_jobs=1, random_state=42
                ),
                {
                    "n_estimators": [100, 150, 200, 250, 300],
                    "max_depth": [None, 15, 20, 30, 40, 50],
                    "min_samples_leaf": [10, 20, 30, 40, 50],
                },
                n_iter=8,
                scoring="f1_micro",
                cv=search_cv,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]


def run_hyperparameter_searches(X_train_scaled: np.ndarray, y_train: pd.Series):
    print("\nHyperparameter search (3-fold CV on training data)")
    searches = build_hyperparameter_searches()
    for search_name, search in searches:
        tic = time()
        search.fit(X_train_scaled, y_train)
        print(
            f"{search_name}: CV F1 (macro) = {search.best_score_:.3f}; "
            f"best parameters = {search.best_params_}; "
            f"done in {time() - tic:.1f}s"
        )
    return searches


def evaluate_experiments(
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    y_train: pd.Series,
    y_val: pd.Series,
    feature_sets: dict,
    searches: list,
    feature_names: np.ndarray,
):
    experiments = [
        ("KNN", make_knn, "Alle features"),
        ("KNN", make_knn, "SelectFromModel"),
        ("KNN", make_knn, "Forward (SFS)"),
        ("RandomForest", make_forest, "Alle features"),
        ("RandomForest", make_forest, "SelectFromModel"),
        ("RandomForest", make_forest, "Forward (SFS)"),
    ]

    print("\nSammenligning på valideringsdata")
    print(
        f"{'Model':<14}{'Feature-sæt':<18}{'Antal':>6}{'Accuracy':>10}{'F1 (macro)':>12}{'Tid (s)':>9}"
    )

    results = []
    for model_name, make_model, set_name in experiments:
        mask = feature_sets[set_name]
        tic = time()

        clf = make_model()
        clf.fit(X_train_scaled[:, mask], y_train)
        y_pred = clf.predict(X_val_scaled[:, mask])

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        results.append((f1, model_name, set_name, clf))
        print(
            f"{model_name:<14}{set_name:<18}{mask.sum():>6}{acc:>10.3f}{f1:>12.3f}{time() - tic:>9.1f}"
        )

    for search_name, search in searches:
        y_pred = search.best_estimator_.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        results.append((f1, search_name, "Alle features", search.best_estimator_))
        print(
            f"{search_name:<24}{'Alle features':<18}{len(feature_names):>6}"
            f"{acc:>10.3f}{f1:>12.3f}{'CV tuned':>9}"
        )

    best_f1, best_model_name, best_set, final_model = max(results, key=lambda r: r[0])
    best_mask = feature_sets[best_set]
    print(f"\nBedste kombination: {best_model_name} med {best_set}")
    print(f"Features: {feature_names[best_mask]}")
    return results, best_model_name, best_set, best_mask, final_model
