from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_preparation import (
    balance_dataset,
    identify_feature_types,
    load_diabetes_dataset,
    oversample_minority_classes,
    remove_prediabetes,
    scale_features,
    split_dataset,
    transform_log_features,
)
from modeling import compute_feature_selection, evaluate_experiments, run_hyperparameter_searches
from reporting import (
    plot_best_tree,
    plot_feature_importances,
    plot_roc_curves,
    print_validation_summary,
)

RUN_TEST = False


def main():
    diabetes = load_diabetes_dataset()
    diabetes = remove_prediabetes(diabetes)

    X, y, X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(diabetes)
    categorical_columns, log_columns = identify_feature_types(X_train)
    X_train_log, X_val_log, X_test_log = transform_log_features(
        X_train, X_val, X_test, log_columns
    )
    X_train_scaled, X_val_scaled, X_test_scaled, _ = scale_features(
        X_train_log, X_val_log, X_test_log
    )

    feature_names = np.array(X.columns)
    ridge, importance, feature_sets = compute_feature_selection(
        X_train_scaled, y_train, feature_names
    )
    plot_feature_importances(feature_names, importance)

    searches = run_hyperparameter_searches(X_train_scaled, y_train)
    results, best_model_name, best_set, best_mask, final_model = evaluate_experiments(
        X_train_scaled,
        X_val_scaled,
        y_train,
        y_val,
        feature_sets,
        searches,
        feature_names,
    )
    print_validation_summary(final_model, X_val_scaled, y_val, best_mask)
    plot_roc_curves(final_model, X_val_scaled, y_val, best_mask)

    best_forest = max(
        (result for result in results if result[1].startswith("RandomForest")),
        key=lambda result: result[0],
        default=None,
    )
    if best_forest is not None:
        forest_mask = feature_sets[best_forest[2]]
        plot_best_tree(best_forest[3], feature_names, forest_mask)

    plt.show()

    if RUN_TEST:
        y_test_pred = final_model.predict(X_test_scaled[:, best_mask])
        print(f"\nResultat på testdata ({best_model_name}, {best_set})")
        from sklearn.metrics import classification_report, confusion_matrix

        print(confusion_matrix(y_test, y_test_pred))
        print(classification_report(y_test, y_test_pred))


if __name__ == "__main__":
    main()
