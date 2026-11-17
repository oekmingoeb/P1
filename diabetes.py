import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV, cross_val_predict, train_test_split, cross_val_score, KFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler

diabetes = pd.read_csv("diabetes_dataset.csv")

num_features = diabetes.select_dtypes(include=[np.number]).columns


X = diabetes.drop(columns=["Diabetes_012"])
# Clean up code
y = diabetes["Diabetes_012"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Remove class 1 from the training set and convert class 2 to 1
X_train, y_train = X_train[y_train != 1], (y_train[y_train != 1] == 2).astype(int)
X_test, y_test = X_test[y_test != 1], (y_test[y_test != 1] == 2).astype(int)

scaler = RobustScaler()

# Tilpas scaleren på træningsdata og transformer dem
features_to_scale = X_train[['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']]


log_transformed = np.log1p(X_train[features_to_scale.columns])
X_test_log_transformed = np.log1p(X_test[features_to_scale.columns])

X_train_scaled = scaler.fit_transform(log_transformed)
X_test_scaled = scaler.transform(X_test_log_transformed)

model = {
    "knn": KNeighborsClassifier(n_neighbors=7, weights='distance', p=1),
    "logreg": LogisticRegression(class_weight='balanced', max_iter=1000, C=0.001),
    "rf": RandomForestClassifier(n_estimators=250, min_samples_leaf=10, max_depth = 30, random_state=42)
}

model["knn"].fit(X_train_scaled, y_train)
model["logreg"].fit(X_train_scaled, y_train)
model["rf"].fit(X_train_scaled, y_train)

validation = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

model_tune = TunedThresholdClassifierCV(model["logreg"], cv=validation, scoring="f1_macro")

knn_crossval = cross_val_predict(model["knn"], X_train_scaled, y_train, cv=validation)
logreg_crossval = cross_val_predict(model_tune, X_train_scaled, y_train, cv=validation)
rf_crossval = cross_val_predict(model["rf"], X_train_scaled, y_train, cv=validation)
model_tune.fit(X_train_scaled, y_train)
print("Tuned Logistic Regression threshold:", model_tune.best_threshold_)

logreg_probabilities = cross_val_predict(
    model["logreg"],
    X_train_scaled,
    y_train,
    cv=validation,
    method="predict_proba",
)[:, 1]
thresholds = [
    0.400,
    0.425,
    0.450,
    0.475,
    0.500,
    0.525,
    0.550,
    0.575,
    0.600,
]
print("\nLogistic Regression performance by threshold:")
print("Threshold  Precision  Recall  F1")
for threshold in thresholds:
    threshold_predictions = (logreg_probabilities >= threshold).astype(int)
    print(
        f"{threshold:9.3f}  "
        f"{precision_score(y_train, threshold_predictions, zero_division=0):9.3f}  "
        f"{recall_score(y_train, threshold_predictions, zero_division=0):6.3f}  "
        f"{f1_score(y_train, threshold_predictions, zero_division=0):.3f}"
    )

P_knn = precision_score(y_train, knn_crossval, average='macro')
F1_knn = f1_score(y_train, knn_crossval, average='macro')
R_knn = recall_score(y_train, knn_crossval, average='macro')

print("KNN Precision:", P_knn)
print("KNN Recall:", R_knn)
print("KNN F1 Score:", F1_knn)
print(
    "\nKNN cross-validated training metrics by class:\n",
    classification_report(
        y_train,
        knn_crossval,
        target_names=["No diabetes", "Diabetes"],
        zero_division=0,
    ),
)

P_logreg = precision_score(y_train, logreg_crossval, average='macro')
F1_logreg = f1_score(y_train, logreg_crossval, average='macro')
R_logreg = recall_score(y_train, logreg_crossval, average='macro')
test_predictions = model_tune.predict(X_test_scaled)

print(
    "\nLogistic Regression test metrics by class:\n",
    classification_report(
        y_test,
        test_predictions,
        target_names=["No diabetes", "Diabetes"],
        zero_division=0,
    ),
)
print("Logistic Regression Precision:", P_logreg)
print("Logistic Regression Recall:", R_logreg)
print("Logistic Regression F1 Score:", F1_logreg)
print(
    "\nLogistic Regression cross-validated training metrics by class:\n",
    classification_report(
        y_train,
        logreg_crossval,
        target_names=["No diabetes", "Diabetes"],
        zero_division=0,
    ),
)

P_rf = precision_score(y_train, rf_crossval, average='macro')
F1_rf = f1_score(y_train, rf_crossval, average='macro')
R_rf = recall_score(y_train, rf_crossval, average='macro')


print("Random Forest Precision:", P_rf)
print("Random Forest Recall:", R_rf)
print("Random Forest F1 Score:", F1_rf)
print(
    "\nRandom Forest cross-validated training metrics by class:\n",
    classification_report(
        y_train,
        rf_crossval,
        target_names=["No diabetes", "Diabetes"],
        zero_division=0,
    ),
)

# ConfusionMatrixDisplay.from_predictions(
#     y_test,
#     logreg_test_predictions,
#     display_labels=["No diabetes", "Diabetes"],
#     cmap="Blues",
# )
# plt.title("Logistic Regression test confusion matrix")

# plot_roc_curves(model, X_test_scaled, y_test)
plt.show()



def plot_roc_curves(models, X_test, y_test):
    classes = np.sort(y_test.unique())
    y_test_binarized = np.column_stack(
        [(y_test == class_value).astype(int) for class_value in classes]
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    for model_name, classifier in models.items():
        probabilities = classifier.predict_proba(X_test)
        all_fpr = np.unique(
            np.concatenate(
                [
                    roc_curve(y_test_binarized[:, index], probabilities[:, index])[0]
                    for index in range(len(classes))
                ]
            )
        )
        mean_tpr = np.zeros_like(all_fpr)
        for index in range(len(classes)):
            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_test_binarized[:, index], probabilities[:, index]
            )
            mean_tpr += np.interp(all_fpr, false_positive_rate, true_positive_rate)
        mean_tpr /= len(classes)
        macro_auc = auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr, linewidth=2, label=f"{model_name} (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set(
        title="Multiclass ROC curves",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.05),
    )
    ax.legend(title="Model")
    fig.tight_layout()