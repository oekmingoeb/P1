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
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
    cross_val_score,
    KFold,
)
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import fbeta_score


def model_train_metrics(model_name):
    crossval = model_crossval_predictions[model_name]


    P = precision_score(y_train, crossval, average='macro')
    F1 = f1_score(y_train, crossval, average='macro')
    F2 = fbeta_score(y_train, crossval, beta=2, average='macro')
    R = recall_score(y_train, crossval, average='macro')

    print(f"{model_name} Precision:", P)
    print(f"{model_name} Recall:", R)
    print(f"{model_name} F1 Score:", F1)
    print(f"{model_name} F2 Score:", F2)
    print(
        f"\n{model_name} cross-validated training metrics by class:\n",
        classification_report(
            y_train,
            crossval,
            target_names=["No diabetes", "Diabetes"],
            zero_division=0,
        ),
    )


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
cols_to_scale = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']

X_train_p, X_test_p = X_train.copy(), X_test.copy()
X_train_p["BMI"] = np.log1p(X_train_p["BMI"])
X_test_p["BMI"] = np.log1p(X_test_p["BMI"])

X_train_p[cols_to_scale] = scaler.fit_transform(X_train_p[cols_to_scale])
X_test_p[cols_to_scale] = scaler.transform(X_test_p[cols_to_scale])

model = {
    "knn": KNeighborsClassifier(n_neighbors=7, weights='distance', p=1),
    "logreg": LogisticRegression(class_weight='balanced', max_iter=1000, C=0.001),
    "rf": RandomForestClassifier(n_estimators=250, min_samples_leaf=10, max_depth = 30, random_state=42)
}

model["knn"].fit(X_train_p, y_train)
model["logreg"].fit(X_train_p, y_train)
model["rf"].fit(X_train_p, y_train)

validation = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
model_thresholds = {"logreg": 0.425, "knn": 0.01, "rf": 0.125}
model_crossval_predictions = {}

# for model_name, classifier in model.items():
#     probabilities = cross_val_predict(
#         classifier,
#         X_train_p,
#         y_train,
#         cv=validation,
#         method="predict_proba",
#     )[:, 1]
#     model_crossval_predictions[model_name] = (
#         probabilities >= model_thresholds[model_name]
#     ).astype(int)
#     print(f"{model_name} threshold: {model_thresholds[model_name]:.3f}")

for model_name, classifier in model.items():
    probabilities = classifier.predict_proba(X_train_p)[:, 1]
    model_crossval_predictions[model_name] = (
        probabilities >= model_thresholds[model_name]
    ).astype(int)


#model_train_metrics(model_name="knn")
#model_train_metrics(model_name="logreg")
#model_train_metrics(model_name="rf")

test_model = "logreg"

test_probabilities = model[test_model].predict_proba(X_test_p)[:, 1]
test_predictions = (
    test_probabilities >= model_thresholds[test_model]
).astype(int)

test_precision = precision_score(
    y_test, test_predictions, average="macro", zero_division=0
)
test_recall = recall_score(
    y_test, test_predictions, average="macro", zero_division=0
)
test_f1 = f1_score(
    y_test, test_predictions, average="macro", zero_division=0
)
test_f2 = fbeta_score(
    y_test, test_predictions, beta=2, average="macro", zero_division=0
)

print(f"\n{test_model} test metrics:")
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1 Score:", test_f1)
print("F2 Score:", test_f2)
print(
    f"\n{test_model} test metrics by class:\n",
    classification_report(
        y_test,
        test_predictions,
        target_names=["No diabetes", "Diabetes"],
        zero_division=0,
    ),
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    display_labels=["No diabetes", "Diabetes"],
    cmap="Blues",
)
plt.title(f"{test_model} test confusion matrix")

false_positive_rate, true_positive_rate, _ = roc_curve(
    y_test, test_probabilities
)
test_auc = auc(false_positive_rate, true_positive_rate)
plt.figure(figsize=(9, 7))
plt.plot(
    false_positive_rate,
    true_positive_rate,
    label=f"{test_model} (AUC = {test_auc:.3f})",
)
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title(f"{test_model} test ROC curve")
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
