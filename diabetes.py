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
model_thresholds = {"logreg": 0.45, "knn": 0.10, "rf": 0.125}
model_crossval_predictions = {}

for model_name, classifier in model.items():
    probabilities = cross_val_predict(
        classifier,
        X_train_p,
        y_train,
        cv=validation,
        method="predict_proba",
    )[:, 1]
    model_crossval_predictions[model_name] = (
        probabilities >= model_thresholds[model_name]
    ).astype(int)
    print(f"{model_name} threshold: {model_thresholds[model_name]:.3f}")

knn_crossval = model_crossval_predictions["knn"]
logreg_crossval = model_crossval_predictions["logreg"]
rf_crossval = model_crossval_predictions["rf"]

P_knn = precision_score(y_train, knn_crossval, average='macro')
F1_knn = f1_score(y_train, knn_crossval, average='macro')
F2_knn = fbeta_score(y_train, knn_crossval, beta=2, average='macro')
R_knn = recall_score(y_train, knn_crossval, average='macro')

print("KNN Precision:", P_knn)
print("KNN Recall:", R_knn)
print("KNN F1 Score:", F1_knn)
print("KNN F2 Score:", F2_knn)
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
F2_logreg = fbeta_score(y_train, logreg_crossval, beta=2, average='macro')
R_logreg = recall_score(y_train, logreg_crossval, average='macro')
test_probabilities = model["logreg"].predict_proba(X_test_p)[:, 1]
test_predictions = (test_probabilities >= model_thresholds["logreg"]).astype(int)

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
print("Logistic Regression F2 Score:", F2_logreg)
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
F2_rf = fbeta_score(y_train, rf_crossval, beta=2, average='macro')
R_rf = recall_score(y_train, rf_crossval, average='macro')


print("Random Forest Precision:", P_rf)
print("Random Forest Recall:", R_rf)
print("Random Forest F1 Score:", F1_rf)
print("Random Forest F2 Score:", F2_rf)
print(
    "\nRandom Forest cross-validated training metrics by class:\n",
    classification_report(
        y_train,
        rf_crossval,
        target_names=["No diabetes", "Diabetes"],
        zero_division=0,
    ),
)