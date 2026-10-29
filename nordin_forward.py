from time import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from imblearn.over_sampling import SMOTENC

diabetes = pd.read_csv(Path(__file__).resolve().with_name("diabetes_dataset.csv"))
target_column = "Diabetes_012"
samples_per_class = 4500
class_counts = diabetes[target_column].value_counts()
if class_counts.min() < samples_per_class:
    raise ValueError(
        f"Cannot sample {samples_per_class} rows per class; "
        f"available counts are {class_counts.sort_index().to_dict()}."
    )
diabetes = (
    diabetes.groupby(target_column, group_keys=False)
    .sample(n=samples_per_class, random_state=42)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
print(
    "Balanced dataset class counts:",
    diabetes[target_column].value_counts().sort_index().to_dict(),
)

X = diabetes.drop(columns=[target_column])
y = diabetes[target_column]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
)

# SMOTENC handles mixed data: binary indicators and ordinal ratings are
# categorical, while continuous measurements are interpolated numerically.
ordinal_columns = [
    column for column in ("Age", "GenHlth", "Education", "Income")
    if column in X_train.columns
]
binary_columns = [column for column in X_train.columns if X_train[column].nunique() == 2]
categorical_columns = list(dict.fromkeys(binary_columns + ordinal_columns))
# Do not log-transform ordinal codes; keep their ordered category values intact.
log_columns = X_train.columns[(X_train.nunique() > 2) & ~X_train.columns.isin(ordinal_columns)]
X_train_log = X_train.copy()
X_val_log = X_val.copy()
X_test_log = X_test.copy()
X_train_log[log_columns] = np.log1p(X_train[log_columns])
X_val_log[log_columns] = np.log1p(X_val[log_columns])
X_test_log[log_columns] = np.log1p(X_test[log_columns])

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_log)
X_val_scaled = scaler.transform(X_val_log)
X_test_scaled = scaler.transform(X_test_log)

# Scale before oversampling so neighbor distances are not dominated by
# features with larger numeric ranges.
class_counts = y_train.value_counts()
if class_counts.nunique() > 1:
    if not categorical_columns:
        raise ValueError("SMOTENC requires at least one categorical feature.")
    if class_counts.min() < 2:
        raise ValueError("SMOTENC requires at least two training examples in every class.")
    smote = SMOTENC(
        categorical_features=[X_train.columns.get_loc(column) for column in categorical_columns],
        categorical_encoder=OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        random_state=42,
        k_neighbors=min(5, int(class_counts.min()) - 1),
    )
    X_train_scaled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    y_train = pd.Series(y_resampled, name=y_train.name)
else:
    print("Skipping SMOTENC because the training classes are already balanced.")

feature_names = np.array(X.columns)

model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_val_scaled)
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

ridge = RidgeClassifierCV(alphas=np.logspace(-6, 6, num=5)).fit(X_train_scaled, y_train)
importance = np.mean(np.abs(ridge.coef_), axis=0)

plt.figure(figsize=(12, 6))
plt.bar(x=feature_names, height=importance)
plt.title("Feature importances via coefficients")
plt.xticks(rotation=90)
plt.tight_layout()
#plt.show()

threshold = np.sort(importance)[-min(3, len(importance))]

tic = time()
sfm = SelectFromModel(ridge, threshold=threshold, prefit=True)
print(f"Features selected by SelectFromModel: {feature_names[sfm.get_support()]}")
print(f"Done in {time() - tic:.3f}s")

sample_idx = np.random.RandomState(42).choice(
    len(X_train_scaled), size=min(10000, len(X_train_scaled)), replace=False
)
X_train_sample = X_train_scaled[sample_idx]
y_train_sample = y_train.iloc[sample_idx]

tic_fwd = time()
sfs_forward = SequentialFeatureSelector(
    ridge,
    n_features_to_select=min(5, X_train_scaled.shape[1]),
    direction="forward",
    scoring="f1_macro",
    cv=3,
    n_jobs=-1,
).fit(X_train_sample, y_train_sample)
print(
    f"Features selected by forward sequential selection: {feature_names[sfs_forward.get_support()]}"
)
print(f"Done in {time() - tic_fwd:.3f}s")



# Sammenlign KNN og RandomForest på forskellige feature-sæt


# get_support() giver en liste med True/False for hver feature.
# np.ones(..., dtype=bool) er en liste med lutter True = "brug alle".
feature_sets = {
    "Alle features": np.ones(len(feature_names), dtype=bool),
    "SelectFromModel": sfm.get_support(),
    "Forward (SFS)": sfs_forward.get_support(),
}


def make_knn():
    return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)


def make_forest():
    # RandomForest = mange beslutningstræer, der stemmer om svaret.
    # class_weight="balanced" giver fejl på de små klasser (1 og 2) mere vægt,
    # så modellen ikke bare gætter "rask" hele tiden.
    # min_samples_leaf=20 forhindrer træerne i at lære enkeltpersoner udenad.
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


# Hver kombination: (navn på model, funktion der laver modellen, feature-sæt)
experiments = [
    ("KNN", make_knn, "Alle features"),
    ("KNN", make_knn, "SelectFromModel"),
    ("KNN", make_knn, "Forward (SFS)"),
    ("RandomForest", make_forest, "Alle features"),
    ("RandomForest", make_forest, "SelectFromModel"),
    ("RandomForest", make_forest, "Forward (SFS)"),
]

print("\nSammenligning på valideringsdata")
print(f"{'Model':<14}{'Feature-sæt':<18}{'Antal':>6}{'Accuracy':>10}{'F1 (macro)':>12}{'Tid (s)':>9}")

results = []
for model_name, make_model, set_name in experiments:
    mask = feature_sets[set_name]
    tic = time()

    # X_train_scaled[:, mask] betyder: alle rækker, men kun de kolonner hvor mask er True
    clf = make_model()
    clf.fit(X_train_scaled[:, mask], y_train)
    y_pred = clf.predict(X_val_scaled[:, mask])

    # F1 (macro): gennemsnit af F1 for klasse 0, 1 og 2, så de små klasser tæller lige meget.
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    results.append((f1, model_name, set_name, clf))
    print(f"{model_name:<14}{set_name:<18}{mask.sum():>6}{acc:>10.3f}{f1:>12.3f}{time() - tic:>9.1f}")

# Vælg kombinationen med højest F1 på valideringsdata
best_f1, best_model_name, best_set, final_model = max(results, key=lambda r: r[0])
best_mask = feature_sets[best_set]
print(f"\nBedste kombination: {best_model_name} med {best_set}")
print(f"Features: {feature_names[best_mask]}")

# Vis valideringsrapporten for vinderen, så du kan se recall for hver klasse
print(classification_report(y_val, final_model.predict(X_val_scaled[:, best_mask])))

# One-vs-rest ROC curves and AUC values for each class.
y_val_scores = final_model.predict_proba(X_val_scaled[:, best_mask])
plt.figure(figsize=(8, 6))
for class_index, class_label in enumerate(final_model.classes_):
    y_val_binary = (y_val == class_label).astype(int)
    fpr, tpr, _ = roc_curve(y_val_binary, y_val_scores[:, class_index])
    class_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {class_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-rest ROC curves (validation data)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



RUN_TEST = False  # Sæt til True til allersidst !! DO NOT TOUCH BUDDY !!

if RUN_TEST:
    y_test_pred = final_model.predict(X_test_scaled[:, best_mask])
    print(f"\nResultat på testdata ({best_model_name}, {best_set})")
    print(confusion_matrix(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))