from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import RidgeCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

diabetes = pd.read_csv("diabetes_dataset.csv")

X = diabetes.drop(columns=["Diabetes_012"])
y = diabetes["Diabetes_012"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

feature_names = np.array(X.columns)

model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_val_scaled)
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X_train_scaled, y_train)
importance = np.abs(ridge.coef_)

plt.figure(figsize=(12, 6))
plt.bar(x=feature_names, height=importance)
plt.title("Feature importances via coefficients")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

threshold = np.sort(importance)[-3] + 0.01

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
    ridge, n_features_to_select=2, direction="forward", cv=3, n_jobs=-1
).fit(X_train_sample, y_train_sample)
print(
    f"Features selected by forward sequential selection: {feature_names[sfs_forward.get_support()]}"
)
print(f"Done in {time() - tic_fwd:.3f}s")
