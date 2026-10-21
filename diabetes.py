import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline


diabetes = pd.read_csv("diabetes_dataset.csv")

num_features = diabetes.select_dtypes(include=[np.number]).columns


#print(diabetes.head)
#print(diabetes.info)
#print(diabetes.isnull().sum())


diabetes.hist(bins=50, figsize=(20,15))

standard_scaler = StandardScaler()
scaled_data = standard_scaler.fit_transform(diabetes[num_features])

scaled_df = pd.DataFrame(scaled_data, columns=diabetes[num_features].columns)


minmax_scaler = MinMaxScaler()
scaled_data_minmax = minmax_scaler.fit_transform(diabetes[num_features])
    
minmax_scaled_df = pd.DataFrame(scaled_data_minmax, columns=diabetes[num_features].columns)


scaled_df.hist(bins=50, figsize=(20,15))
minmax_scaled_df.hist(bins=50, figsize=(20,15))

#plt.show()

X = diabetes.drop(columns=["Diabetes_012"])
# Clean up code
y = diabetes["Diabetes_012"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_proba = model.predict_proba(X_val)[:, 1]
threshold = 0.5
y_pred_t = (y_proba >= threshold).astype(int)

cm = confusion_matrix(X_val, y_pred_t)

accuracy = accuracy_score(X_val, y_pred_t)

precision = precision_score(X_val, y_pred_t)

recall = recall_score(X_val, y_pred_t)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn+fp)

npv = tn / (tn+fn)

f1 = f1_score(X_val, y_pred_t)
print(accuracy, precision, recall, specificity, npv, f1)


fpr, tpr, thresholds = metrics.roc_curve(X_val, y_proba)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  name='example estimator')
display.plot()
plt.show()

