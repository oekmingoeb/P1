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
#get the sum of the diabetes_012 column
print(diabetes["Diabetes_012"].value_counts())

# draw diabetes_012 column as a bar chart
diabetes["Diabetes_012"].value_counts().plot(kind='bar', title='Diabetes_012 Distribution', xlabel='Diabetes_012', ylabel='Count')

plt.show()
