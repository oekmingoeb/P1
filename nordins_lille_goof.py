import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn import metrics
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline

diabetes = pd.read_csv("diabetes_dataset.csv")

num_features = diabetes.select_dtypes(include=[np.number]).columns


#print(diabetes.head)
#print(diabetes.info)
# Compare feature distributions, applying log1p only to non-binary features.
features = diabetes.drop(columns="Diabetes_012").copy()
numeric_columns = features.select_dtypes(include=[np.number]).columns
log_columns = [column for column in numeric_columns if features[column].nunique() > 2]
log_features = features.copy()
log_features[log_columns] = np.log1p(features[log_columns])
scaled_features = {
	"Log non-binary features": log_features.copy(),
	"Log + Min-max scaling": log_features.copy(),
	"Log + Standard scaling": log_features.copy(),
	"Log + Robust scaling": log_features.copy(),
}
scaled_features["Log + Min-max scaling"][numeric_columns] = MinMaxScaler().fit_transform(
	log_features[numeric_columns]
)
scaled_features["Log + Standard scaling"][numeric_columns] = StandardScaler().fit_transform(
	log_features[numeric_columns]
)
scaled_features["Log + Robust scaling"][numeric_columns] = RobustScaler().fit_transform(
	log_features[numeric_columns]
)

fig, axes = plt.subplots(1, 4, figsize=(28, 7))
fig.suptitle("Feature distributions (binary features are not log-transformed)")
sample_size = min(len(features), 5000)
sample_indices = np.random.default_rng(42).choice(
	len(features), size=sample_size, replace=False
)
color_maps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
feature_colors = {
	feature: color_maps[index // 20](index % 20)
	for index, feature in enumerate(numeric_columns)
}
for ax, (chart_title, chart_features) in zip(axes, scaled_features.items()):
	for feature in numeric_columns:
		values = chart_features[feature].iloc[sample_indices].dropna().to_numpy()
		if len(np.unique(values)) < 2:
			continue
		density = gaussian_kde(values)
		x_values = np.linspace(values.min(), values.max(), 300)
		color = feature_colors[feature]
		y_values = density(x_values)
		ax.fill_between(x_values, y_values, color=color, alpha=0.18)
		ax.plot(x_values, y_values, color=color, linewidth=1.2)
		peak_index = np.argmax(y_values)
		ax.annotate(
			feature,
			xy=(x_values[peak_index], y_values[peak_index]),
			xytext=(3, 3),
			textcoords="offset points",
			color=color,
			fontsize=6,
			bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
		)
	ax.set(title=chart_title, xlabel="Transformed feature value", ylabel="Density")
	if chart_title == "Log + Standard scaling":
		ax.set_xlim(-4, 4)
	elif chart_title == "Log + Robust scaling":
		ax.set_xlim(-5, 5)

fig.tight_layout(rect=(0, 0, 1, 0.93))

plt.show()
