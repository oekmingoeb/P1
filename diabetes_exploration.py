from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

TARGET_COLUMN = "Diabetes_012"

plt.rcParams.update(
    {
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
        "figure.titlesize": 15,
    }
)


def load_diabetes_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = Path(__file__).resolve().with_name("diabetes_dataset.csv")
    return pd.read_csv(path)


def remove_duplicate_rows(diabetes: pd.DataFrame) -> pd.DataFrame:
    print(diabetes.head)
    print(diabetes.info)

    features = diabetes.drop(columns=["Diabetes_012"]).copy()
    print("Dublicates", features.duplicated(keep=False).sum())

    duplicate_rows = diabetes[diabetes.duplicated(keep=False)]
    print("Number of unique duplicated rows:", len(duplicate_rows.drop_duplicates()))
    duplicate_counts = duplicate_rows.value_counts(dropna=False)
    if not duplicate_counts.empty:
        print("Unique duplicate with the most occurrences:")
        print(duplicate_counts.head(1).rename("occurrences").reset_index())

    return diabetes.drop_duplicates()


def build_scaled_feature_views(features: pd.DataFrame):
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

    return scaled_features, numeric_columns


def plot_feature_density_comparison(scaled_features: dict, numeric_columns):
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    fig.suptitle("Feature distributions (binary features are not log-transformed)")
    sample_size = min(len(next(iter(scaled_features.values()))), 5000)
    sample_table = next(iter(scaled_features.values()))
    sample_indices = np.random.default_rng(42).choice(
        len(sample_table), size=sample_size, replace=False
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
            y_values = density(x_values)
            color = feature_colors[feature]
            ax.fill_between(x_values, y_values, color=color, alpha=0.18)
            ax.plot(x_values, y_values, color=color, linewidth=1.2)
            peak_index = np.argmax(y_values)
            ax.annotate(
                feature,
                xy=(x_values[peak_index], y_values[peak_index]),
                xytext=(3, 3),
                textcoords="offset points",
                color=color,
                fontsize=16,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
            )
        ax.set(title=chart_title, xlabel="Transformed feature value", ylabel="Density")
        if chart_title == "Log + Standard scaling":
            ax.set_xlim(-4, 4)
        elif chart_title == "Log + Robust scaling":
            ax.set_xlim(-5, 5)

    fig.tight_layout(rect=(0, 0, 1, 0.93))


def plot_bmi_distribution(diabetes: pd.DataFrame):
    bmi_values = diabetes["BMI"].dropna()
    fig_bmi, bmi_axes = plt.subplots(1, 2, figsize=(12, 5))
    bmi_axes[0].hist(bmi_values, bins=30, color="steelblue", edgecolor="white")
    bmi_axes[0].set(title="BMI distribution", xlabel="BMI", ylabel="Frequency")
    bmi_axes[1].hist(np.log1p(bmi_values), bins=30, color="darkorange", edgecolor="white")
    bmi_axes[1].set(
        title="Log-transformed BMI distribution",
        xlabel="log1p(BMI)",
        ylabel="Frequency",
    )
    fig_bmi.tight_layout()


def plot_class_distribution(diabetes: pd.DataFrame):
    class_names = {0: "No diabetes", 1: "Prediabetes", 2: "Diabetes"}
    class_counts = diabetes[TARGET_COLUMN].dropna().value_counts().sort_index()
    labels = [class_names.get(class_value, str(class_value)) for class_value in class_counts.index]
    percentages = class_counts / class_counts.sum() * 100

    fig_class, ax_class = plt.subplots(figsize=(8, 6))
    bars = ax_class.bar(labels, class_counts, color=["steelblue", "darkorange", "firebrick"])
    ax_class.set(
        title="Diabetes class distribution",
        xlabel="Diabetes class",
        ylabel="Number of rows",
    )
    ax_class.set_ylim(0, class_counts.max() * 1.15)
    for bar, percentage in zip(bars, percentages):
        ax_class.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
        )
    fig_class.tight_layout()


def plot_knn_projection(diabetes: pd.DataFrame):
    knn_labels = diabetes[TARGET_COLUMN]
    plot_size = min(5000, len(diabetes))
    if plot_size < len(diabetes):
        plot_indices, _ = train_test_split(
            np.arange(len(diabetes)),
            train_size=plot_size,
            stratify=knn_labels,
            random_state=42,
        )
    else:
        plot_indices = np.arange(len(diabetes))

    knn_features = (
        diabetes.drop(columns=TARGET_COLUMN)
        .select_dtypes(include=[np.number])
        .iloc[plot_indices]
    )
    knn_labels = knn_labels.iloc[plot_indices]
    knn_features_imputed = SimpleImputer(strategy="median").fit_transform(knn_features)
    knn_features_scaled = StandardScaler().fit_transform(knn_features_imputed)
    knn_projection = PCA(n_components=2, random_state=42).fit_transform(knn_features_scaled)

    fig_knn, ax_knn = plt.subplots(figsize=(9, 7))
    class_names = {0: "No diabetes", 1: "Prediabetes", 2: "Diabetes"}
    for class_value in sorted(knn_labels.dropna().unique()):
        class_mask = knn_labels.to_numpy() == class_value
        ax_knn.scatter(
            knn_projection[class_mask, 0],
            knn_projection[class_mask, 1],
            label=class_names.get(class_value, str(class_value)),
            alpha=0.35,
            s=14,
        )
    ax_knn.set(
        title="Scaled feature overview for KNN (2D PCA projection)",
        xlabel="Principal component 1",
        ylabel="Principal component 2",
    )
    ax_knn.legend(title="Diabetes class")
    fig_knn.tight_layout()


def main():
    diabetes = load_diabetes_data()
    diabetes = remove_duplicate_rows(diabetes)

    #features = diabetes.drop(columns=TARGET_COLUMN)
    features = diabetes[["BMI", "Age"]]
    scaled_features, numeric_columns = build_scaled_feature_views(features)
    plot_feature_density_comparison(scaled_features, numeric_columns)
    plot_bmi_distribution(diabetes)
    plot_class_distribution(diabetes)
    plot_knn_projection(diabetes)
    plt.show()


if __name__ == "__main__":
    main()
