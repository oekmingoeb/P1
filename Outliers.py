#BMI Outlier Analysis:

import pandas as pd

# Load dataset
df = pd.read_csv("diabetes_dataset.csv")

# Select BMI feature
Q1 = df["BMI"].quantile(0.25)
Q3 = df["BMI"].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[
    (df["BMI"] < lower_bound) |
    (df["BMI"] > upper_bound)
]

# Print results
print("BMI OUTLIER ANALYSIS")
print("--------------------")
print("Dataset size:", len(df))
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print("Number of outliers:", len(outliers))
print("Percentage of outliers:", 
      round(len(outliers) / len(df) * 100, 2), "%")

# Additional BMI statistics
print("\nBMI PERCENTILES")
print(df["BMI"].describe(
    percentiles=[0.95, 0.99, 0.999]
))


# Separate lower and upper outliers

lower_outliers = df[df["BMI"] < lower_bound]
upper_outliers = df[df["BMI"] > upper_bound]

print("\nOUTLIER BREAKDOWN")
print("-----------------")

print("Lower outliers:", len(lower_outliers))
print("Upper outliers:", len(upper_outliers))

print("Lower percentage:",
      round(len(lower_outliers) / len(df) * 100, 2), "%")

print("Upper percentage:",
      round(len(upper_outliers) / len(df) * 100, 2), "%")

print("\nHighest BMI values:")
print(df["BMI"].nlargest(10).values)
