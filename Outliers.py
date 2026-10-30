#BMI Outlier Analysis:

import pandas as pd
import numpy as np

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

#--------------------------------------------------------------------------


print(f"Datasæt indlæst med {df.shape[0]:,} rækker og {df.shape[1]} kolonner.\n")

# 2. Find og inspicér de rækker, hvor BMI er omkring 98 (f.eks. mellem 95 og 100)
outliers = df[(df['BMI'] >= 95) & (df['BMI'] <= 100)]

print("--- DE 6 PERSONER MED EKSTREM BMI ---")
if len(outliers) > 0:
    # Viser alle kolonner for de specifikke rækker
    print(outliers)
else:
    print("Ingen personer fundet i det specifikke interval. Tjek om kolonnenavnet 'BMI' er rigtigt staves med store/små bogstaver.")
print("\n-------------------------------------\n")

# 3. DATARENSNING (Vælg den metode, der passer bedst til dit projekt):

# METODE A: Sæt urealistisk høje BMI-værdier (f.eks. over 70) til NaN (anbefalet, så du beholder resten af personens data)
df_clean_nan = df.copy()
df_clean_nan.loc[df_clean_nan['BMI'] > 70, 'BMI'] = np.nan

# METODE B: Slet rækkerne helt fra datasættet
df_clean_dropped = df[df['BMI'] <= 70].copy()

print(f"Metode A (NaN): Antal rækker er stadig {df_clean_nan.shape[0]:,}, men urealistiske BMI-tal er nulstillet.")
print(f"Metode B (Slet): Antal rækker i det nye rensede datasæt: {df_clean_dropped.shape[0]:,}")

# 4. Gem det rensede datasæt til en ny fil (så du ikke overskriver dine rådata)
df_clean_nan.to_csv("diabetes_dataset_cleaned.csv", index=False)
print("\nDet rensede datasæt er gemt som 'diabetes_dataset_cleaned.csv'")


#---------------------------------------------------------------------------


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Indlæs datasættet
filnavn = "diabetes_dataset.csv"
df = pd.read_csv(filnavn)

# 2. Bevis 1: Frekvenstabel over de højeste BMI-værdier
# Her ser vi på alle observationer med BMI på 70 eller derover
bmi_top = df[df['BMI'] >= 70]
frekvens_tabel = bmi_top['BMI'].value_counts().sort_index(ascending=False)

print("--- BEVIS 1: FREKVENS AF EKSTREME BMI-VÆRDIER ---")
print("BMI-værdi | Antal personer")
print("--------------------------")
for bmi, count in frekvens_tabel.items():
    print(f"{int(bmi):<9} | {count} personer")
print("--------------------------")
print("Bemærk hvordan tallene stopper, springer over værdier (f.eks. 96, 97),")
print("og pludselig klumper sig sammen på præcis 95 og 98.\n")


# 3. Bevis 2: Visualisering af "halen" for rapporten
plt.figure(figsize=(10, 6))

# Vi plotter fordelingen af BMI mellem 60 og 100 for at zoome ind på fejlen
sns.histplot(data=df[(df['BMI'] >= 60) & (df['BMI'] <= 100)], 
             x='BMI', 
             binwidth=1, 
             kde=False, 
             color='royalblue')

plt.title('Identifikation af Datatekniske Outliers i BMI (Hale-fordeling)', fontsize=14, fontweight='bold')
plt.xlabel('BMI (Afrundet til hele tal)', fontsize=12)
plt.ylabel('Antal observationer (Logistisk skala anbefales hvis der er få)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight fejlene
plt.axvline(x=95, color='crimson', linestyle=':', linewidth=2, label='Kunstig klump (95)')
plt.axvline(x=98, color='crimson', linestyle='--', linewidth=2, label='Kunstig klump (98)')
plt.legend()

# Gem figuren til din rapport
plt.savefig('bmi_outlier_argumentation.png', dpi=300, bbox_inches='tight')
print("Figuren 'bmi_outlier_argumentation.png' er gemt til din rapport.")


# 4. Konklusion og foreslået rensningsgrænse
# Vi tæller hvor mange personer der ryger, hvis vi sætter grænsen ved BMI > 70
grænseværdi = 70
antal_slettet = len(df[df['BMI'] > grænseværdi])
procent_slettet = (antal_slettet / len(df)) * 100

print(f"\n--- ARGUMENTATION TIL RAPPORTEN ---")
print(f"Hvis rensningsgrænsen sættes ved BMI > {grænseværdi}:")
print(f"- Antal slettede observationer: {antal_slettet} ud af {len(df):,}")
print(f"- Det svarer til kun {procent_slettet:.4f}% af det samlede datasæt.")
print(f"Konklusion: Datatabet er minimalt, men vi fjerner effektivt de tekniske fejl.")

