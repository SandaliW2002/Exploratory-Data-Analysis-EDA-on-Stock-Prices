import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '2) Stock Prices Data Set.csv'
df = pd.read_csv(file_path)

print("Dataset Head")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

numerical_cols = df.select_dtypes(include=np.number).columns

print("\nMean")
print(df[numerical_cols].mean())

print("\nMedian")
print(df[numerical_cols].median())

print("\nMode")
print(df[numerical_cols].mode().iloc[0])

print("\nStandard Deviation")
print(df[numerical_cols].std())

for col in numerical_cols:
    plt.figure(figsize=(6,4))
    df[col].hist(bins=20, edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

if len(numerical_cols) > 1:
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            plt.figure(figsize=(6,4))
            sns.scatterplot(x=numerical_cols[i], y=numerical_cols[j], data=df)
            plt.title(f'Scatter plot: {numerical_cols[i]} vs {numerical_cols[j]}')
            plt.show()

corr_matrix = df[numerical_cols].corr()
print("\nCorrelation Matrix")
print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

print("\nEDA Completed")

