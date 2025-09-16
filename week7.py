# First, ensure all necessary libraries are imported
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Task 1: Load and Explore the Dataset
print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("=" * 50)

# Load the dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris_df = pd.read_csv(url)

# Display the first few rows
print("First 5 rows of the dataset:")
print(iris_df.head())

# Explore the structure
print("\nDataset info:")
print(iris_df.info())

print("\nData types:")
print(iris_df.dtypes)

print("\nMissing values:")
print(iris_df.isnull().sum())

# The dataset is already clean, but here's how we would handle missing values if they existed
# iris_df = iris_df.dropna()  # to drop rows with missing values
# or
# iris_df = iris_df.fillna(iris_df.mean())  # to fill with mean values

print("\nDataset shape:", iris_df.shape)

# Task 2: Basic Data Analysis
print("\n\nTASK 2: BASIC DATA ANALYSIS")
print("=" * 50)

# Compute basic statistics for numerical columns
print("Basic statistics:")
print(iris_df.describe())

# Group by species and compute mean of sepal_length
species_group = iris_df.groupby('species')
print("\nMean sepal length by species:")
print(species_group['sepal_length'].mean())

# Additional grouping by species for petal_length
print("\nMean petal length by species:")
print(species_group['petal_length'].mean())

# Interesting findings
print("\nInteresting findings:")
print("- Setosa species has the shortest petals and sepals on average")
print("- Virginica has the longest petals and sepals")
print("- Versicolor is in between for all measurements")

# Task 3: Data Visualization
print("\n\nTASK 3: DATA VISUALIZATION")
print("=" * 50)

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis', fontsize=16, fontweight='bold')

# 1. Line chart (simulating trends over time by using index as time proxy)
# We'll sort by sepal_length to create a meaningful trend
sorted_df = iris_df.sort_values('sepal_length')
axes[0, 0].plot(sorted_df.index, sorted_df['sepal_length'], marker='o', markersize=3)
axes[0, 0].set_title('Trend of Sepal Length (Sorted)')
axes[0, 0].set_xlabel('Observation Index')
axes[0, 0].set_ylabel('Sepal Length (cm)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar chart showing average petal length by species
petal_means = species_group['petal_length'].mean()
axes[0, 1].bar(petal_means.index, petal_means.values, color=['skyblue', 'lightgreen', 'lightcoral'])
axes[0, 1].set_title('Average Petal Length by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Petal Length (cm)')
for i, v in enumerate(petal_means.values):
    axes[0, 1].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')

# 3. Histogram of sepal length distribution
axes[1, 0].hist(iris_df['sepal_length'], bins=15, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Distribution of Sepal Length')
axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(iris_df['sepal_length'].mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {iris_df["sepal_length"].mean():.2f}')
axes[1, 0].legend()

# 4. Scatter plot of sepal length vs petal length
species = iris_df['species'].unique()
colors = ['blue', 'green', 'red']
for i, spec in enumerate(species):
    subset = iris_df[iris_df['species'] == spec]
    axes[1, 1].scatter(subset['sepal_length'], subset['petal_length'], 
                       alpha=0.7, label=spec, color=colors[i])
axes[1, 1].set_title('Sepal Length vs Petal Length')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional Analysis (Bonus)
print("\n\nADDITIONAL ANALYSIS")
print("=" * 50)

# Correlation matrix heatmap
plt.figure(figsize=(8, 6))
numeric_df = iris_df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# Box plot to show distribution of features by species
plt.figure(figsize=(12, 8))
iris_melted = pd.melt(iris_df, id_vars="species", 
                      value_vars=["sepal_length", "sepal_width", "petal_length", "petal_width"])
sns.boxplot(x="variable", y="value", hue="species", data=iris_melted)
plt.title('Distribution of Measurements by Species')
plt.xlabel('Measurement Type')
plt.ylabel('Value (cm)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()