import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ["polarity", "title", "text"]
data = pd.read_csv('DataSet/train.csv', header=None, names=column_names)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display basic statistics
print("\nBasic statistics:")
print(data.describe(include='all'))

# Check the unique values in the 'polarity' column
unique_polarities = data['polarity'].unique()
print("\nUnique values in 'polarity' column:", unique_polarities)

# Count the occurrences of each polarity (only values 1 and 2 should be present)
polarity_counts = data['polarity'].value_counts()
print("\nCount of polarities:")
print(polarity_counts)

# Visualize the polarity distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='polarity', palette='viridis')
plt.title('Distribution of Polarities')
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Negative (1)', 'Positive (2)'])
plt.show()

# Check for missing values in the dataset
missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Check for any duplicates in the dataset
duplicate_rows = data.duplicated().sum()
print("\nNumber of duplicate rows:", duplicate_rows)

# Analyze text length and check for null values in 'text' column
data['text_length'] = data['text'].apply(len)

# Visualize distribution of text lengths
plt.figure(figsize=(10, 6))
sns.histplot(data['text_length'], bins=30, kde=True, color='blue')
plt.title('Distribution of Text Lengths')
plt.xlabel('Length of Text')
plt.ylabel('Frequency')
plt.show()

# Correlation analysis (for numerical features)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='text_length', y='polarity', alpha=0.7)
plt.title('Scatter Plot of Text Length vs Polarity')
plt.xlabel('Length of Text')
plt.ylabel('Polarity')
plt.show()

# Analyze categorical feature 'title'
if 'title' in data.columns:
    plt.figure(figsize=(10, 6))
    title_counts = data['title'].value_counts().head(20)  # Get top 10 titles
    sns.barplot(x=title_counts.index, y=title_counts.values, palette='pastel')
    plt.title('Top 10 Titles Distribution')
    plt.xlabel('Title')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Boxplot for detecting outliers in text length by polarity
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='polarity', y='text_length', palette='Set2')
plt.title('Boxplot of Text Length by Polarity')
plt.xlabel('Polarity')
plt.ylabel('Text Length')
plt.show()

# Additional checks for invalid polarity values
invalid_polarities = data[~data['polarity'].isin([1, 2])]
print("\nRows with invalid polarity values:")
print(invalid_polarities)

# Summary of the findings
print("\nSummary of Findings:")
print(f"Total rows: {data.shape[0]}")
print(f"Unique polarities: {unique_polarities}")
print(f"Missing values: {missing_values.sum()} in total")
print(f"Duplicate rows: {duplicate_rows}")