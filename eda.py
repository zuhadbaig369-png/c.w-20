import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Iris Dataset.csv')

data.head(5)
data.isnull().sum()
data.describe()

labels = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

for label in labels:
    print('Distribution of', label)
    sns.boxplot(x=data[label])
    plt.show()

numeric_data = data.select_dtypes(include=['number'])

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

labels = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

for label in labels:

    print('Distribution of', label)
    sns.histplot(data[label], kde=True)
    plt.show()

labels = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

for label in labels:
    print('Skewness of', label)
    print(data[label].skew())