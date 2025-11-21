# %% Imports
import matplotlib.pyplot as plt
import seaborn as sns
from utils import DataLoader

# %% Load data
data_loader = DataLoader()
data_loader.load_dataset()
data = data_loader.data

# %% Show head
print(data.shape)
data.head()

# %% Show general statistics
data.info()

# %% Show histogram for all columns
columns = data.columns
for col in columns:
    print("col: ", col)
    data[col].hist()
    plt.show()

# %% Show preprocessed dataframe
data_loader.preprocess_data()
data_loader.data.head()

# %% 1. Target Balance
print("--- Target Distribution ---")
print(data['stroke'].value_counts(normalize=True))
sns.countplot(x='stroke', data=data)
plt.show()

# %% 2. Numerical Distributions by Target
features = ['age', 'avg_glucose_level', 'bmi']
for f in features:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=data, x=f, hue='stroke', common_norm=False)
    plt.title(f'Stroke vs No Stroke: {f}')
    plt.show()

# %% 3. Correlation
plt.figure(figsize=(8,6))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# %% Box Plots for Outlier Detection
features = ['avg_glucose_level', 'bmi']

plt.figure(figsize=(10, 5))
for i, col in enumerate(features):
    plt.subplot(1, 2, i+1)
    sns.boxplot(y=data[col])
    plt.title(f'{col} Outliers')

plt.tight_layout()
plt.show()