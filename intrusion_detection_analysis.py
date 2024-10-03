# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
train_data = pd.read_csv('UNSW_NB15_training-set.csv')
test_data = pd.read_csv('UNSW_NB15_testing-set.csv')

# Data Overview
def data_overview(data):
    print("Data Shape:", data.shape)
    print("Data Types:\n", data.dtypes)
    print("Missing Values:\n", data.isnull().sum())
    print("Summary Statistics:\n", data.describe())

# Overview of training data
print("Training Data Overview:")
data_overview(train_data)

# Data Cleaning
def preprocess_data(train_data, test_data):
    # Fill missing values
    train_data.fillna(train_data.mean(), inplace=True)
    test_data.fillna(test_data.mean(), inplace=True)
    
    # Normalize features
    features = train_data.columns.difference(['label'])  # Assuming 'label' is the target variable
    X_train = train_data[features]
    y_train = train_data['label']
    X_test = test_data[features]
    y_test = test_data['label']
    
    # Normalize using MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)

# Exploratory Data Analysis (EDA)
def plot_label_distribution(y_train):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_train)  # Change to x=y_train to avoid warning
    plt.title('Distribution of Attack Types')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('label_distribution.png')  # Save as PNG
    plt.show()

plot_label_distribution(y_train)

# Visualizing Feature Correlations
def plot_feature_correlations(data):
    plt.figure(figsize=(12, 10))
    corr = data.corr()
    sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.savefig('feature_correlation_matrix.png')  # Save as PNG
    plt.show()

plot_feature_correlations(train_data)

# Feature Importance Analysis
def feature_importance(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    feature_importances = pd.DataFrame(importances, index=train_data.columns.difference(['label']), columns=["Importance"]).sort_values("Importance", ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_importances.Importance, y=feature_importances.index)
    plt.title('Feature Importances')
    plt.savefig('feature_importances.png')  # Save as PNG
    plt.show()
    
    return feature_importances

# Save feature importances to a CSV file
feature_importances = feature_importance(X_train, y_train)
feature_importances.to_csv('feature_importances.csv', index=True)

print("Analysis complete. Visualizations generated and feature importances saved.")
