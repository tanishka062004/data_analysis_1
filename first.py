import pandas as pd
import numpy as np
# Load the Titanic dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Check for missing values
print(train_data.isnull().sum())

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

# Fill missing values in the 'Embarked' column with the mode
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

# Drop the 'Cabin' column as it has too many missing values
train_data = train_data.drop(columns=['Cabin'])
# Define a function to remove outliers based on IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Check for missing values again
print(train_data.isnull().sum())


# Remove outliers from 'Fare' and 'Age' columns
train_data = remove_outliers(train_data, 'Fare')
train_data = remove_outliers(train_data, 'Age')


# Check for outliers by describing the data
print(train_data.describe())

print(train_data)