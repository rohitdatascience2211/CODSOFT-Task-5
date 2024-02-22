import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv(r"C:\Users\prade\Downloads\creditcard.csv")

# Convert numeric columns to appropriate data type
numeric_columns = data.columns[data.dtypes != 'object'].tolist()
data[numeric_columns] = data[numeric_columns].astype(float)

# Preprocessing
X = data.drop('Class', axis=1)
y = data['Class']

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using oversampling
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Alternatively, you can use undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Evaluate the model's performance
print("Size of test set:", len(X_test))
if len(X_test) > 0:
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
else:
    print("Test set is empty.")
