import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Example dataset
data = {
    'YearsExperience': [1, 2, 3, 4, 5],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Salary': [40000, 50000, 60000, 70000, 80000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame to check its structure
print("DataFrame before encoding:")
print(df)

# Encode the Gender column
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2})  # Example encoding

# Print the DataFrame after encoding
print("DataFrame after encoding:")
print(df)

# Features and target variable
X = df[['YearsExperience', 'Gender']]
y = df['Salary']

# Print the features to check their structure
print("Features (X) before splitting:")
print(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train your model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the scaler and model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'salary_model.pkl')

print("Model and scaler saved successfully.")
