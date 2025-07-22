import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
 # Sample data creation
data = {
 'YearsExperience': [1, 2, 3, 4, 5],
'Salary': [40000, 50000, 60000, 80000, 100000]
 }
# Create a DataFrame
df = pd.DataFrame(data)
# Features and target variable
X = df[['YearsExperience']]
y = df['Salary']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(X_train)
# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")
   