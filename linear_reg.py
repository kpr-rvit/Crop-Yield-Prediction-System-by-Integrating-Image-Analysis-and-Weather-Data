import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import pickle

# Load the dataset
df = pd.read_csv("WEATHER_DATASET_FINAL.csv")  

# Define synthetic target variable (should be replaced with actual logic for real cases)
df['Target'] = np.where((df['Onboard Temperature'] > 0.52) & 
                        (df['Onboard Humidity'] < 0.12), 100, 75)

# Define features and target
X = df.drop('Target', axis=1)  # Features (X)
y = df['Target']  # Target variable (y)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled features back to DataFrame to retain feature names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled_df, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled_df)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-Squared (R² Score): {r2:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Save model (since Linear Regression model is not in JSON format, we'll use pickle)
with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully as linear_regression_model.pkl!")
