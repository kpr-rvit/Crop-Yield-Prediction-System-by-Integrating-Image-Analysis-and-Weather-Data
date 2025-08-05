import numpy as np
import matplotlib.pyplot as plt

# Bar Graph Data: RMSE, MAE, R² for Integrated Model, Linear Regression, and SVM
categories = ["RMSE", "MAE", "R²"]
integrated_model_metrics = [0.57, 0.02, 0.87]  # Replace with actual values
linear_regression_metrics = [1.51, 0.37, 0.09]  # Linear Regression metrics: RMSE, MAE, R²
svm_metrics = [1.23, 0.12, 0.39]  # SVM metrics: RMSE, MAE, R²

x = np.arange(len(categories))  # the label locations
width = 0.25  # Adjusted width for fewer bars

# Create Bar Graph
plt.figure(figsize=(10, 6))
plt.bar(x - width, integrated_model_metrics, width, label="Integrated Model", color="blue")
plt.bar(x, linear_regression_metrics, width, label="Linear Regression", color="green")
plt.bar(x + width, svm_metrics, width, label="SVM", color="red")

plt.xticks(x, categories)
plt.ylabel("Metric Values")
plt.title("Model Comparison: RMSE, MAE, R²")
plt.legend()
plt.tight_layout()
plt.show()

# Line Graph Data: Accuracy values for each model
models = ["Linear Regression", "SVM", "CNN", "XGBoost", "Integrated Model"]
accuracy = [9, 39, 60, 83, 87]  # Replace with actual values

# Adjusted plot with correct y-axis range
plt.figure(figsize=(8, 6))
plt.plot(models, accuracy, marker="o", linestyle="-", color="blue")

# Add labels and title
plt.xlabel("Models", fontsize=12)
plt.ylabel("Model Fit Percentage", fontsize=12)
plt.title("Comparison of Model Performance", fontsize=14)
plt.ylim(0, 100)  # Adjust y-axis range to accommodate all values
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Annotate points with their accuracy values
for i, acc in enumerate(accuracy):
    plt.text(models[i], acc + 2, f"{acc}%", ha="center", fontsize=10)  # Shift text upward for clarity

# Show the plot
plt.tight_layout()
plt.show()
