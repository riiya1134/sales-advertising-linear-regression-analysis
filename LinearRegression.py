import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("sales.csv", sep=r"\s+")

print("Dataset preview:")
print(data.head())

print("\nColumn Names:", data.columns)

# -----------------------------
# 2. Correlation Analysis
# -----------------------------
print("\nCorrelation Matrix:")
print(data.corr())

# -----------------------------
# 3. Define Variables
# -----------------------------
X = data[['advertising']]
y = data['sales']

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 6. Predictions
# -----------------------------
y_pred = model.predict(X)
y_test_pred = model.predict(X_test)

# -----------------------------
# 7. Model Evaluation
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = model.score(X_test, y_test)

print("\nModel Evaluation (Test Data)")
print("RMSE:", rmse)
print("R² Score:", r2)

# -----------------------------
# 8. Regression Equation
# -----------------------------
print("\nRegression Equation:")
print(f"Sales = {model.coef_[0]:.3f} * Advertising + {model.intercept_:.3f}")

# -----------------------------
# 9. Visualization
# -----------------------------
plt.figure(figsize=(8,6))

plt.scatter(data['advertising'], data['sales'],
            color='blue', alpha=0.6, label='Actual Data')

plt.plot(data['advertising'], y_pred,
         color='red', label='Regression Line')

plt.title("Advertising vs Sales (Linear Regression)")
plt.xlabel("Advertising Budget")
plt.ylabel("Sales")

plt.legend()
plt.show()

# -----------------------------
# 10. Residual Analysis
# -----------------------------
residuals = y_test - y_test_pred

plt.figure(figsize=(8,6))
plt.scatter(y_test_pred, residuals)

plt.axhline(y=0, color='red', linestyle='--')

plt.title("Residual Plot")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")

plt.show()

# -----------------------------
# 11. Prediction Example
# -----------------------------
new_ad_budget = [[30]]
predicted_sales = model.predict(new_ad_budget)

print("\nExample Prediction")
print(f"Predicted Sales for Advertising = 30:", predicted_sales[0])