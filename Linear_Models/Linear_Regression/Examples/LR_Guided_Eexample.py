import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Generate synthetic dataset for a simple linear relationship.
# Here the true relationship is: y = 3*x + 4 with added Gaussian noise.
np.random.seed(42)  # Ensure reproducibility
X = 2 * np.random.rand(100, 1)  # 100 random points between 0 and 2
y = 3 * X + 4 + np.random.randn(100, 1)  # Linear function with noise

# Step 2: Visualize the generated data.
plt.scatter(X, y, color='blue')
plt.title("Generated Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Step 3: Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize the Linear Regression model.
model = LinearRegression()

# Step 5: Train the model on the training set.
model.fit(X_train, y_train)

# Step 6: Extract the learned parameters (slope and intercept).
slope = model.coef_[0][0]
intercept = model.intercept_[0]
print(f"Learned model parameters: slope = {slope:.2f}, intercept = {intercept:.2f}")

# Step 7: Use the trained model to make predictions on the test set.
y_pred = model.predict(X_test)

# Step 8: Evaluate the model using Mean Squared Error (MSE).
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Step 9: Visualize the regression line along with the test set data.
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title("Linear Regression: Test Data vs Predictions")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Explanation:
# 1. Data Generation: We create synthetic data based on y = 3*x + 4 plus noise.
# 2. Data Visualization: A scatter plot shows the underlying trend.
# 3. Data Splitting: The data is divided into training and test sets for evaluation.
# 4. Model Initialization: We create an instance of LinearRegression.
# 5. Model Training: The model learns the linear relationship from the training data.
# 6. Parameter Extraction: The learned slope and intercept are printed.
# 7. Prediction: The model predicts y-values for the test set.
# 8. Evaluation: Mean Squared Error quantifies the model's accuracy.
# 9. Results Plot: The regression line is displayed over the test data for visual confirmation.
