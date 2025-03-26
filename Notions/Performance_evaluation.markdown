# Performance evaluation

## **MSE (Mean Squared Error)**

MSE measures the average squared difference between the predicted values and the actual values.

- **Formula:**  
  MSE = (1/n) * Σ(yᵢ - ŷᵢ)²  
  where:  
  - n is the number of data points  
  - yᵢ is the actual value  
  - ŷᵢ is the predicted value  

- **Purpose:**  
  A lower MSE indicates that the model's predictions are closer to the actual values.  

---

## **RMSE (Root Mean Squared Error)**  

RMSE is the square root of the MSE and provides an error metric in the same units as the target variable.

- **Formula:**  
  RMSE = √MSE = √[(1/n) * Σ(yᵢ - ŷᵢ)²]  
  where:  
  - n is the number of data points  
  - yᵢ is the actual value  
  - ŷᵢ is the predicted value  

- **Purpose:**  
  RMSE is often preferred because it is easier to interpret in the context of the original data.  

---

## **Cross-Validation**

### **What is Cross-Validation?**

Cross-validation is a technique used to evaluate the performance of a machine learning model by splitting the dataset into multiple parts (called "folds"). It ensures that the model is tested on data it hasn't seen during training, which helps to check how well the model generalizes to unseen data.

---

### **How Does Cross-Validation Work?**

The most common type is **k-fold cross-validation**, which works as follows:

1. **Split the Dataset:**  
   - Divide the dataset into `k` equal-sized parts (called "folds"). For example, if `k=5`, the dataset is split into 5 parts.

2. **Train and Test the Model:**  
   - For each fold:
     - Use `k-1` folds as the training set.
     - Use the remaining fold as the test set.
   - Train the model on the training set and evaluate it on the test set.

3. **Repeat for All Folds:**  
   - Repeat the process `k` times, with each fold being used as the test set exactly once.

4. **Calculate the Average Performance:**  
   - Compute the performance metric (e.g., RMSE) for each fold.
   - Take the average of these metrics to get the final evaluation score.

---

### **Why Use Cross-Validation?**

- **Avoid Overfitting:** Ensures the model is not just memorizing the training data but generalizing well to unseen data.
- **Reliable Evaluation:** Provides a more robust estimate of model performance compared to a single train-test split.
- **Efficient Use of Data:** All data points are used for both training and testing, maximizing the use of the dataset.

---

### **How to Implement Cross-Validation in Python**

Here’s how you can implement **k-fold cross-validation** using `cross_val_score` from `sklearn`:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

# Define the model
model = # Your model

# Define the scoring metric (negative MSE because cross_val_score minimizes by default)
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring=scorer)

# Convert negative MSE to positive RMSE
rmse_scores = np.sqrt(-cv_scores)

# Print the results
print(f"Cross-Validation RMSE Scores: {rmse_scores}")
print(f"Mean RMSE: {rmse_scores.mean():.2f}")
print(f"Standard Deviation of RMSE: {rmse_scores.std():.2f}")



