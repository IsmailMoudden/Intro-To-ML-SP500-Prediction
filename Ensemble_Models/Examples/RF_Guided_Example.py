import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Step 1: Load the Iris dataset (a classic multi-class classification problem)
data = load_iris()
X = data.data      # Features: measurements of iris flowers
y = data.target    # Target: iris species (0, 1, 2)

# Step 2: Visualize the dataset using the first two features for simplicity.
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title("Iris Dataset: Scatter Plot")
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.show()

# Step 3: Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize the Random Forest Classifier.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 5: Train the model on the training data.
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set.
y_pred = model.predict(X_test)

# Step 7: Evaluate the model's performance.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Step 8: Analyze feature importance.
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort features by importance
print("\nFeature Importances:")
for i in indices:
    print(f"{data.feature_names[i]}: {importances[i]:.2f}")

# Visualize feature importances.
plt.figure(figsize=(8, 6))
sns.barplot(x=importances[indices], y=np.array(data.feature_names)[indices])
plt.title("Feature Importances in Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Explanation:
# 1. Data Loading: The Iris dataset is loaded, providing both features and target labels.
# 2. Data Visualization: A scatter plot of the first two features helps visualize class distribution.
# 3. Data Splitting: The dataset is divided into training and testing sets for model evaluation.
# 4. Model Initialization: A Random Forest Classifier is created with 100 trees.
# 5. Model Training: The classifier learns patterns from the training data.
# 6. Prediction: The trained model predicts labels on the test set.
# 7. Evaluation: Accuracy and a detailed classification report assess model performance.
# 8. Feature Importance: The contribution of each feature is calculated and visualized.
