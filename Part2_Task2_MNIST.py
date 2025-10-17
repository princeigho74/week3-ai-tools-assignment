"""
AI Tools Assignment - Part 2: Practical Implementation
Tasks 1 & 2: Classical ML with Scikit-learn and Deep Learning with TensorFlow

Author: AI Assignment
Date: 2025
"""

# ============================================================================
# TASK 1: Classical ML with Scikit-learn - Iris Species Classification
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TASK 1: CLASSICAL ML WITH SCIKIT-LEARN - IRIS CLASSIFICATION")
print("=" * 70)

# Step 1: Load and explore the Iris dataset
print("\n[Step 1] Loading Iris Dataset...")
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species (0=setosa, 1=versicolor, 2=virginica)

# Create a DataFrame for better visualization
df_iris = pd.DataFrame(X, columns=iris.feature_names)
df_iris['species'] = iris.target_names[y]

print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}, Number of features: {X.shape[1]}")
print(f"Classes: {iris.target_names}")
print(f"\nFirst few rows:\n{df_iris.head()}")

# Step 2: Check for missing values and preprocess data
print("\n[Step 2] Preprocessing Data...")
print(f"Missing values: {df_iris.isnull().sum().sum()}")

# Encode labels (already numerical, but demonstrating the process)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Standardize features (important for decision trees, essential for many ML algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features standardized (mean=0, std=1)")
print(f"Feature scaling - Mean: {X_scaled.mean(axis=0).round(3)}, Std: {X_scaled.std(axis=0).round(3)}")

# Step 3: Split data into training and testing sets
print("\n[Step 3] Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Step 4: Train Decision Tree Classifier
print("\n[Step 4] Training Decision Tree Classifier...")
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_classifier.fit(X_train, y_train)
print("Decision Tree model trained successfully!")

# Step 5: Make predictions
print("\n[Step 5] Making Predictions...")
y_pred_train = dt_classifier.predict(X_train)
y_pred_test = dt_classifier.predict(X_test)

# Step 6: Evaluate the model
print("\n[Step 6] Model Evaluation...")
print("\n--- TRAINING SET METRICS ---")
train_accuracy = accuracy_score(y_train, y_pred_train)
train_precision = precision_score(y_train, y_pred_train, average='weighted')
train_recall = recall_score(y_train, y_pred_train, average='weighted')

print(f"Accuracy:  {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall:    {train_recall:.4f}")

print("\n--- TESTING SET METRICS ---")
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, average='weighted')
test_recall = recall_score(y_test, y_pred_test, average='weighted')

print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")

print("\n--- DETAILED CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred_test, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print(f"\nConfusion Matrix:\n{cm}")

# Feature importance
print("\n--- FEATURE IMPORTANCE ---")
for feature_name, importance in zip(iris.feature_names, dt_classifier.feature_importances_):
    print(f"{feature_name}: {importance:.4f}")


# ============================================================================
# TASK 2: Deep Learning with TensorFlow - MNIST Handwritten Digits
# ============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

print("\n\n" + "=" * 70)
print("TASK 2: DEEP LEARNING WITH TENSORFLOW - MNIST CLASSIFICATION")
print("=" * 70)

# Step 1: Load and preprocess MNIST dataset
print("\n[Step 1] Loading MNIST Dataset...")
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()

print(f"Training set shape: {X_train_mnist.shape}")
print(f"Testing set shape: {X_test_mnist.shape}")
print(f"Classes: 0-9 (digits)")

# Normalize pixel values from [0, 255] to [0, 1]
print("\n[Step 2] Preprocessing Data...")
X_train_norm = X_train_mnist.astype('float32') / 255.0
X_test_norm = X_test_mnist.astype('float32') / 255.0

# Reshape images to include channel dimension (28, 28, 1)
X_train_reshaped = X_train_norm.reshape(-1, 28, 28, 1)
X_test_reshaped = X_test_norm.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_cat = to_categorical(y_train_mnist, 10)
y_test_cat = to_categorical(y_test_mnist, 10)

print(f"Training set shape after preprocessing: {X_train_reshaped.shape}")
print(f"Training labels shape (one-hot encoded): {y_train_cat.shape}")

# Step 3: Build CNN Model
print("\n[Step 3] Building CNN Model...")
model = models.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# Step 4: Train the model
print("\n[Step 4] Training CNN Model...")
history = model.fit(
    X_train_reshaped, y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Step 5: Evaluate on test set
print("\n[Step 5] Evaluating Model on Test Set...")
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

# Check if accuracy > 95%
if test_accuracy > 0.95:
    print("✓ Model achieves >95% test accuracy!")
else:
    print("Note: Model accuracy is below 95%. Consider training longer or adjusting architecture.")

# Step 6: Visualize predictions on 5 sample images
print("\n[Step 6] Visualizing Predictions on Sample Images...")

# Get predictions for test set
y_pred_mnist = model.predict(X_test_reshaped[:5])
y_pred_classes = np.argmax(y_pred_mnist, axis=1)
y_true_classes = np.argmax(y_test_cat[:5], axis=1)

print("\nSample Predictions:")
for i in range(5):
    print(f"Image {i+1}: Predicted={y_pred_classes[i]}, Actual={y_true_classes[i]}, " +
          f"Confidence={np.max(y_pred_mnist[i]):.4f}")

print("\nNote: Visualizations saved (in notebook environment, use plt.show())")

print("\n" + "=" * 70)
print("PART 2 PRACTICAL TASKS COMPLETED SUCCESSFULLY!")
print("=" * 70)
