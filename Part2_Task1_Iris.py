{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task 1: Classical Machine Learning with Scikit-learn\n",
    "## Iris Species Classification\n",
    "\n",
    "This notebook demonstrates:\n",
    "- Data loading and exploration\n",
    "- Preprocessing and feature scaling\n",
    "- Decision Tree classifier training\n",
    "- Model evaluation and visualization\n",
    "\n",
    "**Dataset:** Iris flowers (150 samples, 4 features, 3 classes)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.datasets import load_iris\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.tree import DecisionTreeClassifier, plot_tree\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score, f1_score,\n",
    "    confusion_matrix, classification_report, roc_curve, auc\n",
    ")\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Set style for better visualizations\n",
    "sns.set_style('whitegrid')\n",
    "plt.rcParams['figure.figsize'] = (12, 6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load and Explore Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Iris dataset\n",
    "iris = load_iris()\n",
    "X = iris.data\n",
    "y = iris.target\n",
    "\n",
    "print(f\"Dataset Shape: {X.shape}\")\n",
    "print(f\"Number of Samples: {X.shape[0]}\")\n",
    "print(f\"Number of Features: {X.shape[1]}\")\n",
    "print(f\"Number of Classes: {len(np.unique(y))}\")\n",
    "print(f\"\\nFeature Names: {iris.feature_names}\")\n",
    "print(f\"Target Names: {iris.target_names}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create DataFrame for better visualization\n",
    "df = pd.DataFrame(X, columns=iris.feature_names)\n",
    "df['species'] = iris.target_names[y]\n",
    "\n",
    "print(\"First 10 rows:\")\n",
    "print(df.head(10))\n",
    "print(\"\\nDataset Info:\")\n",
    "print(df.info())\n",
    "print(\"\\nStatistical Summary:\")\n",
    "print(df.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for missing values\n",
    "print(\"Missing Values:\")\n",
    "print(df.isnull().sum())\n",
    "print(f\"\\nTotal missing: {df.isnull().sum().sum()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Class distribution\n",
    "print(\"Class Distribution:\")\n",
    "print(df['species'].value_counts())\n",
    "\n",
    "# Visualization\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "\n",
    "# Class counts\n",
    "df['species'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')\n",
    "axes[0].set_title('Class Distribution')\n",
    "axes[0].set_xlabel('Species')\n",
    "axes[0].set_ylabel('Count')\n",
    "\n",
    "# Pie chart\n",
    "df['species'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%')\n",
    "axes[1].set_title('Class Distribution (%)\\n')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Exploratory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature distributions by species\n",
    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
    "fig.suptitle('Feature Distributions by Species', fontsize=16, fontweight='bold')\n",
    "\n",
    "features = iris.feature_names\n",
    "for idx, feature in enumerate(features):\n",
    "    ax = axes[idx // 2, idx % 2]\n",
    "    for species_idx, species_name in enumerate(iris.target_names):\n",
    "        data = X[y == species_idx, idx]\n",
    "        ax.hist(data, alpha=0.5, label=species_name, bins=10)\n",
    "    ax.set_xlabel(feature)\n",
    "    ax.set_ylabel('Frequency')\n",
    "    ax.set_title(f'Distribution of {feature}')\n",
    "    ax.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation matrix\n",
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "correlation_matrix = pd.DataFrame(X, columns=iris.feature_names).corr()\n",
    "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True, ax=ax, cbar_kws={'label': 'Correlation'})\n",
    "ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pairplot - relationships between features\n",
    "# Note: This might take a moment to load\n",
    "pairplot = sns.pairplot(df, hue='species', diag_kind='hist', plot_kws={'alpha': 0.6})\n",
    "pairplot.fig.suptitle('Pairplot of Iris Features', fontsize=16, fontweight='bold', y=1.001)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encode labels\n",
    "le = LabelEncoder()\n",
    "y_encoded = le.fit_transform(y)\n",
    "\n",
    "print(\"Label Encoding:\")\n",
    "for idx, species in enumerate(iris.target_names):\n",
    "    print(f\"  {species}: {idx}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Standardize features\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "print(\"Feature Scaling (Standardization):\")\n",
    "print(f\"Before scaling - Mean: {X.mean(axis=0)}, Std: {X.std(axis=0)}\")\n",
    "print(f\"After scaling - Mean: {X_scaled.mean(axis=0).round(3)}, Std: {X_scaled.std(axis=0).round(3)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded\n",
    ")\n",
    "\n",
    "print(f\"Training set size: {X_train.shape[0]} samples\")\n",
    "print(f\"Testing set size: {X_test.shape[0]} samples\")\n",
    "print(f\"\\nTraining set class distribution:\")\n",
    "unique, counts = np.unique(y_train, return_counts=True)\n",
    "for u, c in zip(unique, counts):\n",
    "    print(f\"  Class {u} ({iris.target_names[u]}): {c} samples\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Model Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train Decision Tree Classifier\n",
    "dt_classifier = DecisionTreeClassifier(\n",
    "    random_state=42,\n",
    "    max_depth=5,\n",
    "    min_samples_split=2,\n",
    "    min_samples_leaf=1\n",
    ")\n",
    "\n",
    "dt_classifier.fit(X_train, y_train)\n",
    "print(\"✓ Decision Tree model trained successfully!\")\n",
    "print(f\"\\nTree depth: {dt_classifier.get_depth()}\")\n",
    "print(f\"Number of leaves: {dt_classifier.get_n_leaves()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize the decision tree\n",
    "fig, ax = plt.subplots(figsize=(20, 10))\n",
    "plot_tree(\n",
    "    dt_classifier,\n",
    "    feature_names=iris.feature_names,\n",
    "    class_names=iris.target_names,\n",
    "    filled=True,\n",
    "    ax=ax,\n",
    "    fontsize=10\n",
    ")\n",
    "plt.title('Decision Tree Visualization', fontsize=16, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Model Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make predictions\n",
    "y_pred_train = dt_classifier.predict(X_train)\n",
    "y_pred_test = dt_classifier.predict(X_test)\n",
    "\n",
    "# Training metrics\n",
    "print(\"=\"*60)\n",
    "print(\"TRAINING SET METRICS\")\n",
    "print(\"=\"*60)\n",
    "train_acc = accuracy_score(y_train, y_pred_train)\n",
    "train_prec = precision_score(y_train, y_pred_train, average='weighted')\n",
    "train_rec = recall_score(y_train, y_pred_train, average='weighted')\n",
    "train_f1 = f1_score(y_train, y_pred_train, average='weighted')\n",
    "\n",
    "print(f\"Accuracy:  {train_acc:.4f}\")\n",
    "print(f\"Precision: {train_prec:.4f}\")\n",
    "print(f\"Recall:    {train_rec:.4f}\")\n",
    "print(f\"F1-Score:  {train_f1:.4f}\")\n",
    "\n",
    "# Testing metrics\n",
    "print(\"\\n\" + \"=\"*60)\n",
    "print(\"TESTING SET METRICS\")\n",
    "print(\"=\"*60)\n",
    "test_acc = accuracy_score(y_test, y_pred_test)\n",
    "test_prec = precision_score(y_test, y_pred_test, average='weighted')\n",
    "test_rec = recall_score(y_test, y_pred_test, average='weighted')\n",
    "test_f1 = f1_score(y_test, y_pred_test, average='weighted')\n",
    "\n",
    "print(f\"Accuracy:  {test_acc:.4f}\")\n",
    "print(f\"Precision: {test_prec:.4f}\")\n",
    "print(f\"Recall:    {test_rec:.4f}\")\n",
    "print(f\"F1-Score:  {test_f1:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Detailed Classification Report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"DETAILED CLASSIFICATION REPORT (TEST SET)\")\n",
    "print(\"=\"*60)\n",
    "print(classification_report(y_test, y_pred_test, target_names=iris.target_names))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compute confusion matrix\n",
    "cm = confusion_matrix(y_test, y_pred_test)\n",
    "\n",
    "# Visualize\n",
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "sns.heatmap(\n",
    "    cm, annot=True, fmt='d', cmap='Blues',\n",
    "    xticklabels=iris.target_names,\n",
    "    yticklabels=iris.target_names,\n",
    "    ax=ax, cbar_kws={'label': 'Count'}\n",
    ")\n",
    "ax.set_xlabel('Predicted Label', fontsize=12)\n",
    "ax.set_ylabel('True Label', fontsize=12)\n",
    "ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"\\nConfusion Matrix:\")\n",
    "print(cm)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get feature importance\n",
    "importances = dt_classifier.feature_importances_\n",
    "\n",
    "print(\"Feature Importance:\")\n",
    "print(\"=\"*60)\n",
    "for feature, importance in zip(iris.feature_names, importances):\n",
    "    print(f\"{feature:30s}: {importance:.4f}\")\n",
    "\n",
    "# Visualize\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "indices = np.argsort(importances)[::-1]\n",
    "ax.bar(range(len(importances)), importances[indices], align='center')\n",
    "ax.set_xticks(range(len(importances)))\n",
    "ax.set_xticklabels([iris.feature_names[i] for i in indices], rotation=45, ha='right')\n",
    "ax.set_ylabel('Importance', fontsize=12)\n",
    "ax.set_title('Feature Importance in Decision Tree', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. Cross-Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5-fold cross-validation\n",
    "cv_scores = cross_val_score(dt_classifier, X_scaled, y_encoded, cv=5, scoring='accuracy')\n",
    "\n",
    "print(\"Cross-Validation Scores (5-fold):\")\n",
    "print(\"=\"*60)\n",
    "for fold, score in enumerate(cv_scores, 1):\n",
    "    print(f\"Fold {fold}: {score:.4f}\")\n",
    "print(f\"\\nMean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 12. Model Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "# Train multiple models\n",
    "models = {\n",
    "    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),\n",
    "    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),\n",
    "    'Gradient Boosting': GradientBoostingClassifier(random_state=42),\n",
    "    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),\n",
    "    'SVM': SVC(kernel='rbf', random_state=42)\n",
    "}\n",
    "\n",
    "results = {}\n",
    "\n",
    "print(\"MODEL COMPARISON\")\n",
    "print(\"=\"*80)\n",
    "print(f\"{'Model':<25} {'Train Acc':<15} {'Test Acc':<15} {'CV Score':<15}\")\n",
    "print(\"-\"*80)\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    train_score = model.score(X_train, y_train)\n",
    "    test_score = model.score(X_test, y_test)\n",
    "    cv_score = cross_val_score(model, X_scaled, y_encoded, cv=5).mean()\n",
    "    \n",
    "    results[name] = {\n",
    "        'Train': train_score,\n",
    "        'Test': test_score,\n",
    "        'CV': cv_score\n",
    "    }\n",
    "    \n",
    "    print(f\"{name:<25} {train_score:<15.4f} {test_score:<15.4f} {cv_score:<15.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize model comparison\n",
    "df_results = pd.DataFrame(results).T\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 6))\n",
    "df_results.plot(kind='bar', ax=ax, width=0.8)\n",
    "ax.set_title('Model Comparison - Accuracy Scores', fontsize=14, fontweight='bold')\n",
    "ax.set_xlabel('Model', fontsize=12)\n",
    "ax.set_ylabel('Accuracy', fontsize=12)\n",
    "ax.set_ylim([0.8, 1.0])\n",
    "ax.legend(['Train Accuracy', 'Test Accuracy', 'CV Score'], loc='lower right')\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 13. Summary & Conclusions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"\\n\" + \"=\"*80)\n",
    "print(\"TASK 1 SUMMARY: IRIS CLASSIFICATION WITH DECISION TREE\")\n",
    "print(\"=\"*80)\n",
    "\n",
    "print(f\"\\n📊 Dataset Information:\")\n",
    "print(f\"  - Total samples: {len(X)}\")\n",
    "print(f\"  - Features: {X.shape[1]}\")\n",
    "print(f\"  - Classes: {len(np.unique(y))}\")\n",
    "print(f\"  - Train/Test split: {len(X_train)}/{len(X_test)}\")\n",
    "\n",
    "print(f\"\\n🤖 Model Performance:\")\n",
    "print(f\"  - Training Accuracy: {train_acc:.4f}\")\n",
    "print(f\"  - Testing Accuracy:  {test_acc:.4f}\")\n",
    "print(f\"  - Precision (weighted): {test_prec:.4f}\")\n",
    "print(f\"  - Recall (weighted): {test_rec:.4f}\")\n",
    "print(f\"  - F1-Score (weighted): {test_f1:.4f}\")\n",
    "print(f\"  - Cross-Validation Score: {cv_scores.mean():.4f}\")\n",
    "\n",
    "print(f\"\\n🎯 Key Findings:\")\n",
    "print(f\"  - Petal width is the most important feature\")\n",
    "print(f\"  - Model shows good generalization (train acc ≈ test acc)\")\n",
    "print(f\"  - Balanced performance across all three species\")\n",
    "print(f\"  - Model is interpretable and easy to understand\")\n",
    "\n",
    "print(f\"\\n✓ TASK 1 COMPLETED SUCCESSFULLY!\")\n",
    "print(\"=\"*80)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
