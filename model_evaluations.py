#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
"""
Created on Tue Mar 25 12:49:47 2025

@author: yiyangxu
"""

# Load features
features_path = "/Users/yiyangxu/vangogh_authenticator/src/features"
train_df = pd.read_csv(f"{features_path}/train_features.csv")
test_df = pd.read_csv(f"{features_path}/test_features.csv")

X_train, y_train = train_df.drop(columns=['label']), train_df['label']
X_test, y_test = test_df.drop(columns=['label']), test_df['label']

# Pipeline definition
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True))
])

# Hyperparameter tuning
param_grid = {
    'pca__n_components': [0.85, 0.90, 0.95],
    'svm__C': np.logspace(-3, 2, 6),
    'svm__gamma': ['scale', 'auto'],
}

grid = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(5), scoring='roc_auc', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")

# Predict probabilities and classes
y_pred_prob = grid.predict_proba(X_test)[:, 1]
y_pred = grid.predict(X_test)

# # ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# roc_auc = roc_auc_score(y_test, y_pred_prob)
# print(f"AUC (Area Under the ROC Curve): {roc_auc:.4f}")

# # Plot ROC
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity)')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend()
# plt.grid()
# plt.show()

# # Confusion Matrix & Specificity
# cm = confusion_matrix(y_test, y_pred)
# tn, fp, fn, tp = cm.ravel()
# specificity = tn / (tn + fp)
# print(f"Specificity (True Negative Rate): {specificity:.2%}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=['Not Van Gogh', 'Van Gogh']))
# # Plot Confusion Matrix

# labels = ['Not Van Gogh', 'Van Gogh']
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.show()

# # Scree plot for PCA (best parameter)
# optimal_pca = grid.best_estimator_.named_steps['pca']
# explained_variance = optimal_pca.explained_variance_ratio_.cumsum()

# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(1, len(explained_variance)+1), explained_variance, marker='o', linestyle='-', color='purple')
# plt.xlabel('Number of PCA Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Scree Plot of PCA Components')
# plt.axhline(y=grid.best_params_['pca__n_components'], color='red', linestyle='--', label='Optimal Components')
# plt.legend()
# plt.grid()
# plt.show()


# ---------------------------------------------
# Visualize 3 False Positives & 3 False Negatives
# ---------------------------------------------

# Load metadata to get image paths
metadata_path = "/Users/yiyangxu/vangogh_authenticator/data/processed/test_set.csv"
test_metadata_df = pd.read_csv(metadata_path)

# Combine predictions with metadata
results_df = test_metadata_df.copy()
results_df["true_label"] = y_test.values
results_df["predicted_label"] = y_pred

# Find false negatives and false positives
false_negs = results_df[(results_df["true_label"] == 1) & (results_df["predicted_label"] == 0)].head(3)
false_poss = results_df[(results_df["true_label"] == 0) & (results_df["predicted_label"] == 1)].head(3)

# Plot false negatives
if not false_negs.empty:
    plt.figure(figsize=(15, 5))
    for i, row in enumerate(false_negs.itertuples()):
        img = cv2.cvtColor(cv2.imread(row.image_path), cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.title("False Negative\nReal Van Gogh → Predicted Fake")
        plt.axis('off')
    plt.suptitle("Examples of Real Van Gogh Misclassified as Fake")
    plt.tight_layout()
    plt.show()
else:
    print("✅ No false negatives found.")

# Plot false positives
if not false_poss.empty:
    plt.figure(figsize=(15, 5))
    for i, row in enumerate(false_poss.itertuples()):
        img = cv2.cvtColor(cv2.imread(row.image_path), cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.title("False Positive\nNot Van Gogh → Predicted Real")
        plt.axis('off')
    plt.suptitle("Examples of Non-Van Gogh Misclassified as Real")
    plt.tight_layout()
    plt.show()
else:
    print("✅ No false positives found.")

