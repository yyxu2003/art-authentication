#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 22:02:40 2025

@author: yiyangxu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Load features
train_df = pd.read_csv('/Users/yiyangxu/vangogh_authenticator/src/features/train_features.csv')
test_df = pd.read_csv('/Users/yiyangxu/vangogh_authenticator/src/features/test_features.csv')

X_train, y_train = train_df.drop(columns=['label']), train_df['label']
X_test, y_test = test_df.drop(columns=['label']), test_df['label']

# Define feature groups based on original feature dimensions
feature_groups = [
    ('HOG', (0, 34595)),
    ('Color Histograms', (34596, 34691)),
    ('LBP Texture', (34692, 34717)),
    # ('Line Features', (34718, 34721))
]

results = []

for feature_name, (start_idx, end_idx) in feature_groups:
    print(f"\n=== Evaluating {feature_name} ===")
    
    # Create feature subset
    feature_cols = [f'feat_{i}' for i in range(start_idx, end_idx+1)]
    X_train_sub = X_train[feature_cols]
    X_test_sub = X_test[feature_cols]

    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True))
    ])

    # Hyperparameter grid
    param_grid = {
        'pca__n_components': [0.85, 0.90, 0.95],
        'svm__C': np.logspace(-3, 2, 6),
        'svm__gamma': ['scale', 'auto']
    }

    # Grid search with cross-validation
    grid = GridSearchCV(pipeline, param_grid, 
                       cv=StratifiedKFold(5), 
                       scoring='roc_auc',
                       n_jobs=-1,
                       verbose=1)
    grid.fit(X_train_sub, y_train)
    
    # Evaluate on test set
    best_model = grid.best_estimator_
    y_proba = best_model.predict_proba(X_test_sub)[:, 1]
    y_pred = best_model.predict(X_test_sub)
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'Feature Group': feature_name,
        'AUC': auc,
        'Accuracy': report['accuracy'],
        'Sensitivity': report['1']['recall'],  # Van Gogh recall
        'Specificity': report['0']['recall'],  # Non-Van Gogh recall
        'Best Parameters': grid.best_params_
    }
    
    results.append(metrics)
    
    print(f"Best parameters: {grid.best_params_}")
    print(f"AUC: {auc:.3f}, Accuracy: {metrics['Accuracy']:.3f}")
    print(classification_report(y_test, y_pred))

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot performance comparison
plt.figure(figsize=(12, 6))
metrics_to_plot = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
x = np.arange(len(metrics_to_plot))
bar_width = 0.2

for i, (_, row) in enumerate(results_df.iterrows()):
    plt.bar(x + i*bar_width, 
            row[metrics_to_plot], 
            width=bar_width, 
            label=row['Feature Group'])

plt.title('Feature Group Performance Comparison')
plt.xticks(x + bar_width*1.5, metrics_to_plot)
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# Display detailed results
print("\n=== Final Evaluation Results ===")
print(results_df[['Feature Group', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity']])
