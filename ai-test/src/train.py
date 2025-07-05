import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import pickle

# Paths
TRAIN_PATH = os.path.join(os.path.dirname(__file__), '../data/train.csv')
TEST_PATH = os.path.join(os.path.dirname(__file__), '../data/test.csv')
MODELS_PATH = os.path.join(os.path.dirname(__file__), '../models/')

# Create models directory if it doesn't exist
os.makedirs(MODELS_PATH, exist_ok=True)

# Load data
print("Loading processed data...")
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

# Prepare features and target
X_train = train_data.drop('churn', axis=1)
y_train = train_data['churn']
X_test = test_data.drop('churn', axis=1)
y_test = test_data['churn']

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Define models and their hyperparameter grids
models = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42),
        'params': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }
}

# Train and evaluate models
results = {}

for name, model_info in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}")
    print(f"{'='*50}")
    
    # Grid search with 5-fold CV
    grid_search = GridSearchCV(
        model_info['model'],
        model_info['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Test set predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'best_params': grid_search.best_params_,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'test_roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    results[name] = {
        'model': best_model,
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    # Print results
    print(f"\nBest parameters: {metrics['best_params']}")
    print(f"Cross-validation accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test precision: {metrics['test_precision']:.4f}")
    print(f"Test recall: {metrics['test_recall']:.4f}")
    print(f"Test F1-score: {metrics['test_f1']:.4f}")
    print(f"Test ROC AUC: {metrics['test_roc_auc']:.4f}")
    
    # Save model
    model_file = os.path.join(MODELS_PATH, f'{name.lower()}_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {model_file}")

# Compare models
print(f"\n{'='*50}")
print("MODEL COMPARISON")
print(f"{'='*50}")

comparison_df = pd.DataFrame({
    name: results[name]['metrics'] for name in results.keys()
}).T

print(comparison_df[['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']])

# Save results
results_file = os.path.join(MODELS_PATH, 'training_results.pkl')
with open(results_file, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved to {results_file}")

print("\nTraining completed!") 