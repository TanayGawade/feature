import os
os.environ['MPLBACKEND'] = 'Agg'  # Set backend before importing matplotlib

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, classification_report
import pickle

# Set style for better plots

plt.style.use('default')
sns.set_palette("husl")

# Paths
MODELS_PATH = os.path.join(os.path.dirname(__file__), '../models/')
TEST_PATH = os.path.join(os.path.dirname(__file__), '../data/test.csv')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '../results/')

# Create results directory
os.makedirs(RESULTS_PATH, exist_ok=True)

# Load test data
print("Loading test data...")
test_data = pd.read_csv(TEST_PATH)
X_test = test_data.drop('churn', axis=1)
y_test = test_data['churn']

# Load trained models and results
print("Loading trained models...")
with open(os.path.join(MODELS_PATH, 'training_results.pkl'), 'rb') as f:
    results = pickle.load(f)

# Create evaluation plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')

# 1. Confusion Matrices
for i, (name, result) in enumerate(results.items()):
    ax = axes[0, i]
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{name} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

# 2. ROC Curves
ax = axes[1, 0]
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    auc = result['metrics']['test_roc_auc']
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend()
ax.grid(True)

# 3. Feature Importance (for Random Forest)
ax = axes[1, 1]
rf_model = results['RandomForestClassifier']['model']
if hasattr(rf_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    ax.barh(range(len(feature_importance)), feature_importance['importance'])
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Feature Importance')
else:
    ax.text(0.5, 0.5, 'Feature importance not available', 
            ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Feature Importance')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'model_evaluation.png'), dpi=300, bbox_inches='tight')
print(f"Evaluation plots saved to {os.path.join(RESULTS_PATH, 'model_evaluation.png')}")
plt.close()  # Close the figure to free memory

# Print detailed classification reports
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORTS")
print("="*60)

for name, result in results.items():
    print(f"\n{name}:")
    print("-" * 40)
    print(classification_report(y_test, result['predictions']))
    print(f"ROC AUC: {result['metrics']['test_roc_auc']:.4f}")

# Model comparison summary
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

comparison_data = []
for name, result in results.items():
    comparison_data.append({
        'Model': name,
        'Accuracy': result['metrics']['test_accuracy'],
        'Precision': result['metrics']['test_precision'],
        'Recall': result['metrics']['test_recall'],
        'F1-Score': result['metrics']['test_f1'],
        'ROC AUC': result['metrics']['test_roc_auc']
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv(os.path.join(RESULTS_PATH, 'model_comparison.csv'), index=False)
print(f"\nComparison table saved to {os.path.join(RESULTS_PATH, 'model_comparison.csv')}")

# Model selection recommendation
print("\n" + "="*60)
print("MODEL SELECTION RECOMMENDATION")
print("="*60)

best_model = max(results.keys(), key=lambda x: results[x]['metrics']['test_f1'])
print(f"Recommended model: {best_model}")
print(f"Reason: Highest F1-score ({results[best_model]['metrics']['test_f1']:.4f})")

print("\nEvaluation completed!") 