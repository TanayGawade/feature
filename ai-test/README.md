# Customer Churn Prediction AI Solution

## Overview
This project builds an end-to-end AI solution to predict customer churn using machine learning. It covers data preparation, model training, evaluation, and (optionally) deployment as a REST API.

## Project Structure
- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks for exploration and modeling
- `src/`: Python scripts for data processing, training, and evaluation
- `tests/`: Unit tests

## Setup Instructions
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
- Run the the file 'workflow.py'
- Scripts in `src/` can be run for modular processing and training.

## Approach & Design Decisions

### Data Preparation
- **Data Loading**: Loaded customer churn dataset with 17 features including demographic, service, and billing information
- **Missing Values**: No missing values found in the dataset
- **Categorical Encoding**: Used LabelEncoder for categorical variables (gender, contract_type, payment_method, etc.)
- **Train-Test Split**: 80/20 split with stratification to maintain class balance

### Model Selection
- **LogisticRegression**: Linear model for baseline performance and interpretability
- **RandomForestClassifier**: Ensemble method for capturing non-linear relationships
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, and ROC AUC

### Model Performance
Both models achieved perfect scores on the test set:
- **LogisticRegression**: 
  - Best parameters: C=0.1, penalty='l1', solver='liblinear'
  - Test Accuracy: 100%
  - ROC AUC: 1.000
- **RandomForestClassifier**:
  - Best parameters: max_depth=3, min_samples_leaf=1, min_samples_split=2, n_estimators=50
  - Test Accuracy: 100%
  - ROC AUC: 1.000

### Final Model Selection
**LogisticRegression** was selected as the final model due to:
- Simpler and more interpretable
- Faster prediction times
- Sufficient performance for the task

## Results & Evaluation

### Model Comparison
| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| LogisticRegression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| RandomForestClassifier | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Key Features
- **Tenure**: Length of customer relationship
- **Contract Type**: Month-to-month vs. longer contracts
- **Monthly Charges**: Billing amount
- **Internet Service**: Type of internet connection
- **Online Services**: Security, backup, protection, support

### API Deployment
- Flask REST API with endpoints for single and batch predictions
- Health check and model information endpoints
- Input validation and error handling

## Potential Improvements
- **Data**: Use larger, more diverse dataset for better generalization
- **Feature Engineering**: Create interaction features and polynomial terms
- **Model Selection**: Try more algorithms (XGBoost, Neural Networks)
- **Hyperparameter Tuning**: Use Bayesian optimization
- **Deployment**: Containerize with Docker, add CI/CD pipeline
- **Monitoring**: Add model performance monitoring and drift detection
- **Testing**: Expand unit tests and add integration tests 
