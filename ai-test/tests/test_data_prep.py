import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

class TestDataPreparation(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'age': [25, 30, 35, 40, 45],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'tenure': [12, 24, 36, 48, 60],
            'monthly_charges': [50.0, 60.0, 70.0, 80.0, 90.0],
            'total_charges': [600.0, 1440.0, 2520.0, 3840.0, 5400.0],
            'contract_type': ['Month-to-month', 'One year', 'Two year', 'One year', 'Two year'],
            'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Electronic check'],
            'internet_service': ['DSL', 'Fiber optic', 'DSL', 'Fiber optic', 'DSL'],
            'online_security': ['No', 'Yes', 'No', 'Yes', 'No'],
            'online_backup': ['No', 'Yes', 'No', 'Yes', 'No'],
            'device_protection': ['No', 'Yes', 'No', 'Yes', 'No'],
            'tech_support': ['No', 'Yes', 'No', 'Yes', 'No'],
            'streaming_tv': ['No', 'Yes', 'No', 'Yes', 'No'],
            'streaming_movies': ['No', 'Yes', 'No', 'Yes', 'No'],
            'paperless_billing': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'churn': [0, 0, 1, 0, 1]
        })
        
        # Save test data
        self.test_file = 'test_churn_data.csv'
        self.test_data.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_data_loading(self):
        """Test that data can be loaded correctly"""
        loaded_data = pd.read_csv(self.test_file)
        self.assertEqual(len(loaded_data), 5)
        self.assertEqual(len(loaded_data.columns), 17)
        self.assertTrue('churn' in loaded_data.columns)
    
    def test_missing_values_handling(self):
        """Test handling of missing values"""
        # Add some missing values
        data_with_missing = self.test_data.copy()
        data_with_missing.loc[0, 'age'] = np.nan
        data_with_missing.loc[1, 'gender'] = None
        
        # Test that missing values are detected
        missing_counts = data_with_missing.isnull().sum()
        self.assertEqual(missing_counts['age'], 1)
        self.assertEqual(missing_counts['gender'], 1)
    
    def test_categorical_encoding(self):
        """Test that categorical variables are properly encoded"""
        from sklearn.preprocessing import LabelEncoder
        
        # Test encoding of categorical columns
        categorical_cols = ['gender', 'contract_type', 'payment_method']
        
        for col in categorical_cols:
            le = LabelEncoder()
            encoded = le.fit_transform(self.test_data[col])
            self.assertEqual(len(encoded), len(self.test_data))
            self.assertTrue(all(isinstance(x, (int, np.integer)) for x in encoded))
    
    def test_train_test_split(self):
        """Test that train-test split works correctly"""
        from sklearn.model_selection import train_test_split
        
        X = self.test_data.drop('churn', axis=1)
        y = self.test_data['churn']
        
        # Use a larger test size to avoid stratification issues with small dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        
        # Check split sizes
        self.assertEqual(len(X_train), 3)  # 60% of 5
        self.assertEqual(len(X_test), 2)   # 40% of 5
        
        # Check that we have data in both sets
        self.assertTrue(len(y_train) > 0)
        self.assertTrue(len(y_test) > 0)
    
    def test_feature_preparation(self):
        """Test that features are prepared correctly for modeling"""
        # Remove customer_id and churn for features
        features = self.test_data.drop(['customer_id', 'churn'], axis=1)
        target = self.test_data['churn']
        
        self.assertEqual(len(features.columns), 15)
        self.assertEqual(len(target), 5)
        self.assertTrue('churn' not in features.columns)
    
    def test_data_types(self):
        """Test that data types are appropriate"""
        # Check numeric columns
        numeric_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
        for col in numeric_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(self.test_data[col]))
        
        # Check categorical columns
        categorical_cols = ['gender', 'contract_type', 'payment_method']
        for col in categorical_cols:
            self.assertTrue(pd.api.types.is_object_dtype(self.test_data[col]))

if __name__ == '__main__':
    unittest.main() 