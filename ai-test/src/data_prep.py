import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/churn_data.csv')
TRAIN_PATH = os.path.join(os.path.dirname(__file__), '../data/train.csv')
TEST_PATH = os.path.join(os.path.dirname(__file__), '../data/test.csv')

# 1. Load data
df = pd.read_csv(DATA_PATH)
print('Data loaded. Shape:', df.shape)

# 2. Basic EDA
def basic_eda(df):
    print('\n--- Head ---')
    print(df.head())
    print('\n--- Info ---')
    print(df.info())
    print('\n--- Describe ---')
    print(df.describe(include='all'))
    print('\n--- Missing values ---')
    print(df.isnull().sum())
    print('\n--- Value counts for target ---')
    print(df['churn'].value_counts())

basic_eda(df)

# 3. Cleaning
# Impute missing values (numerical: mean, categorical: most frequent)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Impute numeric
if numeric_cols:
    imputer_num = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

# Impute categorical
if categorical_cols:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

# Encode categoricals
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 4. Train-test split
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save processed data
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
train.to_csv(TRAIN_PATH, index=False)
test.to_csv(TEST_PATH, index=False)
print(f'Processed train and test sets saved to {TRAIN_PATH} and {TEST_PATH}') 