import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.append("../")
from Utils.common_function import common_functions,train_and_evaluate_model,train_and_evaluate_multiple_models
from sklearn.model_selection import train_test_split

common_Fun = common_functions()

# Load and preprocess the dataset
def load_data():
    data = pd.read_csv("customer_feedback_satisfaction.csv")
    data.drop(columns=['CustomerID'], inplace=True)
    data = common_Fun.all_label_encode(data=data)
    return data

# Load and prepare the data
data = load_data()


# Define feature columns (ensure this matches the columns used for training)
feature_columns = ['Gender', 'Country', 'ProductQuality', 'ServiceQuality', 'FeedbackScore', 'LoyaltyLevel', 'Age', 'Income', 'PurchaseFrequency']

# Splitting the data into features and target variable
X = data[feature_columns]
y = data['SatisfactionScore']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training And Model Evaluation
y_pred, r2, model = train_and_evaluate_model(
    model=RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=10, min_samples_split=8),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

# Save the model
dump(model, 'Random_Forest_model.joblib')
