import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump
import sys
sys.path.append("../")
from Utils.common_function import common_Fucnation,train_and_evaluate_model,train_and_evaluate_multiple_models
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv('gym_members_exercise_tracking.csv')
comm_fun = common_Fucnation()

# Data Preprocessing
# Encode categorical variables
data = comm_fun.all_label_encode(data=df)

# Features and Target Selection
X = data.drop(columns=['Calories_Burned'])  # Assuming 'Calories_Burned' is the target
y = data['Calories_Burned']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training And Model Evaluation
y_pred, r2, model = train_and_evaluate_model(
    model=LinearRegression(),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

# Save the model
dump(model, 'LinearRegression_model.joblib')
