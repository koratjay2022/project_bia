import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from sklearn.model_selection import train_test_split
import re
import sys
sys.path.append("../")
from Utils.common_function import common_Fucnation,train_and_evaluate_model,train_and_evaluate_multiple_models

# Load the dataset
df = pd.read_csv('laptop_price.csv', encoding='unicode_escape')
comm_fun = common_Fucnation()

# Data preprocessing
# Fill missing values in 'TotalCharges' and convert to numeric
df['Company_encoded'] = comm_fun.label_encode(data=df,column="Company")

df['TypeName'].replace('Netbook', 'Notebook', inplace=True)
df['TypeName_encoded'] = comm_fun.label_encode(data=df,column="TypeName")

df = comm_fun.replace_values(data=df,column="OpSys",to_replace={'Windows 10 S': 'Windows 10', 'Mac OS X': 'macOS'},)
df = comm_fun.remove_rows(data=df,column="OpSys",value="Android")

df['Ram'] = df['Ram'].replace(r'GB', '', regex=True)
df['Ram'] = df['Ram'].astype(int)

df['ScreenResolution'] = df['ScreenResolution'].str.extract(r'(\d+x\d+)')

df['resolution'] = df['ScreenResolution'].apply(lambda x: re.search(r'\d+x\d+', x).group())
df[['resolution_width', 'resolution_height']] = df['resolution'].str.split('x', expand=True)
df['resolution_width'] = df['resolution_width'].astype(int)
df['resolution_height'] = df['resolution_height'].astype(int)

df['Cpu_Brand'] = df['Cpu'].str.split().str[0]
df['Cpu_Type'] = df['Cpu'].str.extract(r'(Core \w+\d|Atom|FX|Celeron|E-Series|Ryzen|A\d+-Series|Pentium|Xeon|Core M)')
df['Cpu_ClockSpeed'] = df['Cpu'].str.extract(r'(\d+\.\d+GHz|\d+GHz)')

# preprocessing all data added Label Encode
df = comm_fun.all_label_encode(data=df)

# preprocessing Remove Outliers
df = comm_fun.remove_outliers(data=df,column='Price_euros')

### model run
X = df.drop(columns=['Price_euros', 'Product','laptop_ID','Company','TypeName','ScreenResolution','resolution'])
y = df['Price_euros']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
y_pred, r2, model = train_and_evaluate_model(
    model=RandomForestRegressor(),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

# Save the trained model to a file
dump(model, 'random_forest_model.joblib')

# conda activate bia
# python churn_main.py