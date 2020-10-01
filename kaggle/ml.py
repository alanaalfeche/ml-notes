'''Intro to ML'''
import panda as pd

# read data and store data in DF 
data = pd.read_csv(csv_file_path, index_col='Id')

# print a summary statistics of data
data.describe()

# prints the first couple of row of data
data.head()

# print list of columns in the dataset
data.columns

# drops missing values 
data.dropna(axis=0)

# selecting prediction target with dot-notation -- stored as Series
y = data.Target

# selecting features with a column list
features = ['f_1', 'f_2',...]
X = data[features]

'''Build ml models with scikit-learn (sklearn) for DF data

0) Split training data into training and validation data
    - the validation data measures the model's accuracy
    - once a model is selected, predict on testing data
1) Define the model
2) Fit model to make prediction
3) Predict the target
4) Evaluate the accuracy of model's prediction 
    - Mean Absolute Error: mae = |actual - predicted|
'''

# Split data
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0) #random_state is set for reproducibility

# Define model
from sklearn.tree import DecisionTreeRegressor

'''Tackle Overfitting and Underfitting with `max_leaf_nodes` parameter

Overfitting - model matches the training data almost perfectly that it poorly validates new data
Underfitting - model fails to capture important distinctions and patterns in data, and so it performs poorly even with training data
'''
model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)

# Fit
model.fit(train_X, train_y)

# Predict
val_predictions = model.predict(val_X)

# Evaluate
from sklearn.metrics import mean_absolute_error
mae = (val_y, val_predictions)