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
mae = mean_absolute_error(val_y, val_predictions)

'''Intermediate ML'''

'''Missing Values

1) drop columns with missing values 
    + simplest
    - model losses access to a lot of potential information if column is important
2) imputation: filling missing values with some number
    + more accurate models
    - the value is not right in most cases
3) extend imputation: add a column that shows location of the imputed entries
    - usually does not help
finally, score with MAE
'''

# 1) drop columns
cols_with_missing_values = [col for col in train_X.columns
                            if train_X[col].isnull().any()]

reduced_train_x = train_X.drop(cols_with_missing_values, axis=1)
reduced_val_x = val_X.drop(cols_with_missing_values, axis=1)

# 2) imputation
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_train_x = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_val_x = pd.DataFrame(my_imputer.fit_transform(val_X))

# imputation remove column names; adding it back
imputed_train_x.columns = train_X.columns
imputed_val_x.columns = val_X.columns

# print shape of data (num_rows, num_columns)
train_X.shape()

# print number of missing values in each column of training data
missing_val_count_by_column = train_X.isnull().sum() # returns all column with sum of null 
missing_val_count_by_column[missing_val_count_by_column > 0] # returns column names if sum > 0 (bool)

'''Categorical Variables

1) drop categorical variables 
2) label encoding: assigns each unique value to a different integer
    e.g. never = 0, rarely = 1, ... 
    + good for categories with indisputable ranking -- ordinal variables
3) one-hot encoding: creates new columns indicating the presence and absence of each possible value in the original data
    e.g     red     yellow  green
    red     1       0       0
    yellow  0       1       0
    green   0       0       1
    + good for categories with no ordered ranking -- nominal variables
    - large number of values (max <15 different values)
'''

# 1) drop columns
drop_train_X = train_X.select_dtypes(exclude=['object'])
drop_val_X = val_X.select_dtypes(exclude=['object'])

# 2) label encoding
from sklearn.preprocessing import LabelEncoder

# copy to avoid changing original data 
label_train_X = train_X.copy()
label_val_X = val_X.copy()

label_encoder = LabelEncoder()

# get list of categorical variables
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)

# for each column, we randomly assign each unique value to a different integer
# we can expect an additional boost if we provide custom labels however
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(train_X[col])
    label_X_valid[col] = label_encoder.transform(val_X[col])

# 3) one-hot encoding
from sklearn.preprocessing import OneHotEncoder

# apply one-hot encoder to each column with categorical data
# handle_unknown is set to ignore to avoid errors when the validation data contains classes that aren't represented in the training data
# spare is set to False to ensure that encoded columns are returns as numpy array as opposed to sparse matrix
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(val_X[object_cols]))

# one-hot encoding removes index; adding it back
OH_cols_train.index = train_X.index
OH_cols_valid.index = val_X.index

# remove categorical columns (will replace with one-hot encoding)
num_train_X = train_X.drop(object_cols, axis=1)
num_val_X = val_X.drop(object_cols, axis=1)

# add one-hot encoded columns to numerical features
OH_train_X = pd.concat([num_train_X, OH_cols_train], axis=1)
OH_val_X = pd.concat([num_val_X, OH_cols_valid], axis=1)

'''Pipelines benefits: cleaner code, fewer bugs, easier to productionize, more options for model validation
'''
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# preprocessing numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# preprocessing categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# create and evaluate the pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# preprocessing of training data, fit model 
my_pipeline.fit(train_X, train_y)

# preprocessing of validation data, get predictions
preds = my_pipeline.predict(val_X)

# evaluate the model
score = mean_absolute_error(val_y, preds)

'''Cross-Validation 
    - to get a more accurate measurement of model quality
    - run model on different subsets of data
        e.g. divide data into 5 folds, each fold holding 20% of full dataset 
                experiment 1: use fold 1 as validation (or holdout) set and everything else as training data
                experiment 2: use fold 2 as validation .... 
    + for small datasets, run cross-validation for all folds
    + for larger datasets, a single validation set is sufficient 
'''

from sklearn.model_selection import cross_val_score

# multiple by -1 since sklearn calculates *negative* MAE
# cv determines the cross-validation splitting strategy 
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
# get average
scores.mean()

'''XGBoost (or extreme gradient boosting)

Ensemble methods: combination of prediction by several models (i.e. Random Forest)
Gradient boosting: iteratively adds models into an essemble
    1) use naive model to generate prediction
    2) calculate lose 
    3) use lose to train new model 
        - specifically, we determine the model parameters so that adding new model to the ensemble reduce the loss
        - gradient refers to "gradient descent" https://en.wikipedia.org/wiki/Gradient_descent
    4) add new model to the ensemble 
    ...and repeat!
'''

from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=500, 
                        learning_rate=0.05, 
                        n_jobs=4)
''' n_estimators - how many times to go through the modeling cycle
        too low = underfitting 
        too hight = overfitting

    learning_rate - prediction * learning rate before adding to ensemble
        each additional tree helps us less, can help avoid overfitting

    n_jobs - set to the number of cores to use parallelism
'''
my_model.fit(train_X, train_y,
            early_stopping_rounds=5, 
            eval_set=[(val_X, val_y)],
            verbose=False)
''' early_stopping_rounds - how many rounds of straight deteriorating to allow before stopping
        model stops iterating when the validation scores stops improving
        use a high number for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating
'''
preds = my_model.predict(val_X)
score = mean_absolute_error(preds, val_y)

'''Data Leakage 

1) target leakage
    - when training data contains information about the target, but similar data is not available when the model is used for prediction
2) train-test contamination
    - running preprocessing before calling train_test_split()
    - for validations based on simple train-step split, exclude validation from any type of fitting

A combination of caution, common sense, and data exploration can help identify target leakage.
'''