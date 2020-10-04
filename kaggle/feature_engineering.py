'''Baseline Model'''

import pandas as pd

click_data = pd.read_csv('../input/feature-engineering-data/train_sample.csv',
                         parse_dates=['click_time'])

# 1) Construct features from timestamps
# 
# `click_data` DataFrame has a `'click_time'` column with timestamp data.
# Use this column to create features for the coresponding day, hour, minute and second. 
# Store these as new integer columns `day`, `hour`, `minute`, and `second` in a new DataFrame `clicks`.
clicks = click_data.copy()
clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
clicks['hour'] = clicks['click_time'].dt.hour.astype('uint8')
clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
clicks['second'] = clicks['click_time'].dt.second.astype('uint8')

# 2) Label Encoding
# For each of the categorical features `['ip', 'app', 'device', 'os', 'channel']`, use scikit-learn's `LabelEncoder` to create new features in the `clicks` DataFrame. 
# The new column names should be the original column name with `'_labels'` appended, like `ip_labels`.

from sklearn import preprocessing

cat_features = ['ip', 'app', 'device', 'os', 'channel']

# Create new columns in clicks using preprocessing.LabelEncoder()
label_enc = preprocessing.LabelEncoder()
for feature in cat_features:
    encoded = label_enc.fit_transform(clicks[feature])
    # append to clicks dataframe
    clicks[feature + '_labels'] = encoded
    
# 3) Create train/validation/test splits
# Here we'll create training, validation, and test splits. First, `clicks` DataFrame is sorted in order of increasing time. 
# The first 80% of the rows are the train set, the next 10% are the validation set, and the last 10% are the test set.

feature_cols = ['day', 'hour', 'minute', 'second', 
                'ip_labels', 'app_labels', 'device_labels',
                'os_labels', 'channel_labels']

valid_fraction = 0.1
clicks_srt = clicks.sort_values('click_time')
valid_rows = int(len(clicks_srt) * valid_fraction)
train = clicks_srt[:-valid_rows * 2]
# valid size == test size, last two sections of the data
valid = clicks_srt[-valid_rows * 2:-valid_rows]
test = clicks_srt[-valid_rows:]

# 4) Train with LightGBM
# Now we can create LightGBM dataset objects for each of the smaller datasets and train the baseline model.

import lightgbm as lgb

dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
dtest = lgb.Dataset(test[feature_cols], label=test['is_attributed'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10)

# 5) Evaluate the model
from sklearn import metrics

ypred = bst.predict(test[feature_cols])
score = metrics.roc_auc_score(test['is_attributed'], ypred)
print(f"Test score: {score}")


'''Categorial Encodings'''

# count encoding - replaces the categorical value with the number of it times it appeared on the dataset
import category_encoders as ce

data = 'somedata'
cat_features = ['col1', 'col2', 'col3']
count_enc = ce.CountEncoder()
count_encoded = count_enc.fit_transform(data[cat_features])
data = data.join(count_encoded.add_suffix("_count"))

# target encoding - replaces the categorical value with the average value of the target for that value of the feature
# uses targets to create new features, so including validation and test data in target encodings would be a form of data leakage
# only learn target encodings with training dataset only
target_enc = ce.TargetEncoder(cols=cat_features)
target_enc.fit(train[cat_features], train['outcome'])
# Transform the features, rename the columns with _target suffix, and join to dataframe
train_TE = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))
valid_TE = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))

# catboost encoding - similar to target encoding, but the target probability is calculated only from the rows before it
target_enc = ce.CatBoostEncoder(cols=cat_features)
target_enc.fit(train[cat_features], train['outcome'])
# Transform the features, rename columns with _cb suffix, and join to dataframe
train_CBE = train.join(target_enc.transform(train[cat_features]).add_suffix('_cb'))
valid_CBE = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_cb'))

'''Feature Generation'''

# easiest way is to combine categorical variables
# interaction - provide information about correlations between categorical variables 
interactions = data['col1'] + '_' + data['col2']
data_interaction = data.assign(col1_col2=label_enc.fit_transform(interactions))

# or create more categorical features from data 
# also effective to create new numerical features e.g. events in the past _ hours, time since last event 

'''Feature Selection'''

# 1) Univariate feature selection
# measure how strongly the target depends on the feature using a statistical test like x^2 or ANOVA F-value
# F-value measures the linear dependency between feature variable and the target, so it might underestimate the relation between feature and target for nonlinear relationships

from sklearn.feature_selection import SelectKBest, f_classif

feature_cols = data.columns.drop('outcome')
selector = SelectKBest(f_classif, k=5) # keep 5 features
train_data = 'select features using only training set to prevent source of leakage'
X_new = selector.fit_transform(train_data[feature_cols], train_data['outcome'])
selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                 index=train_data.index, 
                                 columns=feature_cols)
# dropped columns have values of all 0s, so var is 0, drop them
selected_columns = selected_features.columns[selected_features.var() != 0]

# 2) L1 (lasso) regularization
# Univariate only consider one feature at a time, L1 selects all features

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

X, y = train_data[train_data.columns.drop("outcome")], train_data['outcome']
# Set the regularization parameter C=1
logistic = LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=7).fit(X, y)
model = SelectFromModel(logistic, prefit=True)
X_new = model.transform(X)
selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                 index=train_data.index, 
                                 columns=feature_cols)
# dropped columns have values of all 0s, so var is 0, drop them
selected_columns = selected_features.columns[selected_features.var() != 0]
