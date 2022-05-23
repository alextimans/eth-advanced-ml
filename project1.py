import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import *
import lightgbm

### Load data
X_train = pd.read_csv('./Desktop/AML/Projects/task1/X_train.csv').drop(columns='id')
y_train = np.array(pd.read_csv('./Desktop/AML/Projects/task1/y_train.csv').drop(columns='id'))
X_test = pd.read_csv('./Desktop/AML/Projects/task1/X_test.csv').drop(columns='id')
scorer = metrics.make_scorer(metrics.r2_score)

### Feature selection 1
feat_select = feature_selection.VarianceThreshold(threshold=0.01)
feat_select.fit(X_train, y_train.ravel())
print(np.count_nonzero(feat_select.get_support()==False))
X_train = feat_select.transform(X_train)
X_test = feat_select.transform(X_test)

### Data scaling
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

### Missing values
pd.DataFrame(X_train).isna().sum().sum()
pd.DataFrame(X_test).isna().sum().sum()
# number of missing values per feature in X_train
plt.scatter(range(0, X_train.shape[1]), pd.DataFrame(X_train).isna().sum())

imputer = impute.KNNImputer(n_neighbors=5, weights='distance')
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

### Outlier detection
outlier_method = ensemble.IsolationForest()
outlier_bool = outlier_method.fit_predict(X_train)
print("Outliers:", np.count_nonzero(outlier_bool == -1))
outlier_index = list(np.where(outlier_bool == -1)[0])

y_train = np.delete(y_train, outlier_index, axis=0)
X_train = np.delete(X_train, outlier_index, axis=0)

### Feature selection 2: Option 1 - Gradient Boosting
feat_estimator = ensemble.GradientBoostingRegressor(n_estimators=350, max_depth=5)
feat_select = feature_selection.SelectFromModel(feat_estimator, threshold=-np.inf, max_features=200)
feat_select.fit(X_train, y_train.ravel())
X_train = feat_select.transform(X_train)
X_test = feat_select.transform(X_test)

### Feature selection 2: Option 2 - LightGBM
param = {'objective': 'regression', 'metric': 'l2', 'min_data_in_leaf': 12, 'max_bin': 130, 'lambda_l2': .9, 'max_depth': 4, 'num_leaves': 12}
train_data = lightgbm.Dataset(pd.DataFrame(X_train), label=pd.DataFrame(y_train))
model = lightgbm.train(param, train_data)
feat = model.feature_importance()
X_train = X_train[:, feat>1]
X_test = X_test[:, feat>1]

### Param tuning: Option 1 - grid search with Gradient Boosting
param_grid = {'n_estimators':[325,350,375,400]}
model = ensemble.GradientBoostingRegressor(max_depth=5, learning_rate=0.1, loss='ls')
grid_search = model_selection.GridSearchCV(model, param_grid, scoring=scorer, cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train.ravel())

### Param tuning: Option 2 - manual grid search with LightGBM
train, test, train_l, test_l = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=1234)
N = 1; tun = []; grid = np.arange(1, N, 1)
for n in range(1, N):
    param = {'objective': 'regression', 'metric': 'l2', 'min_data_in_leaf': 12, 'max_bin': 130, 'lambda_l1': .005,
             'lambda_l2': .9, 'max_depth': 4, 'feature_fraction': .5, 'num_leaves': 12, 'extra_trees': 'true'}
    train_data = lightgbm.Dataset(pd.DataFrame(train), label=pd.DataFrame(train_l))
    model = lightgbm.train(param, train_data)
    y_tr = model.predict(train)
    y_pred = model.predict(test)
    tun.append(metrics.r2_score(y_pred, test_l))
plt.plot(grid, tun)

# Train the model and compare the scores on training/validation set
param={'objective': 'regression', 'metric': 'l2', 'min_data_in_leaf': 12, 'max_bin': 130, 'lambda_l2': .9, 'max_depth': 4, 'num_leaves': 12}
train_data = lightgbm.Dataset(pd.DataFrame(train), label=pd.DataFrame(train_l))
model = lightgbm.train(param, train_data)
feat = model.feature_importance()
y_tr = model.predict(train)
print(metrics.r2_score(y_tr,train_l))
y_pred = model.predict(test)
print(metrics.r2_score(y_pred,test_l))

### Final model
train_data = lightgbm.Dataset(pd.DataFrame(X_train), label=pd.DataFrame(y_train))
final_model = lightgbm.train(param, train_data)
y_tr = final_model.predict(X_train)
print(metrics.r2_score(y_tr, y_train))
final_model.fit(X_train, y_train.ravel())
y_pred = final_model.predict(X_test)
y_pred = pd.DataFrame({'id': range(0, len(y_pred)), 'y': y_pred})
y_pred.to_csv('./y_pred.csv', index=False)
