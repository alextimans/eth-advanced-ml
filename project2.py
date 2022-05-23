import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import *

### Load data
X_train = pd.read_csv('./Desktop/AML/Projects/task2/X_train.csv').drop(columns='id')
y_train = np.array(pd.read_csv('./Desktop/AML/Projects/task2/y_train.csv').drop(columns='id'))
X_test = pd.read_csv('./Desktop/AML/Projects/task2/X_test.csv').drop(columns='id')
scorer = metrics.make_scorer(metrics.balanced_accuracy_score)

### Data exploration
pd.DataFrame(X_train).isna().sum().sum()
pd.DataFrame(X_test).isna().sum().sum()
plt.hist(y_train, edgecolor='black', bins=10)
plt.scatter(range(0, len(X_train['x0'])), X_train['x0'])
plt.hist(X_train['x0'], edgecolor='black', bins=10)

'''
### Data scaling
scaler = preprocessing.StandardScaler()
scaler = preprocessing.PowerTransformer()
scaler = preprocessing.QuantileTransformer(output_distribution='normal')
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

### Outlier detection
outlier_method = neighbors.LocalOutlierFactor(contamination=.025, n_jobs=-1)
outlier_method = ensemble.IsolationForest(contamination=.025, n_jobs=-1)
outlier_bool = outlier_method.fit_predict(X_train)
print("Outliers:", np.count_nonzero(outlier_bool == -1))
outlier_index = list(np.where(outlier_bool == -1)[0])
y_train = np.delete(y_train, outlier_index, axis=0)
X_train = np.delete(X_train, outlier_index, axis=0)

### Feature selection
feat_select = feature_selection.SelectPercentile(score_func=feature_selection.mutual_info_classif)
percentiles = [80, 85, 90]
score_means, score_stds = list(), list()

for perc in percentiles:
    print('Validating', perc, '% of features')
    feat_select.set_params(percentile=perc)
    X_tr = feat_select.fit_transform(X_train, y_train.ravel())
    print('Transformed. Calculating validation score...')
    score = model_selection.cross_val_score(final_model, X_tr, y_train.ravel(), scoring=scorer, cv=10, n_jobs=-1)
    score_means.append(score.mean())
    score_stds.append(score.std())
    print('Score:', round(score.mean(),5), '(+/-', round(score.std(), 5), ')')

feat_select.set_params(percentile=80)
feat_select.fit(X_train, y_train.ravel())
X_train = feat_select.transform(X_train)
X_test = feat_select.transform(X_test)
'''

### Param selection
param_grid = {'C':[1e-2, 1e-1, 1e0, 1e1, 1e2], 'gamma':[1e-3, 1e-2, 1e-1, 1e0, 1e1]}
model = svm.SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced')
grid_search = model_selection.GridSearchCV(model, param_grid, scoring=scorer, cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train.ravel())

### Final model
final_model = svm.SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced')

'''
clf1 = svm.SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced')
clf2 = linear_model.LogisticRegression(C=1e-3, penalty='l2', solver='saga', max_iter=150, class_weight='balanced')
clf3 = naive_bayes.GaussianNB()
final_model = ensemble.VotingClassifier(estimators=[('svc', clf1), ('logr', clf2), ('gnb', clf3)], voting='hard', n_jobs=-1, verbose=True)
'''

cv_score = model_selection.cross_val_score(final_model, X_train, y_train.ravel(), cv=10, scoring=scorer, n_jobs=-1)
print('CV Score:', cv_score.mean())

final_model.fit(X_train, y_train.ravel())
y_pred = final_model.predict(X_test)
y_pred = pd.DataFrame({'id': range(0, len(y_pred)), 'y': y_pred})
y_pred.to_csv('./Desktop/AML/Projects/task2/y_pred.csv', index=False)
