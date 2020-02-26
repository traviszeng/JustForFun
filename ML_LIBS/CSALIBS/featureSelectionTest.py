# coding=utf-8
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
from sklearn.cross_validation import cross_val_score
import numpy as np
import pylab as pl

# -------------download data
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# -------------sperate data and target
y = titanic['survived']
X = titanic.drop(['row.names', 'name', 'survived'], axis=1)
# -------------fulfill lost data with mean value
X['age'].fillna(X['age'].mean(), inplace=True)
X.fillna('UNKNOWN', inplace=True)
# -------------split data，25% for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
# -------------feature vectorization
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))
# -------------
print
'Dimensions of handled vector', len(vec.feature_names_)
# -------------use DTClassifier to predict and measure performance
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
print
dt.score(X_test, y_test)
# -------------selection features ranked in the front 20%,use DTClassifier with the same config to predict and measure performance
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
print
dt.score(X_test_fs, y_test)

percentiles = range(1, 100, 2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
    results = np.append(results, scores.mean())
print
results
# -------------find feature selection percent with the best performance
opt = int(np.where(results == results.max())[0])
print
'Optimal number of features', percentiles[opt]
# TypeError: only integer scalar arrays can be converted to a scalar index
# transfer list to array
# print 'Optimal number of features',np.array(percentiles)[opt]

# -------------use the selected features and the same config to measure performance on test datas
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
print
dt.score(X_test_fs, y_test)

pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()
