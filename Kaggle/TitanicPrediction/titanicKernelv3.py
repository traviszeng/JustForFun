import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns



warnings.filterwarnings('ignore')

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',100)

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

dataset  = [train,test]


#data cleaning
for data in dataset:
    #fill empty ages with median value
    data['Age'].fillna(data['Age'].median(),inplace=True)

    #fill empty embarked with mode value
    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)

    data['Fare'].fillna(data['Fare'].median(),inplace=True)

drop_column = ['PassengerId','Cabin', 'Ticket']
train.drop(drop_column,axis = 1,inplace = True)
test.drop(drop_column,axis = 1,inplace = True)

dataset = [train,test]

#Feature engineering

for data in dataset:
    # Discrete variables
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    data['IsAlone'] = 1  # initialize to yes/1 is alone
    data['IsAlone'].loc[data['FamilySize'] > 1] = 0  # now update to no/0 if family size is greater than 1
    #extract title from name
    data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    #Binning fare and age
    data['FareBin'] = pd.qcut(data['Fare'], 4)
    data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)

stat_min = 10
title_names = (train['Title'].value_counts()<stat_min)#this will create a true false series with title name as index

train['Title']= train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

#code categorical data
label = LabelEncoder()
for data in dataset:
    data['Sex_Code'] = label.fit_transform(data['Sex'])
    data['Embarked_Code'] = label.fit_transform(data['Embarked'])
    data['Title_Code'] = label.fit_transform(data['Title'])
    data['AgeBin_Code'] = label.fit_transform(data['AgeBin'])
    data['FareBin_Code'] = label.fit_transform(data['FareBin'])

target = ['Survived']

train_x= ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
train_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
train_xy = target+train_x

train_x_bin  = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
train_xy_bin = target+train_x_bin

train_dummy = pd.get_dummies(train[train_x])
train_x_dummy = train_dummy.columns.tolist()
train_xy_dummy = target+train_x_dummy


train1_x, test1_x, train1_y, test1_y = train_test_split(train[train_x_calc], train[target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = train_test_split(train[train_x_bin], train[target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = train_test_split(train_dummy[train_x_dummy], train[target], random_state = 0)


#Discrete Variable Correlation by Survival using group by aka pivot table
for x in train_x:
    if train[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(train[[x, target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')

# Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    # GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    # Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()
]


#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

MLA_predict  = train[target]

# index through MLA and save performance to table
row_index = 0
for alg in MLA:
    # set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    # score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, train[train_x_bin], train[target], cv=cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results[
                                                                'test_score'].std() * 3  # let's know the worst that can happen!

    # save MLA predictions - see section 6 for usage
    alg.fit(train[train_x_bin], train[target])
    MLA_predict[MLA_name] = alg.predict(train[train_x_bin])

    row_index += 1

# print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)


sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
#plt.show()