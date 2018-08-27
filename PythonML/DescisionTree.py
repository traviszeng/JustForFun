"""
决策树常见参数和概念:
根结点：表示所有数据样本并可以进一步划分为两个或多个子结点的父结点。

分裂（Splitting）：将一个结点划分为两个或多个子结点的过程。

决策结点：当一个子结点可进一步分裂为多个子结点，那么该结点就称之为决策结点。

叶/终止结点：不会往下进一步分裂的结点，在分类树中代表类别。

分枝/子树：整棵决策树的一部分。

父结点和子结点：如果一个结点往下分裂，该结点称之为父结点而父结点所分裂出来的结点称之为子结点。

结点分裂的最小样本数：在结点分裂中所要求的最小样本数量（或观察值数量）。这种方法通常可以用来防止过拟合，较大的最小样本数可以防止模型对特定的样本学习过于具体的关系，该超参数应该需要使用验证集来调整。

叶结点最小样本数：叶结点所要求的最小样本数。和结点分裂的最小样本数一样，该超参数同样也可以用来控制过拟合。对于不平衡类别问题来说，我们应该取较小的值，因为属于较少类别的样本可能数量上非常少。

树的最大深度（垂直深度）：该超参数同样可以用来控制过拟合问题，较小的深度可以防止模型对特定的样本学习过于具体的关系，该超参数同样需要在验证集中调整。

叶结点的最大数量：叶结点的最大个数可以替代数的最大深度这一设定。因为生成一棵深度为 n 的二叉树，它所能产生的最大叶结点个数为 2^n。

分裂所需要考虑的最大特征数：即当我们搜索更好分离方案时所需要考虑的特征数量，我们常用的方法是取可用特征总数的平方根为最大特征数。

"""


import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

training_data = './94AmericaIncome/adult_train.csv'
test_data = './94AmericaIncome/adult_test.csv'

columns = ['Age','Workclass','fnlgwt','Education','EdNum','MaritalStatus','Occupation','Relationship','Race','Sex','CapitalGain','CapitalLoss','HoursPerWeek','Country','Income']

df_train_set = pd.read_csv(training_data, names=columns)
df_test_set = pd.read_csv(test_data, names=columns, skiprows=1)
df_train_set.drop('fnlgwt', axis=1, inplace=True)
df_test_set.drop('fnlgwt', axis=1, inplace=True)

#数据清洗 将所有列的的特殊字符移除，此外任何空格或者「.」都需要移除

#replace the special character to "Unknown"
for i in df_train_set.columns:
    df_train_set[i].replace(' ?', 'Unknown', inplace=True)
    df_test_set[i].replace(' ?', 'Unknown', inplace=True)
for col in df_train_set.columns:
    if df_train_set[col].dtype != 'int64':
        df_train_set[col] = df_train_set[col].apply(lambda val: val.replace(" ", ""))
        df_train_set[col] = df_train_set[col].apply(lambda val: val.replace(".", ""))
        df_test_set[col] = df_test_set[col].apply(lambda val: val.replace(" ", ""))
        df_test_set[col] = df_test_set[col].apply(lambda val: val.replace(".", ""))


#移除冗余的特征 Education和Country
df_train_set.drop(["Country","Education"],axis=1,inplace=True)
df_test_set.drop(["Country","Education"],axis=1,inplace=True)

#Age 和 EdNum 列是数值型的，我们可以将连续数值型转化为更高效的方式，例如将年龄换为 10 年的整数倍，教育年限换为 5 年的整数倍
colnames = list(df_train_set.columns)
colnames.remove('Age')
colnames.remove('EdNum')
colnames = ['AgeGroup','Education']+colnames

labels = ["{0}-{1}".format(i,i+9) for i in range(0,100,10)]
df_train_set['AgeGroup'] = pd.cut(df_train_set.Age,range(0,101,10),right = False,labels=labels)
df_test_set['AgeGroup'] = pd.cut(df_test_set.Age,range(0,101,10),right = False,labels=labels)

labels = ["{0}-{1}".format(i,i+9) for i in range(0,20,5)]
df_train_set['Education'] = pd.cut(df_train_set.EdNum,range(0,21,5),right = False,labels=labels)
df_test_set['Education'] = pd.cut(df_test_set.EdNum,range(0,21,5),right = False,labels=labels)


df_train_set = df_train_set[colnames]
df_test_set = df_test_set[colnames]


#以图像的形式看一下训练数据中的不同特征的分布和相互依存（inter-dependence）关系。首先看一下关系（Relationships）和婚姻状况（MaritalStatus）特征是如何相互关联的
(ggplot(df_train_set, aes(x = "Relationship", fill = "MaritalStatus"))+ geom_bar(position="fill")+ theme(axis_text_x = element_text(angle = 60, hjust = 1)))


#不同年龄组中，教育对收入的影响
(ggplot(df_train_set, aes(x = "Education", fill = "Income"))+ geom_bar(position="fill")+ theme(axis_text_x = element_text(angle = 60, hjust = 1))+ facet_wrap('~AgeGroup'))


(ggplot(df_train_set, aes(x = "Education", fill = "Income"))+ geom_bar(position="fill")+ theme(axis_text_x = element_text(angle = -90, hjust = 1))+ facet_wrap('~Sex'))

"""
理解了我们数据中的一些关系，所以就可以使用 sklearn.tree.DecisionTreeClassifier 创建一个简单的树分类器模型。
然而，为了使用这一模型，我们需要把所有我们的非数值数据转化成数值型数据。
我们可以直接在 Pandas 数据框架中使用 sklearn.preprocessing.LabeEncoder 模块和 sklearn_pandas 模块就可以轻松地完成这一步骤。
"""
mapper = DataFrameMapper([('AgeGroup', LabelEncoder()),
                          ('Education', LabelEncoder()),
                          ('Workclass', LabelEncoder()),
                          ('MaritalStatus', LabelEncoder()),
                          ('Occupation', LabelEncoder()),
                          ('Relationship', LabelEncoder()),
                          ('Race', LabelEncoder()),
                          ('Sex', LabelEncoder()),
                          ('Income', LabelEncoder())], df_out=True, default=None)

cols = list(df_train_set.columns)
cols.remove("Income")
cols = cols[:-3] + ["Income"] + cols[-3:]

df_train = mapper.fit_transform(df_train_set.copy())
df_train.columns = cols

df_test = mapper.transform(df_test_set.copy())
df_test.columns = cols

cols.remove("Income")
x_train, y_train = df_train[cols].values, df_train["Income"].values
x_test, y_test = df_test[cols].values, df_test["Income"].values

#尝试创建第一个未优化的模型
treeClassifier = DecisionTreeClassifier()
treeClassifier.fit(x_train, y_train)
treeClassifier.score(x_test, y_test)


#在分类问题中，混淆矩阵（confusion matrix）是衡量模型精度的好方法。使用下列代码我们可以绘制任意基于树的模型的混淆矩

def plot_confusion_matrix(cm, classes, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmap = plt.cm.Blues
    title = "Confusion Matrix"
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


y_pred = treeClassifier.predict(x_test)
cfm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(10,6))
plot_confusion_matrix(cfm, classes=["<=50K", ">50K"], normalize=True)

#发现多数类别（<=50K）的精度为 90.5%，少数类别（>50K）的精度只有 60.8%。

#让我们看一下调校此简单分类器的方法。我们能使用带有 5 折交叉验证的 GridSearchCV() 来调校树分类器的各种重要参数
#plt.show()
print("test")

parameters = {'max_features':(None, 9, 6),'max_depth':(None, 24, 16),'min_samples_split': (2, 4, 8),'min_samples_leaf': (16, 4, 12)}

clf = GridSearchCV(treeClassifier, parameters, cv=5, n_jobs=4)
clf.fit(x_train, y_train)
#clf.best_score_, clf.score(x_test, y_test), clf.best_params_

#经过优化，我们发现精度上升到了 85.9%。在上方，我们也可以看见最优模型的参数。现在，让我们看一下 已优化模型的混淆矩阵（confusion matrix）

y_pred = clf.predict(x_test)
cfm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(10,6))
plot_confusion_matrix(cfm, classes=["<=50K", ">50K"], normalize=True)
