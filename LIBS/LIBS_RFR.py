"""
    Random forest LIBS quantitative analysis
    @Travis Zeng on 2018/9/14
"""
"""
RF基本步骤
1.从数据集中选取 N 个随机记录。
2.根据这 N 个记录构建决策树。
3.选择算法中需要的决策树的个数，然后重复步骤1和2。
4.在回归问题的情况下，对于新记录，森林中的每棵树预测Y（输出）的值。
最终值可以通过取得森林中所有树木预测的所有值的平均值来计算。
对于分类问题，林中的每棵树都会预测新记录所属的类别。最后，新记录被分配到赢得多数票的类别。


"""
import numpy as np
import pandas as pd

dataset = pd.read_csv('petrol_consumption.csv')

#本部分先将数据分成特征和标签集，然后将得到的数据分成训练集和测试集。
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

