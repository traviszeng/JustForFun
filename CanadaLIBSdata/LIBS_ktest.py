"""
利用加拿大航天局的LIBS数据进行LIBS定量分析实验
"""
import os
import sys
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import copy
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import warnings

warnings.filterwarnings("ignore")

concentrate_data = pd.read_csv("E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\Sample_Composition_Data.csv")
# 前81行为数据
concentrate_data = concentrate_data.loc[0:81]

# 数据清洗
for indexs in concentrate_data.index:
    for i in range(1, 12):
        if concentrate_data.loc[indexs].values[i] == '-':
            concentrate_data.loc[indexs].values[i] = 0.0

        else:
            try:
                concentrate_data.loc[indexs].values[i] = float(concentrate_data.loc[indexs].values[i])
                if float(concentrate_data.loc[indexs].values[i]) > 1:
                    concentrate_data.loc[indexs].values[i] = concentrate_data.loc[indexs].values[i] / 100
            except ValueError:
                concentrate_data.loc[indexs].values[i] = float(concentrate_data.loc[indexs].values[i][1:])
                if float(concentrate_data.loc[indexs].values[i]) > 1:
                    concentrate_data.loc[indexs].values[i] = concentrate_data.loc[indexs].values[i] / 100

# 检查是否将所有非数字处理好
for column in concentrate_data.columns:
    print(concentrate_data[column].isna().value_counts())

route_200_AVG = "E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\csv Material Large Set 200pulseaverage"
route_1000_AVG = "E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\csv Certified Samples Subset 1000pulseaverage"

postfix_200AVG = "_200AVG.csv"
postfix_1000AVG = "_1000AVG.csv"

"""
加载训练样本
"""

data_set_200AVG = {}
concentrate_set_200AVG = {}
# 加载200AVG的样本，并将其存到data_set_200AVG中
os.chdir(route_200_AVG)
num = 0
for indexs in concentrate_data.index:
    if os.path.exists(concentrate_data.loc[indexs].values[0] + postfix_200AVG):
        num += 1
        print("Get data file:" + concentrate_data.loc[indexs].values[0] + postfix_200AVG)
        data = pd.read_csv(concentrate_data.loc[indexs].values[0] + postfix_200AVG, header=None,
                           names=['WaveLength', 'Intensity'])
        # data中强度<0的统统变为0
        data.loc[data.Intensity < 0, 'Intensity'] = 0
        data_set_200AVG[concentrate_data.loc[indexs].values[0] + "_200AVG"] = data
        concentrate_set_200AVG[concentrate_data.loc[indexs].values[0] + "_200AVG"] = concentrate_data.loc[
                                                                                         indexs].values[1:]
    # 处理hand sample类型的样本
    if re.match('hand sample*', concentrate_data.loc[indexs].values[0]):

        f_list = concentrate_data.loc[indexs].values[0].split()
        filename = f_list[0] + " " + f_list[1] + postfix_200AVG
        if os.path.exists(filename):
            num += 1
            print("Get data file:" + filename)
            data = pd.read_csv(filename, header=None, names=['WaveLength', 'Intensity'])
            # data中强度<0的统统变为0
            data.loc[data.Intensity < 0, 'Intensity'] = 0
            data_set_200AVG[concentrate_data.loc[indexs].values[0] + "_200AVG"] = data
            concentrate_set_200AVG[concentrate_data.loc[indexs].values[0] + "_200AVG"] = concentrate_data.loc[
                                                                                             indexs].values[1:]

print("Get " + str(num) + " 200_AVG files.")
print()

data_set_1000AVG = {}
concentrate_set_1000AVG = {}
num = 0
# 加载1000AVG的样本，并将其存到data_set_1000AVG中
os.chdir(route_1000_AVG)
for indexs in concentrate_data.index:
    if os.path.exists(concentrate_data.loc[indexs].values[0] + postfix_1000AVG):
        num += 1
        print("Get data file:" + concentrate_data.loc[indexs].values[0] + postfix_1000AVG)
        data = pd.read_csv(concentrate_data.loc[indexs].values[0] + postfix_1000AVG, header=None,
                           names=['WaveLength', 'Intensity'])
        # data中强度<0的统统变为0
        data.loc[data.Intensity < 0, 'Intensity'] = 0
        data_set_1000AVG[concentrate_data.loc[indexs].values[0] + "_1000AVG"] = data
        concentrate_set_1000AVG[concentrate_data.loc[indexs].values[0] + "_1000AVG"] = concentrate_data.loc[
                                                                                           indexs].values[1:]
    # 处理hand sample类型的样本
    if re.match('hand sample*', concentrate_data.loc[indexs].values[0]):

        f_list = concentrate_data.loc[indexs].values[0].split()
        filename = f_list[0] + " " + f_list[1] + postfix_1000AVG
        if os.path.exists(filename):
            num += 1
            print("Get data file:" + filename)
            data = pd.read_csv(filename, header=None, names=['WaveLength', 'Intensity'])
            # data中强度<0的统统变为0
            data.loc[data.Intensity < 0, 'Intensity'] = 0
            data_set_1000AVG[concentrate_data.loc[indexs].values[0] + "_1000AVG"] = data
            concentrate_set_1000AVG[concentrate_data.loc[indexs].values[0] + "_1000AVG"] = concentrate_data.loc[
                                                                                               indexs].values[1:]

print("Get " + str(num) + " 1000_AVG files.")

# 去掉hand sample34的记录（全为0）
del concentrate_set_200AVG['hand sample37 barite_200AVG']
del data_set_200AVG['hand sample37 barite_200AVG']

# 画一个折线图尝试看看
s_name = "RockFer2_200AVG"
intensity = data_set_200AVG[s_name].Intensity
wavelength = data_set_200AVG[s_name].WaveLength
intensity = np.array(intensity)
wavelength = np.array(wavelength)
plt.plot(wavelength, intensity)
# plt.show()

print("准备NIST库相关数据")
nist = pd.read_csv("E:\\JustForFun\\CanadaLIBSdata\\andor.nist", header=None,
                   names=['WaveLength', 'Element', 'Type', 'Unknown', 'Importance'])
nist = nist.loc[1:]
# 删除未知列
del nist['Unknown']
# 筛选在样本精度范围的nist线
nist = nist.loc[nist.WaveLength >= 198.066]
nist = nist.loc[nist.WaveLength <= 970.142]
element_dict = {}
for indexs in nist.index:
    if nist.loc[indexs].Element in element_dict:
        element_dict[nist.loc[indexs].Element].append([nist.loc[indexs].WaveLength, nist.loc[indexs].Importance])
    else:
        element_dict[nist.loc[indexs].Element] = [[nist.loc[indexs].WaveLength, nist.loc[indexs].Importance]]

"""
    Ensemble bagging method
    using average
"""


class baggingAveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


"""
    Ensemble stacking method
    一层stacking
"""


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # 在原有模型的拷贝上再次训练
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # 在拷贝的基本模型上进行out-of-fold预测，并用预测得到的作为meta model的feature
        out_of_fold_predictions = np.zeros((len(X), len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                # print(train_index)
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(np.array(X)[train_index], np.array(y)[train_index])
                y_pred = instance.predict(np.array(X)[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 用out-of-foldfeature训练meta-model
        # print(type(out_of_fold_predictions))
        # print(len(y))
        self.meta_model_.fit(np.array(out_of_fold_predictions), y)
        return self

    # 使用基学习器预测测试数据，并将各基学习器预测值平均后作为meta-data feed给meta-model在做预测
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


# 准备训练数据，未作任何处理，将整个光谱输入
X = []
# 波长
waveLength = []
# AL的相关浓度信息
Al_y = []
# Fe
Fe_y = []
# Ca
Ca_y = []
# K
K_y = []
# Mg
Mg_y = []
# Na
Na_y = []
# Si
Si_y = []
# Ti
Ti_y = []
# P
P_y = []
# Mn
Mn_y = []

for samplename, concentrate in concentrate_set_200AVG.items():
    X.append(np.array(data_set_200AVG[samplename].Intensity))
    waveLength.append(np.array(data_set_200AVG[samplename].WaveLength))
    Al_y.append(concentrate[0] * 100)
    Ca_y.append(concentrate[1] * 100)
    Fe_y.append(concentrate[2] * 100)
    K_y.append(concentrate[3] * 100)
    Mg_y.append(concentrate[4] * 100)
    Mn_y.append(concentrate[5] * 100)
    Na_y.append(concentrate[6] * 100)
    Si_y.append(concentrate[7] * 100)
    Ti_y.append(concentrate[8] * 100)
    P_y.append(concentrate[9] * 100)

for samplename, concentrate in concentrate_set_1000AVG.items():
    X.append(np.array(data_set_1000AVG[samplename].Intensity))
    waveLength.append(np.array(data_set_1000AVG[samplename].WaveLength))
    Al_y.append(concentrate[0] * 100)
    Ca_y.append(concentrate[1] * 100)
    Fe_y.append(concentrate[2] * 100)
    K_y.append(concentrate[3] * 100)
    Mg_y.append(concentrate[4] * 100)
    Mn_y.append(concentrate[5] * 100)
    Na_y.append(concentrate[6] * 100)
    Si_y.append(concentrate[7] * 100)
    Ti_y.append(concentrate[8] * 100)
    P_y.append(concentrate[9] * 100)

print('Data preprocessing finished.')

print()
print('1.4 K的实验-------------------------')



print('1.4.1 Vector Machine Regression-----------------------')

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    K_y,
                                                    test_size=0.20)
svr = SVR(C=1.0,epsilon=0.2)
svr.fit(X_train,y_train)


y_pred = svr.predict(X_test)
print('SVR for K Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print()
print('1.4.2 随机森林测试---------------------------------')

rfr = RandomForestRegressor(n_estimators=20,random_state=0)
rfr.fit(X_train,y_train)


y_pred = rfr.predict(X_test)
print('RFR for K Mean squared error is '+str(mean_squared_error(y_test,y_pred)))
#各特征的importance
importance = rfr.feature_importances_
#根据importance大小排序
indices = np.argsort(importance)[::-1]
#打印前十的importance
for i in range(0,10):
    print("importance is "+str(importance[indices[i]]))
    #print("对应的特征峰和importance为 "+str(element_dict['Al'])[indices[i]])

print()
print('1.4.3 LASSO测试---------------------------------')

lasso = Lasso(alpha =0.05, random_state=1)
lasso.fit(X_train,y_train)


y_pred = lasso.predict(X_test)
print('LASSO for K Mean squared error is '+str(mean_squared_error(y_test,y_pred)))
print("KRR TEST----------------------------------")
krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
krr.fit(X_train,y_train)
y_pred = krr.predict(X_test)
print('KRR Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print("Elastic Net TEST----------------------------------")
ENet = ElasticNet(alpha=0.05, l1_ratio=.9, random_state=3)
ENet.fit(X_train,y_train)
y_pred = ENet.predict(X_test)
print('Elastic Net Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print("Gradient Boosting TEST----------------------------------")
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
GBoost.fit(X_train,y_train)
y_pred = GBoost.predict(X_test)
print('GBoost squared error is '+str(mean_squared_error(y_test,y_pred)))

print("Bagging Experiment---------------------")
baggingModel = baggingAveragingModels(models=(krr,rfr,svr,GBoost,ENet,lasso))
baggingModel.fit(X_train,y_train)
y_pred  =baggingModel.predict(X_test)
print('Bagging squared error is '+str(mean_squared_error(y_test,y_pred)))

print("Stacking Experiment------------------------------")
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, krr,svr,rfr),
                                                 meta_model = lasso)
stacked_averaged_models.fit(X_train,y_train)
y_pred = stacked_averaged_models.predict(X_test)
print('Stacking with metamodel is lasso squared error is '+str(mean_squared_error(y_test,y_pred)))

stacked_averaged_models = StackingAveragedModels(base_models = (lasso, GBoost, krr,svr,rfr),
                                                 meta_model = ENet)
stacked_averaged_models.fit(X_train,y_train)
y_pred = stacked_averaged_models.predict(X_test)
print('Stacking with metamodel is ENet squared error is '+str(mean_squared_error(y_test,y_pred)))

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, lasso, krr,svr,rfr),
                                                 meta_model = GBoost)
stacked_averaged_models.fit(X_train,y_train)
y_pred = stacked_averaged_models.predict(X_test)
print('Stacking with metamodel is GBoost squared error is '+str(mean_squared_error(y_test,y_pred)))

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, lasso,svr,rfr),
                                                 meta_model = krr)
stacked_averaged_models.fit(X_train,y_train)
y_pred = stacked_averaged_models.predict(X_test)
print('Stacking with metamodel is krr squared error is '+str(mean_squared_error(y_test,y_pred)))


stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, krr,lasso,rfr),
                                                 meta_model = svr)
stacked_averaged_models.fit(X_train,y_train)
y_pred = stacked_averaged_models.predict(X_test)
print('Stacking with metamodel is svr squared error is '+str(mean_squared_error(y_test,y_pred)))

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, krr,svr,lasso),
                                                 meta_model = rfr)
stacked_averaged_models.fit(X_train,y_train)
y_pred = stacked_averaged_models.predict(X_test)
print('Stacking with metamodel is rfr squared error is '+str(mean_squared_error(y_test,y_pred)))

#寻找对应元素特征峰作为特征
"""
:param 寻找元素
:return 对应元素的特征序列

寻峰范围为0.5，即寻找左右0.5的max值作为特征峰
"""
def selectFeature(element):
    #取得element_dict里的波长和importance 元祖
    element_nist = element_dict[element]
    featurelist  = []
    for samplename,concentrate in concentrate_set_200AVG.items():
        templist = []
        #第一项为特征波长，第二项为importance
        for feature_list in element_dict[element]:
            temp = data_set_200AVG[samplename].loc[data_set_200AVG[samplename].WaveLength<feature_list[0]+0.5]
            feature_value = temp.loc[temp.WaveLength>feature_list[0]-0.5,'Intensity'].max()
            templist.append(feature_value)

        featurelist.append(templist)
    for samplename,concentrate in concentrate_set_1000AVG.items():
        templist = []
        #第一项为特征波长，第二项为importance
        for feature_list in element_dict[element]:
            temp = data_set_1000AVG[samplename].loc[data_set_1000AVG[samplename].WaveLength<feature_list[0]+0.5]
            feature_value = temp.loc[temp.WaveLength>feature_list[0]-0.5,'Intensity'].max()
            templist.append(feature_value)

        featurelist.append(templist)
    return featurelist

print('2.4 K的相关实验--------------------------')
K_x = selectFeature('K')

print('2.4.1 Vector Machine Regression-----------------------')

X_train, X_test, y_train, y_test = train_test_split(K_x,
                                                    K_y,
                                                    test_size=0.20)
svr = SVR(C=1.0,epsilon=0.2)
svr.fit(X_train,y_train)


y_pred = svr.predict(X_test)
print('SVR for K Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print()
print('2.4.2 随机森林测试---------------------------------')

rfr = RandomForestRegressor(n_estimators=20,random_state=0)
rfr.fit(X_train,y_train)


y_pred = rfr.predict(X_test)
print('RFR for K Mean squared error is '+str(mean_squared_error(y_test,y_pred)))
#各特征的importance
importance = rfr.feature_importances_
#根据importance大小排序
indices = np.argsort(importance)[::-1]
#打印前十的importance
for i in range(0,10):
    print("importance is "+str(importance[indices[i]]))
    print("对应的特征峰和importance为 " + str(element_dict['K'][indices[i]]))

print()
print('2.4.3 LASSO测试---------------------------------')

lasso = Lasso(alpha =0.05, random_state=1)
lasso.fit(X_train,y_train)


y_pred = lasso.predict(X_test)
print('LASSO for K Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print("KRR TEST----------------------------------")
krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
krr.fit(X_train,y_train)
y_pred = krr.predict(X_test)
print('KRR Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print("Elastic Net TEST----------------------------------")
ENet = ElasticNet(alpha=0.05, l1_ratio=.9, random_state=3)
ENet.fit(X_train,y_train)
y_pred = ENet.predict(X_test)
print('Elastic Net Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print("Gradient Boosting TEST----------------------------------")
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
GBoost.fit(X_train,y_train)
y_pred = GBoost.predict(X_test)
print('GBoost squared error is '+str(mean_squared_error(y_test,y_pred)))

print("Bagging Experiment---------------------")
baggingModel = baggingAveragingModels(models=(krr,rfr,svr,GBoost,ENet,lasso))
baggingModel.fit(X_train,y_train)
y_pred  =baggingModel.predict(X_test)
print('Bagging squared error is '+str(mean_squared_error(y_test,y_pred)))

print("Stacking Experiment------------------------------")
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, krr,svr,rfr),
                                                 meta_model = lasso)
stacked_averaged_models.fit(X_train,y_train)
y_pred = stacked_averaged_models.predict(X_test)
print('Stacking with metamodel is lasso squared error is '+str(mean_squared_error(y_test,y_pred)))

stacked_averaged_models = StackingAveragedModels(base_models = (lasso, GBoost, krr,svr,rfr),
                                                 meta_model = ENet)
stacked_averaged_models.fit(X_train,y_train)
y_pred = stacked_averaged_models.predict(X_test)
print('Stacking with metamodel is ENet squared error is '+str(mean_squared_error(y_test,y_pred)))

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, lasso, krr,svr,rfr),
                                                 meta_model = GBoost)
stacked_averaged_models.fit(X_train,y_train)
y_pred = stacked_averaged_models.predict(X_test)
print('Stacking with metamodel is GBoost squared error is '+str(mean_squared_error(y_test,y_pred)))

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, lasso,svr,rfr),
                                                 meta_model = krr)
stacked_averaged_models.fit(X_train,y_train)
y_pred = stacked_averaged_models.predict(X_test)
print('Stacking with metamodel is krr squared error is '+str(mean_squared_error(y_test,y_pred)))


stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, krr,lasso,rfr),
                                                 meta_model = svr)
stacked_averaged_models.fit(X_train,y_train)
y_pred = stacked_averaged_models.predict(X_test)
print('Stacking with metamodel is svr squared error is '+str(mean_squared_error(y_test,y_pred)))

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, krr,svr,lasso),
                                                 meta_model = rfr)
stacked_averaged_models.fit(X_train,y_train)
y_pred = stacked_averaged_models.predict(X_test)
print('Stacking with metamodel is rfr squared error is '+str(mean_squared_error(y_test,y_pred)))