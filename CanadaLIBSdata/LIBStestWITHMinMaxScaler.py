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
from sklearn.feature_selection import f_regression, SelectPercentile
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler

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


def loadConcentrateFile():
    print("Loading concentrate file.....")
    concentrate_data = pd.read_csv("E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\Sample_Composition_Data.csv")
    # 前81行为数据
    concentrate_data = concentrate_data.loc[0:81]
    return concentrate_data


print('Data preprocessing begins.')
"""
数据预处理流程：
1.填补空白值
2.处理异常值，Nan value处理
3.str转float
"""


def dataPreprocessing(concentrate_data):
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
    return concentrate_data


route_200_AVG = "E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\csv Material Large Set 200pulseaverage"
route_1000_AVG = "E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\csv Certified Samples Subset 1000pulseaverage"

postfix_200AVG = "_200AVG.csv"
postfix_1000AVG = "_1000AVG.csv"

"""
加载训练样本
"""
data_set_200AVG = {}
concentrate_set_200AVG = {}


def load200AVGTrainingFiles(concentrate_data):
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


def load1000AVGtrainingFiles(concentrate_data):
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


def prepareData():
    concentrate_data = loadConcentrateFile()
    concentrate_data = dataPreprocessing(concentrate_data)
    load200AVGTrainingFiles(concentrate_data)
    load1000AVGtrainingFiles(concentrate_data)
    # 去掉hand sample34的记录（全为0）
    del concentrate_set_200AVG['hand sample37 barite_200AVG']
    del data_set_200AVG['hand sample37 barite_200AVG']


# 画一个折线图尝试看看
"""
s_name = "RockFer2_200AVG"
intensity = data_set_200AVG[s_name].Intensity
wavelength = data_set_200AVG[s_name].WaveLength
intensity = np.array(intensity)
wavelength = np.array(wavelength)
plt.plot(wavelength,intensity)
plt.show()
"""
element_dict = {}


def prepareNIST():
    print("准备NIST库相关数据")
    nist = pd.read_csv("E:\\JustForFun\\CanadaLIBSdata\\andor.nist", header=None,
                       names=['WaveLength', 'Element', 'Type', 'Unknown', 'Importance'])
    nist = nist.loc[1:]
    # 删除未知列
    del nist['Unknown']
    # 筛选在样本精度范围的nist线
    nist = nist.loc[nist.WaveLength >= 198.066]
    nist = nist.loc[nist.WaveLength <= 970.142]

    for indexs in nist.index:
        if nist.loc[indexs].Element in element_dict:
            if [nist.loc[indexs].WaveLength, nist.loc[indexs].Importance] not in element_dict[nist.loc[indexs].Element]:
                element_dict[nist.loc[indexs].Element].append(
                    [nist.loc[indexs].WaveLength, nist.loc[indexs].Importance])
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
# 相关浓度信息
# 元素顺序Al Ca Fe K Mg Mn Na Si Ti P
y = [[], [], [], [], [], [], [], [], [], []]


# 准备训练数据
def prepareTrainingXY():
    for samplename, concentrate in concentrate_set_200AVG.items():
        X.append(np.array(data_set_200AVG[samplename].Intensity))
        waveLength.append(np.array(data_set_200AVG[samplename].WaveLength))
        y[0].append(concentrate[0] * 100)
        y[1].append(concentrate[1] * 100)
        y[2].append(concentrate[2] * 100)
        y[3].append(concentrate[3] * 100)
        y[4].append(concentrate[4] * 100)
        y[5].append(concentrate[5] * 100)
        y[6].append(concentrate[6] * 100)
        y[7].append(concentrate[7] * 100)
        y[8].append(concentrate[8] * 100)
        y[9].append(concentrate[9] * 100)

    for samplename, concentrate in concentrate_set_1000AVG.items():
        X.append(np.array(data_set_1000AVG[samplename].Intensity))
        waveLength.append(np.array(data_set_1000AVG[samplename].WaveLength))
        y[0].append(concentrate[0] * 100)
        y[1].append(concentrate[1] * 100)
        y[2].append(concentrate[2] * 100)
        y[3].append(concentrate[3] * 100)
        y[4].append(concentrate[4] * 100)
        y[5].append(concentrate[5] * 100)
        y[6].append(concentrate[6] * 100)
        y[7].append(concentrate[7] * 100)
        y[8].append(concentrate[8] * 100)
        y[9].append(concentrate[9] * 100)


# 寻找对应元素特征峰作为特征
"""
:param 寻找元素
:return 对应元素的特征序列

寻峰范围为0.5，即寻找左右0.2的max值作为特征峰
"""
"""
添加特征缩放进行实验
"""

def selectFeature(element):
    # 取得element_dict里的波长和importance 元祖
    element_nist = element_dict[element]
    featurelist = []
    for samplename, concentrate in concentrate_set_200AVG.items():
        templist = []
        # 第一项为特征波长，第二项为importance
        for feature_list in element_dict[element]:
            temp = data_set_200AVG[samplename].loc[data_set_200AVG[samplename].WaveLength < feature_list[0] + 0.2]
            feature_value = temp.loc[temp.WaveLength > feature_list[0] - 0.2, 'Intensity'].max()
            templist.append(feature_value)

        featurelist.append(templist)
    for samplename, concentrate in concentrate_set_1000AVG.items():
        templist = []
        # 第一项为特征波长，第二项为importance
        for feature_list in element_dict[element]:
            temp = data_set_1000AVG[samplename].loc[data_set_1000AVG[samplename].WaveLength < feature_list[0] + 0.2]
            feature_value = temp.loc[temp.WaveLength > feature_list[0] - 0.2, 'Intensity'].max()
            templist.append(feature_value)

        featurelist.append(templist)

    # 处理一些元素中存在nan强度的问题
    featurelist = DataFrame(featurelist)
    featurelist = featurelist.fillna(0)
    featurelist = featurelist.values.tolist()

    #标准化
    ss = StandardScaler()
    ss.fit(featurelist)
    featurelist = ss.transform(featurelist)
    return featurelist


print('Data preprocessing finished.')
print()
print('1.未处理X实验，输入整个光谱------------------------------')
print()
extract_element_dict = {}
"""
压缩特征
update 20190123
使用selectPercentile选择相关度高的特征
"""


def compressFeature(element, x, y, compressRate=10):
    oldX = copy.deepcopy(x)

    x = np.array(x)
    y = np.array(y)
    x = SelectPercentile(f_regression, percentile=compressRate).fit_transform(x, y)
    originalfeature_indice = []
    for i in range(0, len(x[0])):
        indice = \
        np.where((oldX[0] == x[0][i]) & (oldX[5] == x[5][i]) & (oldX[10] == x[10][i]) & (oldX[15] == x[15][i]))[0]
        for j in range(0, len(indice)):
            if indice[j] not in originalfeature_indice:
                originalfeature_indice.append(indice[j])

    print(originalfeature_indice)

    for f_indice in originalfeature_indice:
        if element in extract_element_dict:
            extract_element_dict[element].append(element_dict[element][f_indice])
        else:
            extract_element_dict[element] = [element_dict[element][f_indice]]

    return x, y


"""
:param element 需要训练的元素名称 str
:param X 训练使用的X
:param y 训练使用的y
:param flag 用来区分输入的X是全光谱还是特征峰，用来区分是否需要输出rfr的importance 对应的特征峰波长
:param featureCompressRate 特征压缩比
:param times 重复次数
"""
import copy


def elementTest(element, x, y, flag, featureCompressRate, times=10):
    SVR_MSE = []
    RFR_MSE = []
    LASSO_MSE = []
    GBoost_MSE = []
    ENet_MSE = []
    # KRR_MSE = []
    stacking_MSE = []
    bagging_MSE = []

    x, y = compressFeature(element, x, y, featureCompressRate)

    for i in range(0, 10):
        print()
        print("第" + str(i + 1) + "次" + str(element) + "的实验----------------------")
        # 特征选择，先选取前10%的特征进行训练


        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.20)
        print("Part 1 Experiment with Support Vector Machine Regression-------------")
        svr = SVR(C=1.0, epsilon=0.2)
        svr.fit(X_train, y_train)

        y_pred = svr.predict(X_test)
        SVR_MSE.append(mean_squared_error(y_test, y_pred))
        print('SVR Mean squared error is ' + str(mean_squared_error(y_test, y_pred)))

        print()
        print('Part 2 Experiment with Random forest regression---------------------------------')

        rfr = RandomForestRegressor(n_estimators=200, random_state=0)
        rfr.fit(X_train, y_train)

        y_pred = rfr.predict(X_test)
        RFR_MSE.append(mean_squared_error(y_test, y_pred))
        print('RFR Mean squared error is ' + str(mean_squared_error(y_test, y_pred)))
        # 各特征的importance
        importance = rfr.feature_importances_
        # print(importance)
        # 根据importance大小排序
        indices = np.argsort(importance)[::-1]
        # print(indices)
        if len(x[0]) < 10:
            # 打印前十的importance
            for i in range(0, len(x[0])):
                print("importance is " + str(importance[indices[i]]))
                if flag:
                    print("对应的特征峰和importance为 " + str(extract_element_dict[element][indices[i]]))
        else:
            # 打印前十的importance
            for i in range(0, 10):
                print("importance is " + str(importance[indices[i]]))
                if flag:
                    print("对应的特征峰和importance为 " + str(extract_element_dict[element][indices[i]]))

        print()
        print('Part 3 LASSO experiment ---------------------------------')

        lasso = Lasso(alpha=0.05, random_state=1)
        lasso.fit(X_train, y_train)

        y_pred = lasso.predict(X_test)
        print('LASSO  Mean squared error is ' + str(mean_squared_error(y_test, y_pred)))
        LASSO_MSE.append(mean_squared_error(y_test, y_pred))
        print("Part 4 KRR TEST----------------------------------")
        """
        krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
        krr.fit(X_train, y_train)
        y_pred = krr.predict(X_test)
        KRR_MSE.append(mean_squared_error(y_test, y_pred))

        print('KRR Mean squared error is ' + str(mean_squared_error(y_test, y_pred)))
        """
        print("Part 5 Elastic Net TEST----------------------------------")
        ENet = ElasticNet(alpha=0.05, l1_ratio=.9, random_state=3)
        ENet.fit(X_train, y_train)
        y_pred = ENet.predict(X_test)
        print('Elastic Net Mean squared error is ' + str(mean_squared_error(y_test, y_pred)))
        ENet_MSE.append(mean_squared_error(y_test, y_pred))
        print("Part 6 Gradient Boosting TEST----------------------------------")
        GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                           max_depth=4, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10,
                                           loss='huber', random_state=5)
        GBoost.fit(X_train, y_train)
        y_pred = GBoost.predict(X_test)
        GBoost_MSE.append(mean_squared_error(y_test, y_pred))
        print('GBoost squared error is ' + str(mean_squared_error(y_test, y_pred)))

        print("Part 7 Bagging Experiment---------------------")

        baggingModel = baggingAveragingModels(models=(rfr, svr, GBoost, ENet, lasso))
        baggingModel.fit(X_train, y_train)
        y_pred = baggingModel.predict(X_test)
        bagging_MSE.append(mean_squared_error(y_test, y_pred))

        print('Bagging squared error is ' + str(mean_squared_error(y_test, y_pred)))

        print("Part 8 Stacking Experiment------------------------------")
        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, svr, rfr),
                                                         meta_model=lasso)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        print('Stacking with metamodel is lasso squared error is ' + str(mean_squared_error(y_test, y_pred)))
        stacking_MSE.append(mean_squared_error(y_test, y_pred))

        stacked_averaged_models = StackingAveragedModels(base_models=(lasso, GBoost, svr, rfr),
                                                         meta_model=ENet)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        print('Stacking with metamodel is ENet squared error is ' + str(mean_squared_error(y_test, y_pred)))
        stacking_MSE.append(mean_squared_error(y_test, y_pred))

        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, lasso, svr, rfr),
                                                         meta_model=GBoost)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        print('Stacking with metamodel is GBoost squared error is ' + str(mean_squared_error(y_test, y_pred)))
        stacking_MSE.append(mean_squared_error(y_test, y_pred))
        krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, lasso, svr, rfr),
                                                         meta_model=krr)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        print('Stacking with metamodel is krr squared error is ' + str(mean_squared_error(y_test, y_pred)))
        stacking_MSE.append(mean_squared_error(y_test, y_pred))

        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, lasso, rfr),
                                                         meta_model=svr)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        print('Stacking with metamodel is svr squared error is ' + str(mean_squared_error(y_test, y_pred)))
        stacking_MSE.append(mean_squared_error(y_test, y_pred))

        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, svr, lasso),
                                                         meta_model=rfr)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        print('Stacking with metamodel is rfr squared error is ' + str(mean_squared_error(y_test, y_pred)))
        stacking_MSE.append(mean_squared_error(y_test, y_pred))

    plot_x = np.linspace(1, 10, 10)
    plt.plot(plot_x, SVR_MSE, 'b', label='SVR')
    plt.plot(plot_x, RFR_MSE, 'r', label='RFR')
    plt.plot(plot_x, LASSO_MSE, 'g', label='LASSO')
    plt.plot(plot_x, ENet_MSE, 'y', label='ENet')

    # plt.plot(plot_x, KRR_MSE, 'k',label ='KRR')
    plt.plot(plot_x, bagging_MSE, 'M', label='Bagging')
    plt.plot(plot_x, GBoost_MSE, 'c', label='GBoost')
    plt.legend(['SVR', 'RFR', 'LASSO', 'ENet', 'Bagging', 'GBoost'])
    plt.title(element + " bagging vs other learner")
    plt.savefig(element + "with compressRate = " + str(featureCompressRate) + ".png")
    plt.clf()
    # plt.show()

    sub = stacking_MSE[0::6]
    plt.plot(plot_x, sub, 'b', label='meta is lasso')
    sub = stacking_MSE[1::6]
    plt.plot(plot_x, sub, 'r', label='meta is ENet')
    sub = stacking_MSE[2::6]
    plt.plot(plot_x, sub, 'g', label='meta is GBoost')
    sub = stacking_MSE[3::6]
    plt.plot(plot_x, sub, 'y', label='meta is krr')
    sub = stacking_MSE[4::6]
    plt.plot(plot_x, sub, 'k', label='meta is svr')
    sub = stacking_MSE[5::6]
    plt.plot(plot_x, sub, 'm', label='meta is rfr')
    plt.legend(['meta is lasso', 'meta is ENet', 'meta is GBoost', 'meta is krr', 'meta is svr', 'meta is rfr'])
    plt.title(element + " Stacking comparison")
    plt.savefig(element + " stacking with compressRate = " + str(featureCompressRate) + ".png")
    plt.clf()
    # plt.show()


    print(str(np.average(SVR_MSE)))
    print(str(np.average(RFR_MSE)))
    print(str(np.average(LASSO_MSE)))

    print(str(np.average(ENet_MSE)))
    print(str(np.average(GBoost_MSE)))
    print(str(np.average(bagging_MSE)))
    return SVR_MSE, RFR_MSE, LASSO_MSE, ENet_MSE, GBoost_MSE, bagging_MSE, stacking_MSE


"""
elementTest('Al',X,Al_y,0)
elementTest('Ca',X,Ca_y,0)
elementTest('Fe',X,Fe_y,0)
elementTest('K',X,K_y,0)
elementTest('Mg',X,Mg_y,0)
elementTest('Mn',X,Mn_y,0)
elementTest('Na',X,Na_y,0)
elementTest('Si',X,Si_y,0)
elementTest('Ti',X,Ti_y,0)
elementTest('P',X,P_y,0)
"""
print('2.根据NIST库筛选特征-------------------------')


def main(element, index):
    x = selectFeature(element)
    svr_avg = []
    rfr_avg = []
    lasso_avg = []
    enet_avg = []
    gboost_avg = []
    bag_avg = []

    for i in range(1, 11):
        SVR_MSE, RFR_MSE, LASSO_MSE, ENet_MSE, GBoost_MSE, bagging_MSE, stacking_MSE = elementTest(element, x, y[index],
                                                                                                   1,
                                                                                                   i)
        svr_avg.append(np.average(SVR_MSE))
        rfr_avg.append(np.average(RFR_MSE))
        lasso_avg.append(np.average(LASSO_MSE))
        enet_avg.append(np.average(ENet_MSE))
        gboost_avg.append(np.average(GBoost_MSE))
        bag_avg.append(np.average(bagging_MSE))

    plot_x = np.linspace(1, 10, 10)
    plt.subplot(2, 3, 1)
    plt.plot(plot_x, svr_avg)
    plt.subplot(2, 3, 2)
    plt.plot(plot_x, rfr_avg)
    plt.subplot(2, 3, 3)
    plt.plot(plot_x, lasso_avg)
    plt.subplot(2, 3, 4)
    plt.plot(plot_x, enet_avg)
    plt.subplot(2, 3, 5)
    plt.plot(plot_x, gboost_avg)
    plt.subplot(2, 3, 6)
    plt.plot(plot_x, bag_avg)
    plt.savefig(element + ' with different compress rate.png')
    plt.clf()


# 元素顺序Al Ca Fe K Mg Mn Na Si Ti P
y = [[], [], [], [], [], [], [], [], [], []]
if __name__ == '__main__':
    prepareData()
    prepareNIST()
    prepareTrainingXY()
    os.chdir('E:\\JustForFun\\CanadaLIBSdata\\testWithStandardScaler')
    elementList = ['Al', 'Ca', 'Fe', 'K', 'Mg', 'Mn', 'Na', 'Si', 'Ti']
    for i in range(0, len(elementList)):
        main(elementList[i], i)

