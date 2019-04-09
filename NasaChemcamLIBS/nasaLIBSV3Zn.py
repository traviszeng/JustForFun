"""


将目标元素的特征区间max/dominant matrix的特征区间max 将其中的ratio作为特征
"""

import os
import sys
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.feature_selection import f_regression,SelectPercentile
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame


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

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

DATA_base_add = "E:\\NasaChemcamLIBSDataset\\"
DATA_folder_add = "E:\\NasaChemcamLIBSDataset\\Derived\\ica\\"


test_cu_line = [324.754,327.396]
test_fe_line = [248.327,248.637,252.285,302.064]
test_al_line = [309.271,308.216,309.284,394.403,396.153]
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

        #在拷贝的基本模型上进行out-of-fold预测，并用预测得到的作为meta model的feature
        out_of_fold_predictions = np.zeros((len(X), len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                #print(train_index)
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(np.array(X)[train_index], np.array(y)[train_index])
                y_pred = instance.predict(np.array(X)[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        #用out-of-foldfeature训练meta-model
        #print(type(out_of_fold_predictions))
        #print(len(y))
        self.meta_model_.fit(np.array(out_of_fold_predictions), y)
        return self

    # 使用基学习器预测测试数据，并将各基学习器预测值平均后作为meta-data feed给meta-model在做预测
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)



#加载浓度信息
def loadConcentrationData():
    con = pd.read_csv(DATA_base_add+"Concentration.csv")
    #print(con)
    return con

#标准化
def normalize(data):
    norm = (data - data.mean()) / (data.max() - data.min())
    return norm

#处理初始数据文件的一些问题
def processRawData(data):
    data = data.loc[1:6144]
    #处理一下columns的问题
    data.columns = data.columns.str.replace('#',' ')
    data.columns = data.columns.str.strip()
    return data


def loadTrainingSamples():
    for name in con.Name:
        if os.path.exists(DATA_folder_add+name.lower()):
            #print(name.lower())
            for root, dirs, files in os.walk(DATA_folder_add+name.lower()):
                for file in files:
                    #print(os.path.join(root, name))
                    if file.endswith(".csv"):
                        print("Reading files in "+os.path.join(root, file))
                        data = pd.read_csv(os.path.join(root, file),skiprows=14)
                        data = processRawData(data)
                        if name not in trainingData.keys():
                            trainingData[name] = []
                            trainingData[name].append(data)
                        else:
                            trainingData[name].append(data)
                        #print(data)

#获取每个样本的mean
def getMean():
    for key,item in trainingData.items():
        #print(key)
        sum = 0
        data = pd.DataFrame(trainingData[key][0]['wave'])
        #data['wave']  = trainingData['wave']
        for df in item:
            data['avg'+str(sum+1)] = df['mean']
            sum+=1

        meanTrainingData[key] = data

#获取目标区间的ROI
#aim为对应的特征峰
#以0.5为区间的话
def getROI(aim,samplename):
    d = meanTrainingData[samplename]
    d = d.loc[d['wave']>aim-0.25]
    d = d.loc[d['wave']<aim+0.25]
    return d

"""
准备NIST库的元素特征峰信息
放在element_dict中 key为元素名称
"""
globals = {
        'nan': 0
    }
def prepareNIST():
    print("准备NIST库相关数据\n")
    #if os.path.exists("E:\\JustForFun\\CanadaLIBSdata\\element_dict.dat"):
    print('读取NIST缓存')
    f = open("E:\\JustForFun\\CanadaLIBSdata\\element_dict.dat",'r')
    a = f.read()
    element_dict = eval(a,globals)
    f.close()
    return element_dict

def singleTrain(clf,x,y,name,time):
    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.20)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    # RFR_MSE.append(mean_squared_error(y_test, y_pred))
    print('Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")

    y_pred = clf.predict(x)
    plt.plot(y, y_pred, '.')

    x = [1,2,3,100]
    plt.plot(x,x)
    plt.xlabel('Reference Value(ppm)')
    plt.ylabel('Predict Value(ppm)')
    plt.title(name)
    plt.savefig(str(time)+name+'.png')
    plt.clf()


def useXYtrain(x,y):
    stacking_MSE1 = []
    stacking_MSE2 = []
    stacking_MSE3 = []
    stacking_MSE4 = []
    stacking_MSE5 =[]
    stacking_MSE6 = []
    for i in range(0,10):
        print('第'+str(i+1)+'次试验：\n')
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.20)

        svr = SVR(C=1.0, epsilon=0.2)
        singleTrain(svr, x, y, 'SVR', i)
        svr = SVR(C=1.0, epsilon=0.2)
        svr.fit(X_train, y_train)

        y_pred = svr.predict(X_test)
        #SVR_MSE.append(mean_squared_error(y_test, y_pred))
        print('SVR Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")

        """ann = Regressor(layers = [Layer("Sigmoid", units=14),
                                   Layer("Linear")],
                         learning_rate = 0.02,
                         random_state = 2018,
                         n_iter = 10)

        ann.fit(X_train,y_train)
        y_pred = ann.predict(X_test)
        print('ANN Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")"""


        rfr = RandomForestRegressor(n_estimators=200, random_state=0)
        singleTrain(rfr, x, y, 'RFR', i)
        rfr = RandomForestRegressor(n_estimators=200, random_state=0)
        rfr.fit(X_train, y_train)

        y_pred = rfr.predict(X_test)
        #RFR_MSE.append(mean_squared_error(y_test, y_pred))
        print('RFR Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")

        lasso = Lasso(alpha=0.05, random_state=1)
        singleTrain(lasso, x, y, 'LASSO', i)
        lasso.fit(X_train, y_train)

        y_pred = lasso.predict(X_test)
        print('LASSO  Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        #file.write('LASSO  Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")

        ENet = ElasticNet(alpha=0.05, l1_ratio=.9, random_state=3)
        singleTrain(ENet, x, y, 'Elastic NET', i)
        ENet = ElasticNet(alpha=0.05, l1_ratio=.9, random_state=3)
        ENet.fit(X_train, y_train)
        y_pred = ENet.predict(X_test)
        print('Elastic Net Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")

        GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                           max_depth=4, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10,
                                           loss='huber', random_state=5)
        singleTrain(GBoost, x, y, 'Gradient Boosting', i)
        GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                           max_depth=4, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10,
                                           loss='huber', random_state=5)
        GBoost.fit(X_train, y_train)
        y_pred = GBoost.predict(X_test)
        #GBoost_MSE.append(mean_squared_error(y_test, y_pred))
        print('GBoost squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")

        baggingModel = baggingAveragingModels(models=(rfr, svr, GBoost, ENet, lasso))
        singleTrain(baggingModel, x, y, 'Bagging', i)
        baggingModel = baggingAveragingModels(models=(rfr, svr, GBoost, ENet, lasso))

        baggingModel.fit(X_train, y_train)
        y_pred = baggingModel.predict(X_test)

        print('Bagging squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")

        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, svr, rfr),
                                                         meta_model=lasso)
        singleTrain(stacked_averaged_models, x, y, 'stacking with lasso', i)
        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, svr, rfr),
                                                         meta_model=lasso)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        print('Stacking with metamodel is lasso squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        #file.write('Stacking with metamodel is lasso squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        stacking_MSE1.append(mean_squared_error(y_test, y_pred))

        stacked_averaged_models = StackingAveragedModels(base_models=(lasso, GBoost, svr, rfr),
                                                         meta_model=ENet)
        singleTrain(stacked_averaged_models, x, y, 'stacking with ENet', i)
        stacked_averaged_models = StackingAveragedModels(base_models=(lasso, GBoost, svr, rfr),
                                                         meta_model=ENet)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        print('Stacking with metamodel is ENet squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        #file.write('Stacking with metamodel is ENet squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        stacking_MSE2.append(mean_squared_error(y_test, y_pred))

        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, lasso, svr, rfr),
                                                         meta_model=GBoost)
        singleTrain(stacked_averaged_models, x, y, 'stacking with GBoost', i)
        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, lasso, svr, rfr),
                                                         meta_model=GBoost)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        print('Stacking with metamodel is GBoost squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        #file.write('Stacking with metamodel is GBoost squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        stacking_MSE3.append(mean_squared_error(y_test, y_pred))
        krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, lasso, svr, rfr),
                                                         meta_model=krr)
        singleTrain(stacked_averaged_models, x, y, 'stacking with krr', i)
        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, lasso, svr, rfr),
                                                         meta_model=krr)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        print('Stacking with metamodel is krr squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        #file.write('Stacking with metamodel is krr squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        stacking_MSE4.append(mean_squared_error(y_test, y_pred))

        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, lasso, rfr),
                                                         meta_model=svr)
        singleTrain(stacked_averaged_models, x, y, 'stacking with svr', i)
        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, lasso, rfr),
                                                         meta_model=svr)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        print('Stacking with metamodel is svr squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        #file.write('Stacking with metamodel is svr squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        stacking_MSE5.append(mean_squared_error(y_test, y_pred))

        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, svr, lasso),
                                                         meta_model=rfr)
        singleTrain(stacked_averaged_models, x, y, 'stacking with rfr', i)
        stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, svr, lasso),
                                                         meta_model=rfr)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        print('Stacking with metamodel is rfr squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        #file.write('Stacking with metamodel is rfr squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        stacking_MSE6.append(mean_squared_error(y_test, y_pred))

    plot_x = np.linspace(1, 10, 10)
    plt.plot(plot_x,stacking_MSE1,'b')
    plt.plot(plot_x, stacking_MSE2, 'r')
    plt.plot(plot_x, stacking_MSE3, 'y')
    plt.plot(plot_x, stacking_MSE4, 'k')
    plt.plot(plot_x, stacking_MSE5, 'g')
    plt.plot(plot_x, stacking_MSE6, 'm')
    plt.legend(('Lasso avg='+str(np.mean(stacking_MSE1)),
                'Enet avg='+str(np.mean(stacking_MSE2)),
                'Gboost avg = '+str(np.mean(stacking_MSE3)),
                'KRR avg = '+str(np.mean(stacking_MSE4)),
                'SVR avg = '+str(np.mean(stacking_MSE5)),
                'RFR avg = '+str(np.mean(stacking_MSE6))), loc='upper right')
    plt.title('Different meta-learning machine')
    plt.savefig('DifferentLearner.png')
    plt.clf()


    plt.plot()




#def test(element):
    #trainingCase = {}
    #element_info = element_dict[element]
"""
    for info in element_info:
        print('Using wave='+str(info[0])+" importance="+str(info[1])+" for testing...")

        aim = info[0]
        if aim<240.86501 or aim>905.57349 or aim-0.25<240.86501 or aim+0.25>905.57349:
            continue

        for key, item in meanTrainingData.items():
            for name in meanTrainingData[key].columns:
                if not name == 'wave':
                    meanTrainingData[key][name] = normalize(meanTrainingData[key][name])

            trainingCase[key] = getROI(aim, key)



        x = []
        y = []
        for key, item in trainingCase.items():
            # print(key)
            for name in trainingCase[key].columns:
                # print(name)
                if not name == 'wave':
                    x.append(list(trainingCase[key][name]))
                    y.append(con.loc[con.Name == key].Cu.values[0])
        if len(x)==len(y) and len(x)!=0 and len(x[0])!=0:
            useXYtrain(x,y)
"""

def newMain(element):
    x = []
    y = []

    for key,val in meanTrainingData.items():
        for name in ['avg1','avg2','avg3','avg4','avg5']:
            xtemp = []
            for culine in test_cu_line:
                for feline in test_fe_line:
                    try:
                        xtemp.append(max(getROI(culine,key)[name])/max(getROI(feline,key)[name]))
                    except ValueError:
                        print('No line in '+str(culine))
                        pass

            x.append(xtemp)
            y.append(con.loc[con.Name==key][element].values[0])

    #useXYtrain(x,y)
    return x,y

def processingRatios():
    xy = x_df.join(y_df)
    #处理异常值
    xy = xy.loc[xy.feature4 > 0]
    xy = xy.loc[xy.feature4 < 4]
    xy = xy.loc[xy.Target<100]
    xy_0 = xy[xy.Target <= 0]
    xy = xy[xy.Target>0]
    xy.append(xy_0.mean(), ignore_index=True)

    newx = []
    newy = []



    for i in range(0, 320):
        try:
            newx.append(xy.loc[i].tolist()[:-1])
            newy.append(xy.loc[i].tolist()[-1])
        except KeyError:
            pass

    return newx,newy


if __name__=='__main__':
    con = loadConcentrationData()
    trainingData = {}
    loadTrainingSamples()
    meanTrainingData = {}
    getMean()
    print(meanTrainingData)
    # element_dict = prepareNIST()
    element = 'Zn'
    x, y = newMain(element)
    os.chdir("E:\\JustForFun\\NasaChemcamLIBS\\" + element)
    y_df = pd.DataFrame(y, columns=['Target'])
    x_df = pd.DataFrame(x, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7',
                                    'feature8'])
    processingRatios()
    # test(element)
    newx, newy = processingRatios()
    useXYtrain(newx, newy)

