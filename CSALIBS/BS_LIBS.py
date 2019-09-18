"""
利用加拿大航天局的LIBS数据进行LIBS定量分析实验
"""
import os
import sys
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor,AdaBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.feature_selection import f_regression,SelectPercentile
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import gc
from mlxtend.regressor import StackingRegressor



import xgboost as xgb
import lightgbm as lgb
import copy
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

import warnings
warnings.filterwarnings("ignore")

route_200_AVG = "E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\csv Material Large Set 200pulseaverage"
route_1000_AVG = "E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\csv Certified Samples Subset 1000pulseaverage"

postfix_200AVG ="_200AVG.csv"
postfix_1000AVG = "_1000AVG.csv"

def loadConcentrateFile():
    print("Loading concentrate file.....\n")
    concentrate_data = pd.read_csv("E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\Sample_Composition_Data.csv")
    #前81行为数据
    concentrate_data = concentrate_data.loc[0:81]
    return concentrate_data

def drawTrain(y_pred,y_test,name,time):

    #clf.fit(X_train,y_train)
    #y_pred = clf.predict(X_test)
    # RFR_MSE.append(mean_squared_error(y_test, y_pred))
    #print('Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")

    #y_pred = clf.predict(X_test)
    plt.plot(y_test, y_pred, '.')
    maxx = max(y_test)
    if max(y_pred)>maxx:
        maxx = max(y_pred)
    xx = [1,2,3,maxx]
    plt.plot(xx,xx)
    plt.xlabel('Reference Value(%)')
    plt.ylabel('Predict Value(%)')
    plt.title(name)
    plt.savefig(str(time)+name+'.png')
    plt.clf()

"""
数据预处理流程：
1.填补空白值
2.处理异常值，Nan value处理
3.str转float
"""
def dataPreprocessing(concentrate_data):
    #数据清洗
    for indexs in concentrate_data.index:
        for i in range(1,12):
            if concentrate_data.loc[indexs].values[i]=='-':
                concentrate_data.loc[indexs].values[i] = 0.0

            else:
                try:
                    concentrate_data.loc[indexs].values[i] = float(concentrate_data.loc[indexs].values[i])
                    if float(concentrate_data.loc[indexs].values[i])>1:
                        concentrate_data.loc[indexs].values[i] = concentrate_data.loc[indexs].values[i]/100
                except ValueError:
                    concentrate_data.loc[indexs].values[i] = float(concentrate_data.loc[indexs].values[i][1:])
                    if float(concentrate_data.loc[indexs].values[i])>1:
                        concentrate_data.loc[indexs].values[i] = concentrate_data.loc[indexs].values[i]/100

    #检查是否将所有非数字处理好
    for column in concentrate_data.columns:
        print(concentrate_data[column].isna().value_counts())
    return concentrate_data


"""
加载训练样本
"""
data_set_200AVG = {}
concentrate_set_200AVG = {}
def load200AVGTrainingFiles(concentrate_data):

    #加载200AVG的样本，并将其存到data_set_200AVG中
    os.chdir(route_200_AVG)
    num = 0
    for indexs in concentrate_data.index:
        if os.path.exists(concentrate_data.loc[indexs].values[0]+postfix_200AVG):
            num+=1
            print("Get data file:"+concentrate_data.loc[indexs].values[0]+postfix_200AVG)
            data = pd.read_csv(concentrate_data.loc[indexs].values[0]+postfix_200AVG,header = None,names = ['WaveLength','Intensity'])
            #data中强度<0的统统变为0
            data.loc[data.Intensity < 0, 'Intensity'] = 0
            data_set_200AVG[concentrate_data.loc[indexs].values[0]+"_200AVG"] = data
            concentrate_set_200AVG[concentrate_data.loc[indexs].values[0]+"_200AVG"] = concentrate_data.loc[indexs].values[1:]
        #处理hand sample类型的样本
        if re.match('hand sample*',concentrate_data.loc[indexs].values[0]):

            f_list = concentrate_data.loc[indexs].values[0].split()
            filename = f_list[0]+" "+f_list[1]+postfix_200AVG
            if os.path.exists(filename):
                num+=1
                print("Get data file:"+filename)
                data = pd.read_csv(filename,header = None,names = ['WaveLength','Intensity'])
                # data中强度<0的统统变为0
                data.loc[data.Intensity < 0, 'Intensity'] = 0
                data_set_200AVG[concentrate_data.loc[indexs].values[0]+"_200AVG"] = data
                concentrate_set_200AVG[concentrate_data.loc[indexs].values[0]+"_200AVG"] = concentrate_data.loc[indexs].values[1:]


    print("Get "+str(num)+" 200_AVG files.\n")
    print()

data_set_1000AVG = {}
concentrate_set_1000AVG = {}
def load1000AVGtrainingFiles(concentrate_data):
    num = 0
    #加载1000AVG的样本，并将其存到data_set_1000AVG中
    os.chdir(route_1000_AVG)
    for indexs in concentrate_data.index:
        if os.path.exists(concentrate_data.loc[indexs].values[0]+postfix_1000AVG):
            num+=1
            print("Get data file:"+concentrate_data.loc[indexs].values[0]+postfix_1000AVG)
            data = pd.read_csv(concentrate_data.loc[indexs].values[0]+postfix_1000AVG,header = None,names = ['WaveLength','Intensity'])
            # data中强度<0的统统变为0
            data.loc[data.Intensity < 0, 'Intensity'] = 0
            data_set_1000AVG[concentrate_data.loc[indexs].values[0]+"_1000AVG"] = data
            concentrate_set_1000AVG[concentrate_data.loc[indexs].values[0]+"_1000AVG"] = concentrate_data.loc[indexs].values[1:]
        #处理hand sample类型的样本
        if re.match('hand sample*',concentrate_data.loc[indexs].values[0]):

            f_list = concentrate_data.loc[indexs].values[0].split()
            filename = f_list[0]+" "+f_list[1]+postfix_1000AVG
            if os.path.exists(filename):
                num+=1
                print("Get data file:"+filename)
                data = pd.read_csv(filename,header = None,names = ['WaveLength','Intensity'])
                # data中强度<0的统统变为0
                data.loc[data.Intensity < 0, 'Intensity'] = 0
                data_set_1000AVG[concentrate_data.loc[indexs].values[0]+"_1000AVG"] = data
                concentrate_set_1000AVG[concentrate_data.loc[indexs].values[0]+"_1000AVG"] = concentrate_data.loc[indexs].values[1:]

    print("Get "+str(num)+" 1000_AVG files.\n")


"""
准备concentration相关数据
"""
def prepareConcentrationData():
    concentrate_data = loadConcentrateFile()
    concentrate_data = dataPreprocessing(concentrate_data)
    load200AVGTrainingFiles(concentrate_data)
    load1000AVGtrainingFiles(concentrate_data)
    # 去掉hand sample34的记录（全为0）
    del concentrate_set_200AVG['hand sample37 barite_200AVG']
    del data_set_200AVG['hand sample37 barite_200AVG']


#获得特征峰左右0.25的特征点
def getROI(dataset,samplename,aim):
    d = dataset[samplename]
    d = d.loc[d['WaveLength'] > aim - 0.25]
    d = d.loc[d['WaveLength'] < aim + 0.25]
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

element_dict = prepareNIST()
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


""""#准备训练数据，未作任何处理，将整个光谱输入
X = []
#波长
waveLength = []
#相关浓度信息
#元素顺序Al Ca Fe K Mg Mn Na Si Ti P
y = [[],[],[],[],[],[],[],[],[],[]]
#准备训练数据
def prepareTrainingXY():
    for samplename,concentrate in concentrate_set_200AVG.items():
        X.append(np.array(data_set_200AVG[samplename].Intensity))
        waveLength.append(np.array(data_set_200AVG[samplename].WaveLength))
        y[0].append(concentrate[0]*100)
        y[1].append(concentrate[1]*100)
        y[2].append(concentrate[2]*100)
        y[3].append(concentrate[3]*100)
        y[4].append(concentrate[4]*100)
        y[5].append(concentrate[5]*100)
        y[6].append(concentrate[6]*100)
        y[7].append(concentrate[7]*100)
        y[8].append(concentrate[8]*100)
        y[9].append(concentrate[9]*100)

    for samplename,concentrate in concentrate_set_1000AVG.items():
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
"""

test_al_line = [309.271,308.216,309.284,394.403,396.153]
test_fe_line = [248.327,248.637,252.285,302.064]
def newMain():
    x = []
    y = []
    for samplename, concentrate in concentrate_set_200AVG.items():
        xtemp = []
        for line in test_al_line:

            try:
                #print(max(getROI(data_set_200AVG,samplename,culine)['WaveLength']))
                #print(max(getROI(data_set_200AVG,samplename,feline)['WaveLength']))
                xtemp.append(max(getROI(data_set_200AVG,samplename,line)['Intensity'])/max(data_set_200AVG[samplename]['Intensity']))
            except ValueError:
                print('No line in '+str(line))
                pass

        x.append(xtemp)
        y.append(concentrate[0]*100)

    for samplename, concentrate in concentrate_set_1000AVG.items():
        xtemp = []
        for line in test_al_line:

            try:
                #print(max(getROI(data_set_200AVG, samplename, culine)['WaveLength']))
                #print(max(getROI(data_set_200AVG, samplename, feline)['WaveLength']))
                xtemp.append(max(getROI(data_set_1000AVG, samplename, line)['Intensity']) / max(data_set_1000AVG[samplename]['Intensity']))
            except ValueError:
                print('No line in ' + str(line))
                pass

        x.append(xtemp)
        y.append(concentrate[0] * 100)



    #useXYtrain(x,y)
    return x,y


def processingRatios():
    xy = x_df.join(y_df)
    # 处理异常值
    xy = xy.loc[xy.feature4 > 0]
    xy = xy.loc[xy.feature4 < 4]
    xy = xy.loc[xy.Target < 100]
    xy_0 = xy[xy.Target <= 0]
    xy = xy[xy.Target > 0]
    xy.append(xy_0.mean(), ignore_index=True)

    newx = []
    newy = []

    print("行数为" + str(xy.shape[0]))
    for i in range(0, xy.shape[0]):
        try:
            newx.append(xy.loc[i].tolist()[:-1])
            newy.append(xy.loc[i].tolist()[-1])
        except KeyError:
            pass

    return newx, newy


# 学习器选择 设定一个阈值 MSE低于该阈值的学习器认为适合
MSE_bar = 250
Selected_learnerCode = []


def selectLearner(x, y):
    SVR_MSE = []
    RFR_MSE = []
    GBOOST_MSE = []
    ENET_MSE = []
    LASSO_MSE = []

    for i in range(0, 10):
        print('第' + str(i + 1) + '次试验：\n')
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.20)

        # svr = SVR(C=1.0, epsilon=0.2)
        # drawTrain(svr, x, y, 'SVR', i)
        svr = SVR(C=1.0, epsilon=0.1, kernel='rbf')
        parameters = {'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-3, 3, 7)}
        print("GridSearch starting...")
        clf = GridSearchCV(svr, parameters, n_jobs=-1, scoring='neg_mean_squared_error')
        clf.fit(X_train, y_train)

        print('The parameters of the best model are: ')
        print(clf.best_params_)
        y_pred = clf.best_estimator_.predict(X_test)
        SVR_MSE.append(mean_squared_error(y_test, y_pred))
        print('SVR Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        del clf, svr
        """ann = Regressor(layers = [Layer("Sigmoid", units=14),
                                   Layer("Linear")],
                         learning_rate = 0.02,
                         random_state = 2018,
                         n_iter = 10)

        ann.fit(X_train,y_train)
        y_pred = ann.predict(X_test)
        print('ANN Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")"""

        parameters = {'n_estimators': [10, 50, 100, 500, 1000]}
        rfr = RandomForestRegressor(n_estimators=200, random_state=0)
        # drawTrain(rfr, x, y, 'RFR', i)
        # rfr = RandomForestRegressor(n_estimators=200, random_state=0)
        clf = GridSearchCV(rfr, parameters, n_jobs=-1, scoring='neg_mean_squared_error')
        clf.fit(X_train, y_train)
        print('The parameters of the best model are: ')
        print(clf.best_params_)
        y_pred = clf.best_estimator_.predict(X_test)
        RFR_MSE.append(mean_squared_error(y_test, y_pred))
        print('RFR Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        del clf, rfr

        parameters = {'alpha': np.logspace(-2, 2, 5)}
        lasso = Lasso(alpha=0.05, random_state=1, max_iter=1000)
        # drawTrain(lasso, x, y, 'LASSO', i)
        clf = GridSearchCV(lasso, parameters, n_jobs=-1, scoring='neg_mean_squared_error')
        clf.fit(X_train, y_train)
        print('The parameters of the best model are: ')
        print(clf.best_params_)

        y_pred = clf.best_estimator_.predict(X_test)
        LASSO_MSE.append(mean_squared_error(y_test, y_pred))
        print('LASSO  Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        # file.write('LASSO  Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        del clf, lasso

        parameters = {'alpha': np.logspace(-2, 2, 5)}
        # ENet = ElasticNet(alpha=0.05, l1_ratio=.9, random_state=3)
        # drawTrain(ENet, x, y, 'Elastic NET', i)
        ENet = ElasticNet(alpha=0.05, l1_ratio=.9, random_state=3)
        clf = GridSearchCV(ENet, parameters, n_jobs=-1, scoring='neg_mean_squared_error')
        clf.fit(X_train, y_train)
        print('The parameters of the best model are: ')
        print(clf.best_params_)
        y_pred = clf.best_estimator_.predict(X_test)
        ENET_MSE.append(mean_squared_error(y_test, y_pred))
        print('Elastic Net Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        del clf, ENet
        """GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                           max_depth=4, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10,
                                           loss='huber', random_state=5)
        drawTrain(GBoost, x, y, 'Gradient Boosting', i)"""
        parameters = {'n_estimators': [100, 500, 1000, 2000, 3000, 5000]}
        GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                           max_depth=4, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10,
                                           loss='huber', random_state=5)
        clf = GridSearchCV(GBoost, parameters, n_jobs=-1, scoring='neg_mean_squared_error')
        clf.fit(X_train, y_train)
        y_pred = clf.best_estimator_.predict(X_test)
        print(clf.best_params_)
        GBOOST_MSE.append(mean_squared_error(y_test, y_pred))
        print('GBoost squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        del clf, GBoost

    if sum(SVR_MSE) / 10 <= MSE_bar:
        Selected_learnerCode.append('SVR')
    else:
        Selected_learnerCode.append('')

    if sum(RFR_MSE) / 10 <= MSE_bar:
        Selected_learnerCode.append('RFR')
    else:
        Selected_learnerCode.append('')

    if sum(ENET_MSE) / 10 <= MSE_bar:
        Selected_learnerCode.append('ENET')
    else:
        Selected_learnerCode.append('')

    if sum(LASSO_MSE) / 10 <= MSE_bar:
        Selected_learnerCode.append('LASSO')
    else:
        Selected_learnerCode.append('')

    if sum(GBOOST_MSE) / 10 <= MSE_bar:
        Selected_learnerCode.append('GBOOST')
    else:
        Selected_learnerCode.append('')

    print(GBOOST_MSE)


def useXYtrain(x,y,times):
    flag = 0
    for  i in range(0,len(Selected_learnerCode)):
        if Selected_learnerCode[i]!='':
            flag+=1
    if flag==0:
        print('No proper learner\n')
        return
    stacking_MSE = [[],[],[],[],[],[]]
    MSE = [[],[],[],[],[],[],[]]

    Ada_MSE = []

    for i in range(0,times):
        print('第'+str(i+1)+'次试验：\n')
        Learners_map = {}
        Learners = []
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.20)
        svr = SVR(C=1.0, epsilon=0.2)
        parameters = { 'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-3, 3, 7)}
        print("GridSearch starting...")
        clfsvr = GridSearchCV(svr, parameters, n_jobs=-1, scoring='neg_mean_squared_error')
        clfsvr.fit(X_train, y_train)

        print('The parameters of the best model are: ')
        print(clfsvr.best_params_)
        y_pred = clfsvr.best_estimator_.predict(X_test)
        #drawTrain(y_pred, y_test, 'SVR', i)
        #SVR_MSE.append(mean_squared_error(y_test, y_pred))

        yy = clfsvr.best_estimator_.predict(x)
        drawTrain(y, yy, 'SVR MSE = '+str(mean_squared_error(y_test, y_pred)), i)
        MSE[0].append(mean_squared_error(y_test, y_pred))

        if 'SVR' in Selected_learnerCode:
            print('SVR Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
            Learners.append(clfsvr.best_estimator_)

        Learners_map['SVR'] = svr

        """ann = Regressor(layers = [Layer("Sigmoid", units=14),
                                   Layer("Linear")],
                         learning_rate = 0.02,
                         random_state = 2018,
                         n_iter = 10)

        ann.fit(X_train,y_train)
        y_pred = ann.predict(X_test)
        print('ANN Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")"""

        parameters = {'n_estimators': [10, 50, 100, 200, 300, 400, 500, 1000]}
        rfr = RandomForestRegressor(n_estimators=200, random_state=0)
        # drawTrain(rfr, x, y, 'RFR', i)
        # rfr = RandomForestRegressor(n_estimators=200, random_state=0)
        clfrfr = GridSearchCV(rfr, parameters, n_jobs=-1, scoring='neg_mean_squared_error')
        clfrfr.fit(X_train, y_train)
        print('The parameters of the best model are: ')
        print(clfrfr.best_params_)
        y_pred = clfrfr.best_estimator_.predict(X_test)
        yy = clfrfr.best_estimator_.predict(x)
        MSE[1].append(mean_squared_error(y_test, y_pred))
        drawTrain(y, yy, 'RFR MSE = '+str(mean_squared_error(y_test, y_pred)), i)
        #RFR_MSE.append(mean_squared_error(y_test, y_pred))


        if 'RFR' in Selected_learnerCode:
            print('RFR Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
            Learners.append(clfrfr.best_estimator_)

        Learners_map['RFR'] = rfr

        parameters = {'alpha': np.logspace(-2, 2, 5)}
        lasso = Lasso(alpha=0.05, random_state=1, max_iter=1000)
        # drawTrain(lasso, x, y, 'LASSO', i)
        clflasso = GridSearchCV(lasso, parameters, n_jobs=-1, scoring='neg_mean_squared_error')
        clflasso.fit(X_train, y_train)
        yy = clflasso.best_estimator_.predict(x)
        print('The parameters of the best model are: ')
        print(clflasso.best_params_)
        y_pred = clflasso.best_estimator_.predict(X_test)
        drawTrain(y, yy, 'LASSO MSE = '+str(mean_squared_error(y_test, y_pred)), i)
        MSE[2].append(mean_squared_error(y_test, y_pred))



        if 'LASSO' in Selected_learnerCode:
            print('LASSO  Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
            #file.write('LASSO  Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
            Learners.append(clflasso.best_estimator_)

        Learners_map['LASSO'] =lasso




        #drawTrain(ENet, X_train, y_train,X_test,y_test, 'Elastic NET', i)
        parameters = {'alpha': np.logspace(-2, 2, 5), 'l1_ratio': np.linspace(0, 1.0, 11)}
        # ENet = ElasticNet(alpha=0.05, l1_ratio=.9, random_state=3)
        # drawTrain(ENet, x, y, 'Elastic NET', i)
        ENet = ElasticNet(alpha=0.05, l1_ratio=.9, random_state=3)
        clfENet = GridSearchCV(ENet, parameters, n_jobs=-1, scoring='neg_mean_squared_error')
        clfENet.fit(X_train, y_train)
        print('The parameters of the best model are: ')
        print(clfENet.best_params_)
        y_pred = clfENet.best_estimator_.predict(X_test)
        yy = clfENet.best_estimator_.predict(x)
        MSE[3].append(mean_squared_error(y_test, y_pred))
        drawTrain(y, yy, 'Elastic Net MSE = '+str(mean_squared_error(y_test, y_pred)), i)
        if 'ENET' in Selected_learnerCode:
            print('Elastic Net Mean squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
            Learners.append(clfENet.best_estimator_)

        Learners_map['ENET'] = ENet

        parameters = { 'n_estimators': [100, 500, 1000, 2000, 3000, 5000]}
        GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                           max_depth=4, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10,
                                           loss='huber', random_state=5)
        clfGBoost = GridSearchCV(GBoost, parameters, n_jobs=-1, scoring='neg_mean_squared_error')
        clfGBoost.fit(X_train, y_train)
        print('The parameters of the best model are: ')
        print(clfGBoost.best_params_)
        y_pred = clfGBoost.best_estimator_.predict(X_test)
        yy = clfGBoost.best_estimator_.predict(x)
        MSE[4].append(mean_squared_error(y_test, y_pred))
        #GBoost_MSE.append(mean_squared_error(y_test, y_pred))
        drawTrain(y, yy, 'GBoost MSE = '+str(mean_squared_error(y_test, y_pred)), i)
        if 'GBOOST' in Selected_learnerCode:
            print('GBoost squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
            Learners.append(clfGBoost.best_estimator_)

        Learners_map['GBOOST'] = GBoost


        #Adaboost
        #Adaboost = AdaBoostRegressor(base_estimator=SVR(C=1.0, epsilon=0.2))
        Adaboost = AdaBoostRegressor()
        Adaboost.fit(X_train,y_train)
        y_pred = Adaboost.predict(X_test)
        yy = Adaboost.predict(x)
        drawTrain(y,yy,'Adaboost MSE = '+ str(mean_squared_error(y_test, y_pred)),i)
        print('Adaboost with SVR squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        Ada_MSE.append(mean_squared_error(y_test, y_pred))

        #BAGGING
        baggingModel = baggingAveragingModels(models=(clfsvr.best_estimator_,clfrfr.best_estimator_,clfENet.best_estimator_,clfGBoost.best_estimator_,clflasso.best_estimator_))
        baggingModel.fit(X_train, y_train)
        y_pred = baggingModel.predict(X_test)
        MSE[5].append(mean_squared_error(y_test, y_pred))
        yy = baggingModel.predict(x)
        drawTrain(y, yy, 'Bagging before selected MSE = '+str(mean_squared_error(y_test, y_pred)), i)
        print('Bagging before selected squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")

        baggingModel = baggingAveragingModels(models=tuple(Learners))
        #drawTrain(baggingModel, X_train, y_train,X_test,y_test, 'Bagging', i)
        #baggingModel = baggingAveragingModels(models=tuple(Learners))

        baggingModel.fit(X_train, y_train)
        y_pred = baggingModel.predict(X_test)
        MSE[6].append(mean_squared_error(y_test, y_pred))
        yy = baggingModel.predict(x)
        drawTrain(y, yy, 'Bagging after selected selected MSE = '+str(mean_squared_error(y_test, y_pred)), i)

        print('Bagging after selected squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")

        All_learner = ['SVR','RFR','LASSO','ENET','GBOOST']
        for k in range(0,len(Selected_learnerCode)):

            """learnerList = []
            for kk in range(0,len(Selected_learnerCode)):
                if Selected_learnerCode[kk]!='' :
                    learnerList.append(Learners_map[Selected_learnerCode[kk]])"""
            """stacked_averaged_models = StackingAveragedModels(base_models=tuple(learnerList),
                                                             meta_model=Learners_map[All_learner[k]])
            drawTrain(stacked_averaged_models, X_train, y_train,X_test,y_test, 'stacking with '+All_learner[k], i)"""
            #stacked_averaged_models = StackingAveragedModels(base_models=tuple(learnerList),
            #                                                 meta_model=Learners_map[All_learner[k]])
            params = {}
            """
            if 'SVR' in Selected_learnerCode:
                params['svr__C'] = np.logspace(-3, 3, 7)
                params['svr__gamma'] = np.logspace(-3, 3, 7)

            if 'RFR' in Selected_learnerCode:
                params['randomforestregressor__n_estimators'] =[10, 50, 100, 500, 1000]

            if 'LASSO' in Selected_learnerCode:
                params['lasso__alpha'] = np.logspace(-2, 2, 5)

            if 'ENET' in Selected_learnerCode:
                params['elasticnet__alpha'] = np.logspace(-2, 2, 5)

            if 'GBOOST' in Selected_learnerCode:
                params['gradientboostingregressor__n_estimators']= [100, 500, 1000, 2000, 3000, 5000]"""

            if k==0:
                params['meta-svr__C'] = np.logspace(-3, 3, 7)
                params['meta-svr__gamma'] = np.logspace(-3, 3, 7)
            if k==1:
                params['meta-randomforestregressor__n_estimators'] = [10, 50, 100, 500, 1000]
            if k==2:
                params['meta-lasso__alpha'] = np.logspace(-2, 2, 5)
            if k==3:
                params['meta-elasticnet__alpha'] = np.logspace(-2, 2, 5)
            if k==4:
                params['meta-gradientboostingregressor__n_estimators'] = [100, 500, 1000, 2000, 3000, 5000]
            """
            params = {'svr__C': np.logspace(-3, 3, 7),
                      'svr__gamma': np.logspace(-3, 3, 7),
                      'randomforestregressor__n_estimators': [10, 50, 100, 500, 1000],
                      'lasso__alpha': np.logspace(-2, 2, 5),
                      'elasticnet__alpha':np.logspace(-2, 2, 5),
                      'gradientboostingregressor__n_estimators': [100, 500, 1000, 2000, 3000, 5000],
                      }"""
            stacked_averaged_models = StackingRegressor(regressors=Learners,
                                                        meta_regressor=Learners_map[All_learner[k]])
            grid = GridSearchCV(estimator=stacked_averaged_models,param_grid=params)
            grid.fit(X_train, y_train)
            y_pred = grid.best_estimator_.predict(X_test)
            yy = grid.best_estimator_.predict(x)
            drawTrain(y, yy, 'stacking with '+All_learner[k]+' MSE = '+str(mean_squared_error(y_test, y_pred)), i)
            print('Stacking with metamodel is '+All_learner[k]+' squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
            # file.write('Stacking with metamodel is lasso squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
            stacking_MSE[k].append(mean_squared_error(y_test, y_pred))

        #stacked_averaged_models = StackingAveragedModels(base_models=tuple(learnerList),
        #                                                 meta_model=baggingModel)
        #drawTrain(stacked_averaged_models, X_train, y_train, X_test, y_test, 'stacking with Bagging models'  , i)
        """stacked_averaged_models = StackingAveragedModels(base_models=tuple(learnerList),
                                                         meta_model=Learners_map[All_learner[k]])"""
        stacked_averaged_models = StackingRegressor(regressors=Learners,
                                                    meta_regressor=baggingModel)
        #grid = GridSearchCV(estimator=stacked_averaged_models, param_grid=params)
        stacked_averaged_models.fit(X_train, y_train)
        y_pred = stacked_averaged_models.predict(X_test)
        yy = stacked_averaged_models.predict(x)
        drawTrain(y, yy, 'stacking with bagging MSE = '+str(mean_squared_error(y_test, y_pred)) , i)
        print('Stacking with metamodel is bagging models squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        # file.write('Stacking with metamodel is lasso squared error is ' + str(mean_squared_error(y_test, y_pred)) + "\n")
        stacking_MSE[5].append(mean_squared_error(y_test, y_pred))


        gc.collect()


    print("Adaboost mean is "+str(np.mean(Ada_MSE)))

    min_stacking_MSE = []
    for i in range(0, times):
        minMSE = stacking_MSE[0][i]
        for j in range(1, 6):
            if stacking_MSE[j][i] < minMSE:
                minMSE = stacking_MSE[j][i]
        min_stacking_MSE.append(minMSE)

    plot_x = np.linspace(1, times, times)
    if len(MSE[0])>0:
        plt.plot(plot_x,MSE[0],'b')
    if len(MSE[1]) > 0:
        plt.plot(plot_x, MSE[1], 'r')
    if len(MSE[2]) > 0:
        plt.plot(plot_x, MSE[2], 'y')
    if len(MSE[3]) > 0:
        plt.plot(plot_x, MSE[3], 'k')
    if len(MSE[4]) > 0:
        plt.plot(plot_x, MSE[4], 'g')
    if len(MSE[5]) > 0:
        plt.plot(plot_x, MSE[5], 'm')
    if len(MSE[6]) > 0:
        plt.plot(plot_x, MSE[6], color='coral', linestyle=':', marker='|')
    plt.plot(plot_x,min_stacking_MSE,color = 'cyan')
    plt.xlabel('Repeat times')
    plt.ylabel('MSE')
    plt.legend(( 'SVR avg = '+str(np.mean(MSE[0])),
                'RFR avg = '+str(np.mean(MSE[1])),
                'Lasso avg=' + str(np.mean(MSE[2])),
                 'Enet avg=' + str(np.mean(MSE[3])),
                 'Gboost avg = ' + str(np.mean(MSE[4])),
                 'Bagging before avg = ' + str(np.mean(MSE[5])),
                'Bagging after avg = ' + str(np.mean(MSE[6])),
                 'BS-LIBS avg = '+str(np.mean(min_stacking_MSE))
                ), loc='upper right')
    plt.title('Different learning machine')
    plt.savefig('DifferentLearner.png')
    plt.clf()
    plt.plot()

    plot_x = np.linspace(1, times, times)
    plt.plot(plot_x,Ada_MSE,'b')
    plt.plot(plot_x,MSE[6],'r')
    plt.plot(plot_x,min_stacking_MSE,'g')
    plt.legend(('Adaboost avg = '+str(np.mean(Ada_MSE)),
        'Bagging after avg = ' + str(np.mean(MSE[6])),
                'BS-LIBS avg = ' + str(np.mean(min_stacking_MSE))
                ), loc='upper right')
    plt.title('Bagging VS BS-LIBS VS Adaboost')
    plt.xlabel('Repeat times')
    plt.ylabel('MSE')
    plt.savefig('Bagging VS BS-LIBS&Adaboost.png')
    plt.clf()
    plt.plot()



    plot_x = np.linspace(1, times, times)
    if len(stacking_MSE[0]) > 0:
        plt.plot(plot_x, stacking_MSE[0], 'b')
    if len(stacking_MSE[1]) > 0:
        plt.plot(plot_x, stacking_MSE[1], 'r')
    if len(stacking_MSE[2]) > 0:
        plt.plot(plot_x, stacking_MSE[2], 'y')
    if len(stacking_MSE[3]) > 0:
        plt.plot(plot_x, stacking_MSE[3], 'k')
    if len(stacking_MSE[4]) > 0:
        plt.plot(plot_x, stacking_MSE[4], 'g')
    if len(stacking_MSE[5]) > 0:
        plt.plot(plot_x, stacking_MSE[5], 'm')
    plt.legend(('SVR avg = ' + str(np.mean(stacking_MSE[0])),
                'RFR avg = ' + str(np.mean(stacking_MSE[1])),
                'Lasso avg=' + str(np.mean(stacking_MSE[2])),
                'Enet avg=' + str(np.mean(stacking_MSE[3])),
                'Gboost avg = ' + str(np.mean(stacking_MSE[4])),
                'Bagging avg = ' + str(np.mean(stacking_MSE[5]))
                ), loc='upper right')
    plt.title('Different meta-learning machine(Adaboost avg MSE=' + str(np.mean(Ada_MSE)) + ')')
    plt.xlabel('Repeat times')
    plt.ylabel('MSE')
    plt.savefig('DifferentMetaLearner.png')
    plt.clf()
    plt.plot()



if __name__=='__main__':
    prepareConcentrationData()
    x, y = newMain()

    y_df = pd.DataFrame(y, columns=['Target'])
    """x_df = pd.DataFrame(x, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7',
                                    'feature8','feature9','feature10','feature11','feature12','feature13','feature14','feature15','feature16'
                                    ,'feature17','feature18','feature19','feature20'])"""

    x_df = pd.DataFrame(x, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])

    newx, newy = processingRatios()

    selectLearner(newx, newy)

    element = 'Al'
    REPEAT_TIMES = 100
    if not os.path.exists("E:\\LIBS_experiment\\" + element + 'v7_Canada'):
        os.mkdir("E:\\LIBS_experiment\\" + element + 'v7_Canada')
    os.chdir("E:\\LIBS_experiment\\" + element + 'v7_Canada')
    useXYtrain(newx, newy, REPEAT_TIMES)
    

