import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split

DATA_base_add = "E:\\NasaChemcamLIBSDataset\\"
DATA_folder_add = "E:\\NasaChemcamLIBSDataset\\Derived\\ica\\"

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





if __name__=='__main__':
    con = loadConcentrationData()
    trainingData = {}
    loadTrainingSamples()
    meanTrainingData = {}
    getMean()
    print(meanTrainingData)
    element_dict = prepareNIST()
