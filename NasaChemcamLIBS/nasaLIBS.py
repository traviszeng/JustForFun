import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


DATA_base_add = "E:\\NasaChemcamLIBSDataset\\"
DATA_folder_add = "E:\\NasaChemcamLIBSDataset\\Derived\\ica\\"

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
        print(key)
        sum = 0
        data = pd.DataFrame(trainingData[key][0]['wave'])
        #data['wave']  = trainingData['wave']
        for df in item:
            data['avg'+str(sum+1)] = df['mean']
            sum+=1

        meanTrainingData[key] = data





if __name__=='__main__':
    con = loadConcentrationData()
    trainingData = {}
    loadTrainingSamples()
    meanTrainingData = {}
    getMean()
    print(meanTrainingData)
