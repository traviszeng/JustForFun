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

#处理初始数据文件的一些问题
def processRawData(data):
    data = data.loc[1:6144]
    #处理一下columns的问题
    for n in data.columns:
        n.replace('#',' ')
        n.strip()
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
                        data = pd.read_csv(os.path.join(root, file),skiprows=16)
                        data = processRawData(data)
                        if name not in trainingData.keys():
                            trainingData[name] = []
                            trainingData[name].append(data)
                        else:
                            trainingData[name].append(data)
                        #print(data)






if __name__=='__main__':
    con = loadConcentrationData()
    trainingData = {}
    loadTrainingSamples()
