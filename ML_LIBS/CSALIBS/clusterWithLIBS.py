"""
对于样本先用聚类找出相关的样本
再根据不同的类型的样本训练

"""
import pandas as pd
import numpy as np



def loadConcentrateFile():
    print("Loading concentrate file.....")
    concentrate_data = pd.read_csv("E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\Sample_Composition_Data.csv")
    #前81行为数据
    concentrate_data = concentrate_data.loc[0:81]
    return concentrate_data


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

    concentrate_data.drop(62)
    return concentrate_data


def mainElement(concentrateData):
    del concentrateData['Name']
    del concentrateData['C']
    label = []
    for i in range(0,len(concentrateData)):
        label.append(np.where(concentrateData.loc[i]==np.max(concentrateData.loc[i])))

    return label

data  = loadConcentrateFile()
data  = dataPreprocessing(data)
label = mainElement(data)
