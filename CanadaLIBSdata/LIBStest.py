"""
利用加拿大航天局的LIBS数据进行LIBS定量分析实验
"""
import os
import sys
import pandas as pd

concentrate_data = pd.read_csv("E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\Sample_Composition_Data.csv")
#前81行为数据
concentrate_data = concentrate_data.loc[0:81]

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
	

route_200_AVG = "E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\csv Material Large Set 200pulseaverage"
route_1000_AVG = "E:\\JustForFun\\CanadaLIBSdata\\LIBS OpenData csv\\csv Certified Samples Subset 1000pulseaverage"

postfix_200AVG ="_200AVG.csv"
postfix_1000AVG = "_1000AVG.csv"

"""
加载训练样本
"""

data_set_200AVG = {}
#加载200AVG的样本，并将其存到data_set_200AVG中
os.chdir(route_200_AVG)
num = 0
for indexs in concentrate_data.index:
    if os.path.exists(concentrate_data.loc[indexs].values[0]+postfix_200AVG):
        num+=1
        print("Get data file:"+concentrate_data.loc[indexs].values[0]+postfix_200AVG)
        data = pd.read_csv(concentrate_data.loc[indexs].values[0]+postfix_200AVG,header = None,names = ['WaveLength','Intensity'])
        data_set_200AVG[concentrate_data.loc[indexs].values[0]+"_200AVG"] = data

print("Get "+str(num)+" 200_AVG files.")
print()

data_set_1000AVG = {}
num = 0
#加载1000AVG的样本，并将其存到data_set_1000AVG中
os.chdir(route_1000_AVG)
for indexs in concentrate_data.index:
    if os.path.exists(concentrate_data.loc[indexs].values[0]+postfix_1000AVG):
        num+=1
        print("Get data file:"+concentrate_data.loc[indexs].values[0]+postfix_1000AVG)
        data = pd.read_csv(concentrate_data.loc[indexs].values[0]+postfix_1000AVG,header = None,names = ['WaveLength','Intensity'])
        data_set_1000AVG[concentrate_data.loc[indexs].values[0]+"_1000AVG"] = data

print("Get "+str(num)+" 1000_AVG files.")

