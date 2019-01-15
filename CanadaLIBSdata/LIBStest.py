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
	


