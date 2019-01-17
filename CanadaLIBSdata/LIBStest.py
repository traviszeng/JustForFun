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
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import copy

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
concentrate_set_200AVG = {}
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
        

print("Get "+str(num)+" 200_AVG files.")
print()

data_set_1000AVG = {}
concentrate_set_1000AVG = {}
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

print("Get "+str(num)+" 1000_AVG files.")

#去掉hand sample34的记录（全为0）
del concentrate_set_200AVG['hand sample37 barite_200AVG']
del data_set_200AVG['hand sample37 barite_200AVG']

#画一个折线图尝试看看
s_name = "RockFer2_200AVG"
intensity = data_set_200AVG[s_name].Intensity
wavelength = data_set_200AVG[s_name].WaveLength
intensity = np.array(intensity)
wavelength = np.array(wavelength)
plt.plot(wavelength,intensity)
#plt.show()

print("准备NIST库相关数据")
nist = pd.read_csv("E:\\JustForFun\\CanadaLIBSdata\\andor.nist",header = None,names = ['WaveLength','Element','Type','Unknown','Importance'])
nist = nist.loc[1:]
#删除未知列
del nist['Unknown']
#筛选在样本精度范围的nist线
nist = nist.loc[nist.WaveLength>=198.066]
nist = nist.loc[nist.WaveLength<=970.142]




#准备训练数据，未作任何处理，将整个光谱输入
X = []
#波长
waveLength = []
#AL的相关浓度信息
Al_y = []
#Fe
Fe_y = []
#Ca
Ca_y = []
#K
K_y = []
#Mg
Mg_y = []
#Na
Na_y = []
#Si
Si_y = []
#Ti
Ti_y = []
#P
P_y = []
#Mn
Mn_y = []

for samplename,concentrate in concentrate_set_200AVG.items():
    X.append(np.array(data_set_200AVG[samplename].Intensity))
    waveLength.append(np.array(data_set_200AVG[samplename].WaveLength))
    Al_y.append(concentrate[0]*100)
    Ca_y.append(concentrate[1]*100)
    Fe_y.append(concentrate[2]*100)
    K_y.append(concentrate[3]*100)
    Mg_y.append(concentrate[4]*100)
    Mn_y.append(concentrate[5]*100)
    Na_y.append(concentrate[6]*100)
    Si_y.append(concentrate[7]*100)
    Ti_y.append(concentrate[8]*100)
    P_y.append(concentrate[9]*100)

for samplename,concentrate in concentrate_set_1000AVG.items():
    X.append(np.array(data_set_1000AVG[samplename].Intensity))
    waveLength.append(np.array(data_set_1000AVG[samplename].WaveLength))
    Al_y.append(concentrate[0]*100)
    Ca_y.append(concentrate[1]*100)
    Fe_y.append(concentrate[2]*100)
    K_y.append(concentrate[3]*100)
    Mg_y.append(concentrate[4]*100)
    Mn_y.append(concentrate[5]*100)
    Na_y.append(concentrate[6]*100)
    Si_y.append(concentrate[7]*100)
    Ti_y.append(concentrate[8]*100)
    P_y.append(concentrate[9]*100)





print('Data preprocessing finished.')
print()
print('1.单独学习器实验------------------------------')
print()
print('1.1 Support Vector Regression-------------------------')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

print('1.1.1 Al的实验-----------------------')

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Al_y,
                                                    test_size=0.20)
clf = SVR(C=1.0,epsilon=0.2)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.2 Ca的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Ca_y,
                                                    test_size=0.20)
clf = SVR(C=1.0,epsilon=0.2)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))


print('1.1.3 Fe的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Fe_y,
                                                    test_size=0.20)
clf = SVR(C=1.0,epsilon=0.2)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))


print('1.1.4 K的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    K_y,
                                                    test_size=0.20)
clf = SVR(C=1.0,epsilon=0.2)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.5 Mg的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Mg_y,
                                                    test_size=0.20)
clf = SVR(C=1.0,epsilon=0.2)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.6 Mn的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Mn_y,
                                                    test_size=0.20)
clf = SVR(C=1.0,epsilon=0.2)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.7 Na的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Na_y,
                                                    test_size=0.20)
clf = SVR(C=1.0,epsilon=0.2)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.8 Si的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Si_y,
                                                    test_size=0.20)
clf = SVR(C=1.0,epsilon=0.2)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.9 Ti的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Ti_y,
                                                    test_size=0.20)
clf = SVR(C=1.0,epsilon=0.2)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.10 P的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    P_y,
                                                    test_size=0.20)
clf = SVR(C=1.0,epsilon=0.2)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print()
print('1.2 随机森林测试---------------------------------')
from sklearn.ensemble import RandomForestRegressor
print('1.1.1 Al的实验-----------------------')

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Al_y,
                                                    test_size=0.20)
clf = RandomForestRegressor(n_estimators=20,random_state=0)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.2 Ca的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Ca_y,
                                                    test_size=0.20)
clf = RandomForestRegressor(n_estimators=20,random_state=0)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))


print('1.1.3 Fe的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Fe_y,
                                                    test_size=0.20)
clf = RandomForestRegressor(n_estimators=20,random_state=0)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))


print('1.1.4 K的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    K_y,
                                                    test_size=0.20)
clf = RandomForestRegressor(n_estimators=20,random_state=0)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.5 Mg的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Mg_y,
                                                    test_size=0.20)
clf = RandomForestRegressor(n_estimators=20,random_state=0)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.6 Mn的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Mn_y,
                                                    test_size=0.20)
clf = RandomForestRegressor(n_estimators=20,random_state=0)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.7 Na的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Na_y,
                                                    test_size=0.20)
clf = RandomForestRegressor(n_estimators=20,random_state=0)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.8 Si的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Si_y,
                                                    test_size=0.20)
clf = RandomForestRegressor(n_estimators=20,random_state=0)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.9 Ti的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Ti_y,
                                                    test_size=0.20)
clf = RandomForestRegressor(n_estimators=20,random_state=0)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.1.10 P的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    P_y,
                                                    test_size=0.20)
clf = RandomForestRegressor(n_estimators=20,random_state=0)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print()
print('1.3 KRR测试---------------------------------')

print('1.3.1 Al的实验-----------------------')

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Al_y,
                                                    test_size=0.20)
clf = Lasso(alpha =0.0005, random_state=1)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.3.2 Ca的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Ca_y,
                                                    test_size=0.20)
clf = Lasso(alpha =0.0005, random_state=1)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))


print('1.3.3 Fe的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Fe_y,
                                                    test_size=0.20)
clf = Lasso(alpha =0.0005, random_state=1)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))


print('1.3.4 K的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    K_y,
                                                    test_size=0.20)
clf = Lasso(alpha =0.0005, random_state=1)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.3.5 Mg的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Mg_y,
                                                    test_size=0.20)
clf = Lasso(alpha =0.0005, random_state=1)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.3.6 Mn的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Mn_y,
                                                    test_size=0.20)
clf = Lasso(alpha =0.0005, random_state=1)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.3.7 Na的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Na_y,
                                                    test_size=0.20)
clf = Lasso(alpha =0.0005, random_state=1)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.3.8 Si的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Si_y,
                                                    test_size=0.20)
clf = Lasso(alpha =0.0005, random_state=1)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.3.9 Ti的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Ti_y,
                                                    test_size=0.20)
clf = Lasso(alpha =0.0005, random_state=1)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))

print('1.3.10 P的实验-----------------------')
del clf
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    P_y,
                                                    test_size=0.20)
clf = Lasso(alpha =0.0005, random_state=1)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print('Mean squared error is '+str(mean_squared_error(y_test,y_pred)))
