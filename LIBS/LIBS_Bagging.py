"""

    LIBS ensemble using bagging

"""

import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR

from sklearn import metrics
import numpy as np

from sklearn.ensemble import RandomForestRegressor

"""
加载NIST库文件
"""
def loadFile(filename):
    #print('loading '+filename)
    file = open(filename,encoding = 'utf-8')

    datalist = file.readlines()

    file.close()
    newdatalist = []
    for row in datalist:
        newdatalist.append(row.split('|'))

    return newdatalist

"""
处理文件获得的数据list,对空白的地方填0补全

"""
def processDataList(newdatalist):

    datalist = []
    flag = 0

    LEN = len(newdatalist[0])

    #print(newdatalist)
    for row in newdatalist:
        if row!=['\n']:

            data = []
            for i in range(0,LEN):
                try:
                    data.append(float(row[i]))

                except ValueError:
                    data.append(float(0))

        #data.append(float(label))
            if flag!=0:
                datalist.append(data)
            flag+=1


    return datalist

"""
params:
    element: (string) name of target element

return:
    特征谱线 list

"""
def getCP(element):
    file = open('E:\\ANN data\\spc\\'+element+'.txt','r')
    lines = file.readlines()
    file.close()
    CP = []
    for i in range(0,len(lines)):
        wave = float(lines[i].split()[1])
        if not (wave<float(200) or wave>float(601)):
            CP.append(wave)

    return CP


"""
获取整个CP列表的特征峰的峰值，并返回该特征峰的最大峰值和对应的波长
"""
def findAllPeakValue(CP,dataList):

    maxList = []
    for c in CP:
        maxWave,Max = findPeakValueOfOneWave(dataList,c)
        maxList.append([c,maxWave,Max])

    return maxList


"""
找到一个特征峰对应的峰值
寻找左右1范围内最大的峰值
"""
def findPeakValueOfOneWave(dataList,wave):
    Width = float(0.5)
    Max = 0
    maxWave = 0

    for data in dataList:
        if data[0]>float(wave-Width) and data[0]<float(wave+Width):
            if Max<data[1]:
                Max  = data[1]
                maxWave = data[0]

    return maxWave,Max


if __name__=='__main__':

    elementList = ['Cu','Ba','Pb','Cd']

    for element in elementList:

    #element = 'Cu'
        print('Testing element is '+element)
        print('1.Get element characteristic peaks'+20*'-')
        CP = getCP(element)
        #print(CP)
        print()
        print('2.Get element peak according to NIST'+20*'-')
        trainingData = []
        X = []
        Y = []
        for i in range(1, 51):
            rawData = findAllPeakValue(CP, processDataList(
                loadFile('E:\\ANN data\\data\\' + element + '\\' + str(i) + 'ppm.txt')))
            rawData2 = findAllPeakValue(CP, processDataList(
                loadFile('E:\\ANN data\\data\\' + element + '\\' + str(i + 0.5) + 'ppm.txt')))

            x = []
            for data in rawData:
                x.append(data[2])

            X.append(x)

            Y.append(i)

            x = []
            for data in rawData2:
                x.append(data[2])

            X.append(x)

            Y.append(i+0.5)


        #print(train_X)
        #print(train_y)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=0.20)

        print('3.Using single model(Unoptimized)'+20*'-')
        print('SVR:')
        pipe_svr = make_pipeline(MinMaxScaler(),
                                SVR(C=1.0, epsilon=0.2))

        pipe_svr.fit(X_train, y_train)

        y_pred = pipe_svr.predict(X_test)
        print('Test Accuracy: %.3f' % pipe_svr.score(X_test, y_test))
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        print('RFR:')
        pipe_rfr = make_pipeline(MinMaxScaler(),
                                 RandomForestRegressor(n_estimators=20, random_state=0))

        pipe_rfr.fit(X_train, y_train)

        y_pred = pipe_rfr.predict(X_test)
        print('Test Accuracy: %.3f' % pipe_svr.score(X_test, y_test))
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        pipe_krr = make_pipeline(MinMaxScaler(),
                                KernelRidge(alpha=1.0))

        pipe_krr.fit(X_train, y_train)

        y_pred = pipe_krr.predict(X_test)
        print('Test Accuracy: %.3f' % pipe_krr.score(X_test, y_test))
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        print('4.Using bagging method'+20*'-')
        y_pred1 = pipe_svr.predict(X_test)
        y_pred2 = pipe_krr.predict(X_test)
        y_pred3 = pipe_rfr.predict(X_test)

        y_pred = (y_pred1+y_pred2+y_pred3)/3

        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




