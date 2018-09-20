"""
    LIBS gradient boosting regression

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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

# #############################################################################
# Load data
#boston = datasets.load_boston()
#X, y = shuffle(boston.data, boston.target, random_state=13)

#X = X.astype(np.float32)
#offset = int(X.shape[0] * 0.9)
#X_train, y_train = X[:offset], y[:offset]
#X_test, y_test = X[offset:], y[offset:]\

if __name__=='__main__':

    elementList = ['Cu', 'Ba', 'Pb', 'Cd']

    for element in elementList:


        print('Testing element is ' + element)
        print('1.Get element characteristic peaks' + 20 * '-')
        CP = getCP(element)
        # print(CP)

        print()
        print('2.Get element peak according to NIST' + 20 * '-')
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

            Y.append(i + 0.5)


        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.20)


        # #############################################################################
        # Fit regression model
        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        clf = ensemble.GradientBoostingRegressor(**params)

        clf.fit(X_train, y_train)
        mse = mean_squared_error(y_test, clf.predict(X_test))
        print("MSE: %.4f" % mse)

        # #############################################################################
        # Plot training deviance

        # compute test set deviance
        test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

        for i, y_pred in enumerate(clf.staged_predict(X_test)):
            test_score[i] = clf.loss_(y_test, y_pred)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
                 label='Training Set Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
                 label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')

        # #############################################################################
        # Plot feature importance
        """
        feature_importance = clf.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, boston.feature_names[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.show()
        """
        plt.show()