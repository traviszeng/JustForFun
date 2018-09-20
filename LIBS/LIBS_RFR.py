"""
    Random forest LIBS quantitative analysis
    @Travis Zeng on 2018/9/14
"""
"""
RF基本步骤
1.从数据集中选取 N 个随机记录。
2.根据这 N 个记录构建决策树。
3.选择算法中需要的决策树的个数，然后重复步骤1和2。
4.在回归问题的情况下，对于新记录，森林中的每棵树预测Y（输出）的值。
最终值可以通过取得森林中所有树木预测的所有值的平均值来计算。
对于分类问题，林中的每棵树都会预测新记录所属的类别。最后，新记录被分配到赢得多数票的类别。


"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics



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

def RFRtrainDemo():
    dataset = pd.read_csv('petrol_consumption.csv')

    #本部分先将数据分成特征和标签集，然后将得到的数据分成训练集和测试集。
    X = dataset.iloc[:, 0:4].values
    y = dataset.iloc[:, 4].values


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.ensemble import RandomForestRegressor

    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)



    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



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



        pipeRFR = make_pipeline(MinMaxScaler(),
                                RandomForestRegressor(n_estimators=20,random_state=0))

        pipeRFR.fit(X_train,y_train)
        y_pred = pipeRFR.predict(X_test)
        print(y_pred)
        print('Test Accuracy: %.3f' % pipeRFR.score(X_test, y_test))



        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

