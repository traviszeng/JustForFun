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

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
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
    index = 0
    for element in elementList:
        index += 1

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

        #X_train.columns = CP

        #for i, j in sorted(zip(X_train.columns, pipeRFR.feature_importances_)):
            #print(i, j)

        #for i,j in sorted(zip(pipeRFR.steps[1][1].feature_importances_,CP)):
            #print(j,i)

        print('4.Testing learning curve ' + 20 * '-')
        # a demo of learning curve and validation curve

        """
        这个函数的作用为：对于不同大小的训练集，确定交叉验证训练和测试的分数。
        一个交叉验证发生器将整个数据集分割k次，分割成训练集和测试集。
        不同大小的训练集的子集将会被用来训练评估器并且对于每一个大小的训练子集都会产生一个分数，然后测试集的分数也会计算。
        然后，对于每一个训练子集，运行k次之后的所有这些分数将会被平均。

            estimator：所使用的分类器

            X:array-like, shape (n_samples, n_features)

             训练向量，n_samples是样本的数量，n_features是特征的数量

            y:array-like, shape (n_samples) or (n_samples, n_features), optional

            目标相对于X分类或者回归

            train_sizes:array-like, shape (n_ticks,), dtype float or int

            训练样本的相对的或绝对的数字，这些量的样本将会生成learning curve。如果dtype是float，他将会被视为最大数量训练集的一部分（这个由所选择的验证方法所决定）。否则，他将会被视为训练集的绝对尺寸。要注意的是，对于分类而言，样本的大小必须要充分大，达到对于每一个分类都至少包含一个样本的情况。

            cv:int, cross-validation generator or an iterable, optional

            确定交叉验证的分离策略

            --None，使用默认的3-fold cross-validation,

            --integer,确定是几折交叉验证

            --一个作为交叉验证生成器的对象

            --一个被应用于训练/测试分离的迭代器

            verbose : integer, optional

            控制冗余：越高，有越多的信息



            返回值：

            train_sizes_abs：array, shape = (n_unique_ticks,), dtype int

            用于生成learning curve的训练集的样本数。由于重复的输入将会被删除，所以ticks可能会少于n_ticks.

            train_scores : array, shape (n_ticks, n_cv_folds)

            在训练集上的分数

            test_scores : array, shape (n_ticks, n_cv_folds)

            在测试集上的分数
        """
        train_sizes, train_scores, test_scores = learning_curve(estimator=RandomForestRegressor(n_estimators=20,random_state=0),
                                                                X=X,
                                                                y=Y,
                                                                train_sizes=np.linspace(0.1, 1.0, 10),
                                                                cv=10,
                                                                n_jobs=-1,
                                                                scoring='neg_mean_squared_error')

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.subplot(2,2,index)
        plt.plot(train_sizes, train_mean,
                 color='blue', marker='o',
                 markersize=5, label='training accuracy')

        plt.fill_between(train_sizes,
                         train_mean + train_std,
                         train_mean - train_std,
                         alpha=0.15, color='blue')

        plt.plot(train_sizes, test_mean,
                 color='green', linestyle='--',
                 marker='s', markersize=5,
                 label='validation accuracy')

        plt.fill_between(train_sizes,
                         test_mean + test_std,
                         test_mean - test_std,
                         alpha=0.15, color='green')

        plt.grid()
        plt.xlabel('Number of training samples')
        plt.ylabel('Neg MSE')
        plt.legend(loc='lower right')

        plt.ylim([-800, 10])
        plt.tight_layout()

    plt.show()