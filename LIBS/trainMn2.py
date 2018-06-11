"""
神经网络进行LIBS定量测量的version 3.0

"""

"""
主要流程：
假设有一组真实数据作为标定
1.根据特征峰的特征谱线重要性程度以及训练集特征找到可以量化元素的特征峰（采用特征峰强度而不是积分强度）
2.将训练集光谱强度放缩到真实集的范围
3.训练
4.使用另外两组进行预测评估

problems:
1.可能会出现特征谱线重要程度很大但是训练集中却无法体现（如Al）
2.可能出现不同时间获取的数据单位基准不同，导致数量级不同

"""

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from sklearn import datasets
from tensorflow.python.framework import ops


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
获取元素特征谱线集合与其重要系数

return:

importanceList  as [wave,importance]
"""
def getCPandImportance(element):
    file = open('E:\\ANN data\\spc\\' + element + '.txt', 'r')
    lines = file.readlines()
    importanceList = []
    for line in lines:
        s = []
        """
        波长
        """
        wave = float(line.split()[1])
        """
        该波长对元素贡献的重要程度
        """
        importance = int(line.split()[3])
        s.append(wave)
        s.append(importance)
        importanceList.append(s)

    return importanceList




"""
process real data file

return:
    转换为一个list[波长,光强]

"""
def rawDataToDataList(filename):
    file = open(filename,'r',encoding = 'utf-8')
    rawData = file.readlines()
    dataList = []
    for data in rawData:
        l = data.split()
        dataList.append([float(l[0]),float(l[2])-float(l[1])])

    return dataList

"""
转换为Array的形式，返回两个list分别记录波长和光谱强度
"""
def listToArray(dataList):
    waveList = []
    strengthList = []
    for data in dataList:
        waveList.append(data[0])
        #对光强是负数的进行处理
        if data[1]<0:
            strengthList.append(float(0))
        else:
            strengthList.append(data[1])

    return waveList,strengthList

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
返回分别记录波长和峰值的两个列表
"""
def getPeakValueList(Dict):
    waveList = []
    for key in Dict:
        waveList.append(float(key))

    waveList = sorted(waveList)
    maxList = []
    for i in range(0,len(waveList)):
        maxList.append( Dict[str(waveList[i])])

    return maxList,waveList

"""
获取真实数据中在训练集特征峰上最大的值

"""
def getPeakReal(wavelist,datalist):
    Width = float(0.5)
    mlist = []
    wlist = []
    for w in wavelist:
        Max = 0
        maxWave = 0
        for data in datalist:
            if data[0]>float(w-Width) and data[0]<float(w+Width):
                if Max<data[1]:
                    Max  = data[1]
                    maxWave = data[0]

            mlist.append(Max)
            wlist.append(maxWave)

    return wlist,mlist

"""
过滤获得峰值点：将小于200或者小于600的过滤，并获得峰值点
"""
def findLocalOptimal(datalist):
    optList = []
    for i in range(1,len(datalist)):
        if datalist[i][0]<float(200) or datalist[i][0]>float(600):
            pass
        elif datalist[i][1]>datalist[i+1][1] and datalist[i][1]>datalist[i-1][1]:
            optList.append([datalist[i][0],datalist[i][1]])
        else:
            continue

    return optList

"""
:return waveStrengthList 获取的和训练集对应的最大强度和波长的列表
"""
def getPeakOptimal(waveList,optList):
    Width = float(0.5)
    waveOptList = []
    waveStrengthList =[]
    for wave in waveList:
        list = []
        MAX = 0
        MAXWAVE = 0
        for opt in optList:
            #如果波长在+ -1范围内
            if opt[0]<wave+Width and opt[0]>wave-Width:
                list.append(opt)
                if opt[1]>MAX:
                    MAX = opt[1]
                    MAXWAVE = opt[0]
            if opt[0]>wave+1.5:
                waveStrengthList.append([MAXWAVE,MAX])
                break

        waveOptList.append(list)

    return waveOptList,waveStrengthList

"""
获得和处理后的训练集波长集合对应的波长和最大强度列表

"""



"""
找到最接近该波长的特征谱线，并返回其importance
"""
def findClosestWave(wave):
    min = 9999
    closestWave = float(0)
    importance = 0
    #print(cpImportanceList)
    #print("test"+str(wave))
    for CPI in cpImportanceList:
        if abs(CPI[0]-wave)<min:
            min = abs(CPI[0]-wave)
            closestWave = CPI[0]
            importance = CPI[1]

    #print([closestWave,importance])

    return [closestWave,importance]


"""
找到标杆波长，并将训练集的对应强度放缩到真实数据范围：
标杆波长满足的条件：
1.重要性系数>500
2.强度较大（不一定是最大）
"""
"""
流程：
找到最大强度的波长
查看10 20 50对应的是否成一定线性关系
查看importance是否>500
进行放缩
:return 放缩的倍数
"""

def findBenchmarkWave():#tenmaxList,twemaxList,fifmaxList,tenwaveList,twewaveList,fifwaveList,wslist):
    """
    获得最大光强对应的wave

    """
    print(tenmaxList)
    WIDTH = float(0.5)

    flag = True
    while(flag):
        #print("show")
        print(tenmaxList)
        #print(tenwaveList)
        tenmax = tenmaxList.index(max(tenmaxList))
        #print("10ppm最大的是"+str(tenmax))
        #tenmax = tenmaxList.index(max(tenmaxList))
        #print("20ppm最大的是" + str(tenmax))
        #fifmax = fifmaxList.index(max(fifmaxList))
        #print("50ppm最大的是" + str(fifmax))

        #如果该强度成一定线性关系
        print(abs(twemaxList[tenmax]/tenmaxList[tenmax]))
        print(abs(fifmaxList[tenmax]/tenmaxList[tenmax]))
        if abs(twemaxList[tenmax]/tenmaxList[tenmax]-2)<WIDTH and abs(fifmaxList[tenmax]/tenmaxList[tenmax]-5)<WIDTH:
            #print("第一个判断通过")
            #且其对应的波长对该元素重要性>500
            #print(findClosestWave(tenwaveList[tenmax])[1])
            #print(findClosestWave(tenwaveList[tenmax])[1])
            if findClosestWave(tenwaveList[tenmax])[1]>500 and findClosestWave(twewaveList[tenmax])[1]>500 and findClosestWave(fifwaveList[tenmax])[1]>500:

                flag = False
                #放缩的倍数
                times = wslist[tenmax][1]/tenmaxList[tenmax]
                print(element+'对应的标杆波长为:')
                print(str(tenwaveList[tenmax]),str(tenmaxList[tenmax]))
                print(str(twewaveList[tenmax]), str(twemaxList[tenmax]))
                print(str(fifwaveList[tenmax]), str(fifmaxList[tenmax]))
                print(str(wslist[tenmax]))
                print(times)

                return times

        #print("zhixing")
        #删掉不满足条件的特征谱线
        del tenwaveList[tenmax]
        del twewaveList[tenmax]
        del fifwaveList[tenmax]

        del tenmaxList[tenmax]
        del twemaxList[tenmax]
        del fifmaxList[tenmax]

        del wslist[tenmax]

    return 0


"""
神经网络训练方法
"""
def trainANN(element,HIDDEN_NUM, LEARNING_RATE, BATCH_SIZE, Data,isShowFigure):
    ops.reset_default_graph()

    # 假定数据集合为data，前LEN项为LEN个特征谱线的光强，最后一项为铁元素浓度
    file = open('E:\\ANN_TRAIN\\LIBS_ANN v3 ' + element + '3.txt', 'a')
    #print(Data)
    data = Data

    # for dat in data:
    # print(dat)

    LEN = len(data[0]) - 1
    print(LEN)

    x_vals = np.array([x[0:(LEN-1)] for x in data])
    #print(x_vals)
    y_vals = (np.array([x[LEN] for x in data]))
    #print(y_vals)

    maxList = x_vals.max(axis=0)
    minList = x_vals.min(axis=0)
    print(maxList)
    print(minList)

    # 创建一个session（session是用户使用TensorFlow使得交互接口）
    session = tf.Session()

    # 设置一个图随机种子，使得返回结果可复现
    seed = 2
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # 将数据集划分为训练集和测试集（8：2）
    train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]

    # print(x_vals_train.max(axis=0))

    # 先将训练集的x和y放缩到0-1之间，并将不是数字的项（NaN）置为0，将非常大的项置为1，将非常小的项（负数）置为-1
    x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
    x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))



    #print(type(x_vals_test))

    # 定义batch的值,以5个数据作为一个批计算gradient
    batch_size = BATCH_SIZE

    # 声明占位符，input是LEN，traget是1
    x_data = tf.placeholder(shape=[None, LEN-1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    hidden_layer_nodes = HIDDEN_NUM
    A1 = tf.Variable(tf.random_normal(shape=[LEN-1, hidden_layer_nodes]))  # inputs -> hidden nodes
    b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))  # one biases for each hidden node
    A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))  # hidden inputs -> 1 output
    b2 = tf.Variable(tf.random_normal(shape=[1]))  # 1 bias for the output

    # 声明训练模型，使用relu激励函数，
    # 第一步：创建一个隐藏层输出
    # 第二部：创建训练模型的最后输出
    hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
    final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

    #添加L2正则化防止过拟合
    #tf.add_to_collection(tf.GraphKeys.WEIGHTS, A1)
    #tf.add_to_collection(tf.GraphKeys.WEIGHTS, A2)
    #regularizer = tf.contrib.layers.l2_regularizer(scale=5.0 / 50000)
    #reg_term = tf.contrib.layers.apply_regularization(regularizer)


    # 定义均方误差作为损失函数
    #loss = tf.reduce_mean(tf.square(y_target - final_output)+reg_term)
    loss = tf.reduce_mean(tf.square(y_target - final_output))

    # 声明优化算法，梯度下降，设置学习率为0.005
    # LEARNING_RATE = 0.005
    my_opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_step = my_opt.minimize(loss)

    # 初始化模型变量
    init = tf.global_variables_initializer()
    # print(init)
    session.run(init)

    # 迭代训练5000次，初始化两个列表存储training loss和testing loss
    # 每次迭代训练的时候，随机选择批量训练数据来拟合模型

    loss_vec = []
    test_loss = []
    for i in range(1000):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        session.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

        temp_loss = session.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(np.sqrt(temp_loss))

        test_temp_loss = session.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_loss.append(np.sqrt(test_temp_loss))
        # print('第' + str(i) + '轮的测试loss为' + str(np.sqrt(test_temp_loss)))
        if (i + 1) % 50 == 0:
            file.write('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss)+'\n')
            print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss)+'\n')
    flag = 1
    """if not (loss_vec[4999]<1 and loss_vec[4900]<1 and loss_vec[4950]<1):
        flag = 0"""

    #打印权值
    session.run(tf.Print(A1,[A1],summarize = LEN*HIDDEN_NUM))
    session.run(tf.Print(A2, [A2], summarize=HIDDEN_NUM))
    # 可视化loss
    plt.plot(loss_vec, 'k-', label='Train Loss')
    plt.plot(test_loss, 'r--', label='Test Loss')
    plt.title('Loss (MSE) per Generation')
    plt.legend(loc='upper right')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    if isShowFigure==1:
        plt.show()
    """if isShowFigure==0:
        return 0"""
    real_predict_data = []
    for t in TZFList1:
        real_predict_data.append(t[2])
    # real_predict_data.append(10)

    """for i in range(0,len(real_predict_data)):
        real_predict_data[i] = (real_predict_data[i]-minStrength[i])/(maxStrength[i]-minStrength[i])"""
    real_predict_data = np.array(real_predict_data)
    print(real_predict_data)
    # real_predict_data = np.nan_to_num(normalize_cols(real_predict_data))
    for i in range(len(CP)):
        real_predict_data[i] = (real_predict_data[i] - minList[i]) / (maxList[i] - minList[i])
        if real_predict_data[i] < 0:
            real_predict_data[i] = 0
        elif real_predict_data[i]>1:
            real_predict_data[i]=1
    real_predict_data = np.nan_to_num(real_predict_data)
    print(real_predict_data)

    hidden = session.run(hidden_output, feed_dict={x_data: [real_predict_data]})
    final1 = session.run(final_output, feed_dict={hidden_output: hidden})
    print(str(final1[0][0]) + '\n')
    file.write(str(final1[0][0]) + '\n')


    real_predict_data = []
    for t in TZFList2:
        real_predict_data.append(t[2])
    #real_predict_data.append(10)

    """for i in range(0,len(real_predict_data)):
        real_predict_data[i] = (real_predict_data[i]-minStrength[i])/(maxStrength[i]-minStrength[i])"""
    real_predict_data = np.array(real_predict_data)
    print(real_predict_data)
    #real_predict_data = np.nan_to_num(normalize_cols(real_predict_data))
    for i in range(len(CP)):
        real_predict_data[i] = (real_predict_data[i]-minList[i])/(maxList[i]-minList[i])
        if real_predict_data[i]<0:
            real_predict_data[i]=0
        elif real_predict_data[i]>1:
            real_predict_data[i]=1
    real_predict_data = np.nan_to_num(real_predict_data)
    print(real_predict_data)


    hidden = session.run(hidden_output,feed_dict = {x_data:[real_predict_data]})
    final2 = session.run(final_output, feed_dict={hidden_output: hidden})
    print(str(final2[0][0])+'\n')
    file.write(str(final2[0][0])+'\n')

    real_predict_data = []
    for t in TZFList5:
        real_predict_data.append(t[2])
    # real_predict_data.append(10)

    real_predict_data = np.array(real_predict_data)
    #real_predict_data = np.nan_to_num(normalize_cols(real_predict_data))
    print(real_predict_data)
    for i in range(len(CP)):
        real_predict_data[i] = (real_predict_data[i]-minList[i])/(maxList[i]-minList[i])
        if real_predict_data[i]<0:
            real_predict_data[i]=0
        elif real_predict_data[i]>1:
            real_predict_data[i]=1
    real_predict_data = np.nan_to_num(real_predict_data)
    print(real_predict_data)

    hidden = session.run(hidden_output, feed_dict={x_data: [real_predict_data]})
    final5 = session.run(final_output, feed_dict={hidden_output: hidden})
    print(str(final5[0][0])+'\n')
    file.write(str(final5[0][0])+'\n')


    RMSE = (final5-50)*(final5-50)+(final2-20)*(final2-20)
    global minRMSE
    if RMSE < minRMSE and flag ==1:

        minRMSE = RMSE
        # file = open('E:\\ANN_TRAIN\\LIBS_ANN v3' + element + '.txt', 'a')
        file.write("RMSE=" + str(RMSE) + '\n')
        file.write("best learning rate is:" + str(LEARNING_RATE) + '\n')
        file.write("best LAYER NUM is:" + str(HIDDEN_NUM) + '\n')
        #file.close()

    file.close()



    return maxList,minList

"""
使用min-max将特征向量放缩到0-1之间
"""
def normalize_cols(m):
    #print(m)
    col_max = m.max(axis=0)
    #print(col_max)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)




if __name__=='__main__':


    elementList = ['Ba']
    for element in elementList:
        minRMSE = 99999
        # 特征谱线集
        oldCP = getCP(element)
        # 特征谱线和重要系数的列表
        OldcpImportanceList = getCPandImportance(element)

        """
            @Todo:
                Apply fast forward selection or GA select proper line intensity
        """

        CP = []
        cpImportanceList = []
        for i in range(0,len(oldCP)):
            """if element=='Al':
                if i!=4 and i!=10 and i!=8:
                    CP.append(oldCP[i])
                    cpImportanceList.append(OldcpImportanceList[i])
            elif element=='Ba':
                if i!=0 and i!=1 and i!=2 and i!=3 and i!=4 and i!=5:
                    CP.append(oldCP[i])
                    cpImportanceList.append(OldcpImportanceList[i])
            elif element=='Cu':
                if i==3 or i==7 or i==8 or i==12 or i==15 or i==16:
                    CP.append(oldCP[i])
                    cpImportanceList.append(OldcpImportanceList[i])
            elif element=='Cd':
                if i==2:  #or i==7 or i==8  or i==10 :
                    CP.append(oldCP[i])
                    cpImportanceList.append(OldcpImportanceList[i])
            else:"""
            CP.append(oldCP[i])
            cpImportanceList.append(OldcpImportanceList[i])

        """"
        加载模拟的NIST数据
        """
        tenppmData = processDataList(loadFile('E:\\ANN data\\data\\' + element + '\\10ppm.txt'))
        tweppmData = processDataList(loadFile('E:\\ANN data\\data\\' + element + '\\20ppm.txt'))
        fifppmData = processDataList(loadFile('E:\\ANN data\\data\\' + element + '\\50ppm.txt'))

        tenppmData = findAllPeakValue(CP,tenppmData)
        tweppmData = findAllPeakValue(CP,tweppmData)
        fifppmData = findAllPeakValue(CP,fifppmData)

        print('10ppm real:')
        print('')
        rawData = rawDataToDataList('E:\\ANN data\\data\\realData\\'+element+'10.txt')
        TZFList1 = findAllPeakValue(CP, rawData)

        for TZF in TZFList1:
            print(TZF)

        print('20ppm real:')
        print('')

        rawData = rawDataToDataList('E:\\ANN data\\data\\realData\\'+element+'20.txt')
        TZFList2 = findAllPeakValue(CP, rawData)

        for TZF in TZFList2:
            print(TZF)

        print('50ppm real:')
        print('')

        rawData = rawDataToDataList('E:\\ANN data\\data\\realData\\'+element+'50.txt')
        TZFList5 = findAllPeakValue(CP, rawData)

        for TZF in TZFList5:
            print(TZF)

        timesList = []
        for i in range(0, len(CP)):
            #timesList.append(((TZFList1[i][2] / tenppmData[i][2])+(TZFList2[i][2]/tweppmData[i][2])+(TZFList5[i][2]/fifppmData[i][2]))/3)
            timesList.append(TZFList1[i][2] / tenppmData[i][2])
            #timesList.append(((TZFList1[i][2] / tenppmData[i][2])+(TZFList2[i][2]/tweppmData[i][2]))/2)

        print(timesList)

        # 获得神经网络训练集
        trainingData = []
        for i in range(1, 51):
            rawData = findAllPeakValue(CP,processDataList(loadFile('E:\\ANN data\\data\\' + element + '\\' + str(i) + 'ppm.txt')))
            rawData2 = findAllPeakValue(CP,processDataList(loadFile('E:\\ANN data\\data\\' + element + '\\' + str(i+0.5) + 'ppm.txt')))

            for j in range(0,len(CP)):
                rawData[j][2] = rawData[j][2] * timesList[j]
                rawData2[j][2] = rawData2[j][2] * timesList[j]

            oneData = []
            for j in range(0, len(CP)):
                oneData.append(rawData[j][2])
            oneData.append(i)

            trainingData.append(oneData)

            oneData1 = []
            for j in range(0, len(CP)):
                oneData1.append(rawData2[j][2])
            oneData1.append(i+0.5)

            trainingData.append(oneData1)

    #trainingDatat = np.array([x[0:11] for x in trainingData])
    #maxList1 = trainingDatat.max(axis=0)
    #minList1 = trainingDatat.min(axis=0)
    #Ba

    maxList, minList = trainANN(element, 7, 0.0001, 5, trainingData, 0)
    #Cu
    #maxList, minList = trainANN(element, 10, 0.0001, 5, trainingData, 0)
    #Cd
    ##maxList,minList = trainANN(element,18, 0.0001, 5, trainingData,0)
    #Pb
    #maxList, minList = trainANN(element, 18, 0.0001, 5, trainingData, 0)
    """for learningRate in [0.00001,0.00002,0.00003,0.00005,0.0001,0.0002,0.0003,0.0005,0.001,0.002,0.003,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.5,1,0.3]:
        for i in range(len(CP), 50):
            trainANN(element,i,learningRate,5,trainingData,0)"""





    """"[94370860.92715232, 74355971.8969555, 392.3635511829468, 321.4634146341464, 5.642718026401211, 0.3425428719902997, 1.7219195305951387, 5.2931675242996, 1.7454860252287945]
    [129635761.58940399, 107307551.9803229, 316.39379386487155, 235.4878048780488, 4.505937655302278, 0.25687749660121045, 1.7411630059736414, 5.472877358490566, 1.6498544259938954]
    [210839161.86094412, 180347668.86078954, 232.28970824980817, 176.8253983359186, 3.9845804090513384, 0.20619044933959188, 1.7438758959749787, 5.49163807890223, 1.5385280743025573]
    """
    
    





