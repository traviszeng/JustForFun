import os
import matplotlib.pyplot as plt


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



elements = ['Ba','Cd','Cu','Pb']
for element in elements:
    CP = getCP(element)

    trainingData = []
    for i in range(1, 51):
        rawData = findAllPeakValue(CP, processDataList(
            loadFile('E:\\ANN data\\data\\' + element + '\\' + str(i) + 'ppm.txt')))
        rawData2 = findAllPeakValue(CP, processDataList(
            loadFile('E:\\ANN data\\data\\' + element + '\\' + str(i + 0.5) + 'ppm.txt')))

        for j in range(0, len(CP)):
            rawData[j][2] = rawData[j][2]
            rawData2[j][2] = rawData2[j][2]

        oneData = []
        for j in range(0, len(CP)):
            oneData.append(rawData[j][2])
        oneData.append(i)

        trainingData.append(oneData)

        oneData1 = []
        for j in range(0, len(CP)):
            oneData1.append(rawData2[j][2])
        oneData1.append(i + 0.5)

        trainingData.append(oneData1)



    color=['r','b','k','g','y','m']

    label = []
    Y = []
    for i in range(1,51):
        Y.append(i)
        Y.append(i+0.5)
    for i in range(0,len(CP)):
        X = []
        label.append(str(CP[i]))

        for t in trainingData:
            X.append(t[i])
            print(t)

        plt.plot(X,Y,color[i%len(color)],label = label[i])

    plt.show()




