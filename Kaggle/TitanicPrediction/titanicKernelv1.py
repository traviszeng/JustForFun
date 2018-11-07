import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',100)

train = pd.read_csv("train.csv")
#将年龄中的nan变为0
train.loc[np.isnan(train.Age),'Age']=0

#female为0 male为1
train.loc[train.Sex=="male",'Sex'] = 1
train.loc[train.Sex=='female','Sex']= 0


#处理乘船地点
#C---0
#S---1
#Q---2
train.loc[train.Embarked=='C','Embarked']=0
train.loc[train.Embarked=='S','Embarked']=1
train.loc[train.Embarked=='Q','Embarked']=2

#处理名字 分成title和名字
train['Title'] = train['Name'].str.split(',').map(lambda x:x[1])
train['Title'] = train['Title'].str.split('.').map(lambda x:x[0])

train['Title_type'] = LabelEncoder().fit_transform(train['Title'])
print(train.loc['Title_type'==1])
print(train.head(20))