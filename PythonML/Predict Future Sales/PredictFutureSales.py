import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from grader import Grader

transactions = pd.read_csv('sales_train.csv.gz')
items = pd.read_csv('items.csv')
item_categories = pd.read_csv('item_categories.csv')
shops = pd.read_csv('shops.csv')

#设置最多列数20
pd.set_option('display.max_columns',20)
#print(transactions.head(50))
#print(items.head(5))
#print(item_categories.head(5))
#print(shops.head(5))

#print(shops)

"""
#df = pd.DataFrame(transactions['date'])
#print(df)
#print(transactions)

del transactions['date']
year = []
month = []
day = []
for i in range(0,len(transactions)):
    d,m,y = str(df.iloc[i,0]).split('.')
    year.append(int(y))
    month.append(int(m))
    day.append(int(d))

transactions['year'] = pd.Series(year)
transactions['month'] = pd.Series(month)
transactions['day'] = pd.Series(day)

selected = transactions[(transactions['year']==2014) & (transactions['month']==9)]
"""

#print(selected.head(5))
#print(transactions.loc[1:2])
#print(transactions.head(5))
#print(transactions['date'])
#task 1
selected = transactions[transactions['date_block_num']==20]

sum = 0.0
for i in range(0,len(selected)):
    #print(sum)
    sum+=float(selected.iloc[i,4]*selected.iloc[i,5])

print(sum)

#task2
selected2 = transactions[(transactions['date_block_num']<20) & (transactions['date_block_num']>16)]
item_categories['sum']=0.0

for i in range(0,len(selected2)):
    item_categories.iloc[items.iloc[selected2.iloc[i,3],2],2]+=(selected2.iloc[i,4]*selected2.iloc[i,5])

print(item_categories['sum'].idxmax())

#task3
print(len(items))