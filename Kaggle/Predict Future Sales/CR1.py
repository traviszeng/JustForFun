import numpy as np
import pandas as pd
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',100)

from itertools import product
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline


from xgboost import XGBRegressor
from xgboost import plot_importance

def plot_features(booster,figsize):
    fig,ax = plt.subplot(1,1,figsize=figsize)
    return plot_importance(booster=booster,ax=ax)

import time
import sys
import gc
import pickle

#print(sys.version_info)

#加载数据
items = pd.read_csv('items.csv')
shops = pd.read_csv('shops.csv')
cats = pd.read_csv('item_categories.csv')
train = pd.read_csv('sales_train.csv.gz')
#将ID设为index
test = pd.read_csv('test.csv.gz').set_index('ID')

#用seaborn的boxplot查看变量分布情况
plt.figure(figsize=(10,4))
plt.xlim(train.item_cnt_day.min(),train.item_cnt_day.max()*1.1)
sns.boxplot(x=train.item_cnt_day)

plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price)

#决定移除item_cnt_day>1000和item_price>1000000的记录
train = train[train.item_cnt_day<100001]
train = train[train.item_price<1001]

#有一条记录中的item_price小于零 将其移除
train =train[train.item_price>0]

#有一些shop是重名的 根据商店名更正相应的唯一的shop_id
# Якутск Орджоникидзе, 56
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11