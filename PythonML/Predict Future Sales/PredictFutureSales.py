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
#print(transactions.head(5))
#print(items.head(5))
#print(item_categories.head(5))
#print(shops.head(5))


df = pd.DataFrame(transactions['date'])
#print(df)
print(str(df.iloc[0,0]).split('.'))
print(transactions.loc[1:2])

#print(transactions['date'])