from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os


print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))

sns.set(style='white', context='notebook', palette='deep')
pylab.rcParams['figure.figsize'] = 12,8
warnings.filterwarnings('ignore')
mpl.style.use('ggplot')
sns.set_style('white')


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Scatter plot Purpose To identify the type of relationship (if any) between two quantitative variables
# Modify the graph above by assigning each species an individual color.
g = sns.FacetGrid(train, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();

#This gives us a much clearer idea of the distribution of the input attributes:
train.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False)
plt.figure()



# To plot the species data using a box plot:
sns.boxplot(x="Fare", y="Age", data=test )
#plt.show()

# Use Seaborn's striplot to add data points on top of the box plot
# Insert jitter=True so that the data points remain scattered and not piled into a verticle line.
# Assign ax to each axis, so that each plot is ontop of the previous axis.

ax= sns.boxplot(x="Fare", y="Age", data=train)
ax= sns.stripplot(x="Fare", y="Age", data=train, jitter=True, edgecolor="gray")

# Tweek the plot above to change fill and border color color using ax.artists.
# Assing ax.artists a variable name, and insert the box number into the corresponding brackets

ax= sns.boxplot(x="Fare", y="Age", data=train)
ax= sns.stripplot(x="Fare", y="Age", data=train, jitter=True, edgecolor="gray")

boxtwo = ax.artists[2]
boxtwo.set_facecolor('red')
boxtwo.set_edgecolor('black')
boxthree=ax.artists[1]
boxthree.set_facecolor('yellow')
boxthree.set_edgecolor('black')


#It looks like perhaps two of the input variables have a Gaussian distribution. This is useful to note as we can use algorithms that can exploit this assumption.
train.hist(figsize=(15,20))

f,ax=plt.subplots(1,2,figsize=(20,10))
train[train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train[train['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)


f,ax=plt.subplots(1,2,figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train,ax=ax[1])
ax[1].set_title('Survived')


f,ax=plt.subplots(1,2,figsize=(18,8))
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=train,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')



"""
Now we can look at the interactions between the variables.

First, let’s look at scatterplots of all pairs of attributes. 

This can be helpful to spot structured relationships between input variables.
"""
#两两feature之间的关系
pd.plotting.scatter_matrix(train,figsize=(10,10))


pd.plotting.scatter_matrix(train,figsize=(10,10))

plt.figure(figsize=(7,4))
sns.heatmap(train.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())



sns.heatmap(train.corr(),annot=False,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
#plt.show()


"""
Data Preprocessing
"""
print(train.isnull().sum())

print(train['Age'].unique())

print(train["Pclass"].value_counts())

cols = train.columns
features = cols[0:12]
labels = cols[4]
print(features)
print(labels)