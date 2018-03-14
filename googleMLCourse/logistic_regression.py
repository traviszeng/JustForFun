"""
Task:
1.将（在之前的练习中构建的）房屋价值中位数预测模型重新构建为二元分类模型
2.比较逻辑回归与线性回归解决二元分类问题的有效性
"""
"""
与在之前的练习中一样，我们将使用加利福尼亚州住房数据集，
但这次我们会预测某个城市街区的住房成本是否高昂，从而将其转换成一个二元分类问题。
此外，我们还会暂时恢复使用默认特征
"""
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


def preprocess_features(california_housing_dataframe):
  """Prepares input features from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
  return processed_features


#创建一个新的布尔特征作为目标
def preprocess_targets(california_housing_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Create a boolean categorical feature representing whether the
  # medianHouseValue is above a set threshold.
  output_targets["median_house_value_is_high"] = (
    california_housing_dataframe["median_house_value"] > 265000).astype(float)
  return output_targets



if __name__=="__main__":
    
    tf.logging.set_verbosity(tf.logging.ERROR)
    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.1f}'.format

    california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

    #将数据随机化非常重要，不然可能会导致训练集和验证集的分布情况不一
    california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index))
    
    #选择训练集和验证集
    training_examples = preprocess_features(california_housing_dataframe.head(12000))
    #training_examples.describe()

    training_targets = preprocess_targets(california_housing_dataframe.head(12000))
    #training_targets.describe()

    validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
    #validation_examples.describe()

    validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
    #validation_targets.describe()
