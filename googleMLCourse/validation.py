#goal:
#1.使用多个特征而非单个特征来进一步提高模型的有效性
#2.调试模型输入数据中的问题
#3.使用测试数据集检查模型是否过拟合验证数据

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

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

# california_housing_dataframe = california_housing_dataframe.reindex(
#     np.random.permutation(california_housing_dataframe.index))


#对数据集进行预处理
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

    #创建复合特征
    processed_features["rooms_per_person"] = (california_housing_dataframe["total_rooms"] /california_housing_dataframe["population"])

    return processed_features

#将median_house_value/1000
def preprocess_targets(california_housing_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["median_house_value"] = (california_housing_dataframe["median_house_value"] / 1000.0)
  return output_targets

#选择训练集和训练集的target
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_examples.describe()

training_targets = preprocess_targets(california_housing_dataframe.head(12000))
training_targets.describe()
