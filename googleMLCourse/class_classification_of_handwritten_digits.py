"""
训练线性模型和神经网络，以对传统 MNIST 数据集中的手写数字进行分类
比较线性分类模型和神经网络分类模型的效果
可视化神经网络隐藏层的权重

"""

import glob
import io
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

#将特征列以及标签列抽取出来
def parse_labels_and_features(dataset):
    """Extracts labels and features.

    This is a good place to scale or transform the features if needed.

    Args:
      dataset: A Pandas `Dataframe`, containing the label on the first column and
        monochrome pixel values on the remaining columns, in row major order.
    Returns:
      A `tuple` `(labels, features)`:
        labels: A Pandas `Series`.
        features: A Pandas `DataFrame`.
    """
    labels = dataset[0]

    # DataFrame.loc index ranges are inclusive at both ends.
    features = dataset.loc[:, 1:784]
    # Scale the data to [0, 1] by dividing out the max value, 255.
    features = features / 255

    return labels, features


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.1f}'.format

    mnist_dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/mnist_train_small.csv",
        sep=",",
        header=None)

    # Use just the first 10,000 records for training/validation
    mnist_dataframe = mnist_dataframe.head(10000)

    mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
    mnist_dataframe.head()

    #第一列中包含类别标签。其余列中包含特征值，每个像素对应一个特征值，有 28×28=784 个像素值，其中大部分像素值都为零；
    #您也许需要花一分钟时间来确认它们不全部为零。
    training_targets, training_examples = parse_labels_and_features(mnist_dataframe[:7500])
    training_examples.describe()

    validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe[7500:10000])
    validation_examples.describe()


    #显示一个随机样本及其对应的标签
    rand_example = np.random.choice(training_examples.index)
    _, ax = plt.subplots()
    ax.matshow(training_examples.loc[rand_example].values.reshape(28, 28))
    ax.set_title("Label: %i" % training_targets.loc[rand_example])
    ax.grid(False)