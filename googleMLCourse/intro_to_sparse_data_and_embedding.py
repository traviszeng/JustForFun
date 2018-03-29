"""
学习目标：

将影评字符串数据转换为稀疏特征矢量
使用稀疏特征矢量实现情感分析线性模型
通过将数据投射到二维空间的嵌入来实现情感分析 DNN 模型
将嵌入可视化，以便查看模型学到的词语之间的关系

"""

"""
我们根据这些数据训练一个情感分析模型，以预测某条评价总体上是好评（标签为 1）还是差评（标签为 0）。

为此，我们会使用词汇表（即我们预计将在数据中看到的每个术语的列表），将字符串值 terms 转换为特征矢量。
在本练习中，我们创建了侧重于一组有限术语的小型词汇表。
其中的大多数术语明确表示是好评或差评，但有些只是因为有趣而被添加进来。

词汇表中的每个术语都与特征矢量中的一个坐标相对应。
为了将样本的字符串值 terms 转换为这种矢量格式，我们按以下方式处理字符串值：
如果该术语没有出现在样本字符串中，则坐标值将为 0；如果出现在样本字符串中，则值为 1。未出现在该词汇表中的样本中的术语将被弃用。


"""

import collections
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from sklearn import metrics



""""
配置输入管道，以将数据导入 TensorFlow 模型中。
我们可以使用以下函数来解析训练数据和测试数据（格式为 TFRecord），然后返回一个由特征和相应标签组成的字典。
"""
def _parse_function(record):
  """Extracts features and labels.
  
  Args:
    record: File path to a TFRecord file    
  Returns:
    A `tuple` `(labels, features)`:
      features: A dict of tensors representing the features
      labels: A tensor with the corresponding labels.
  """
  features = {
    "terms": tf.VarLenFeature(dtype=tf.string), # terms are strings of varying lengths
    "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32) # labels are 0 or 1
  }
  
  parsed_features = tf.parse_single_example(record, features)
  
  terms = parsed_features['terms'].values
  labels = parsed_features['labels']

  return  {'terms':terms}, labels

#现在，我们构建一个正式的输入函数，可以将其传递给 TensorFlow Estimator 对象的 train() 方法。
# Create an input_fn that parses the tf.Examples from the given files,
# and split them into features and targets.
def _input_fn(input_filenames, num_epochs=None, shuffle=True):
    # Same code as above; create a dataset and map features and labels
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    # Our feature data is variable-length, so we pad and batch
    # each field of the dataset structure to whatever size is necessary
    ds = ds.padded_batch(25, ds.output_shapes)

    ds = ds.repeat(num_epochs)

    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


#使用LinearClassifier来评估
def useLinearClassifier():

    """
    第一个模型，我们将使用 54 个信息性术语来构建 LinearClassifier 模型
    """
    #为我们的术语构建特征列。categorical_column_with_vocabulary_list 函数可使用“字符串-特征矢量”映射来创建特征列。
    # 54 informative terms that compose our model vocabulary 
    informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                         "excellent", "poor", "boring", "awful", "terrible",
                         "definitely", "perfect", "liked", "worse", "waste",
                         "entertaining", "loved", "unfortunately", "amazing",
                         "enjoyed", "favorite", "horrible", "brilliant", "highly",
                         "simple", "annoying", "today", "hilarious", "enjoyable",
                         "dull", "fantastic", "poorly", "fails", "disappointing",
                         "disappointment", "not", "him", "her", "good", "time",
                         "?", ".", "!", "movie", "film", "action", "comedy",
                         "drama", "family", "man", "woman", "boy", "girl")

    terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms", vocabulary_list=informative_terms)

    #构建 LinearClassifier，在训练集中训练该模型，并在评估集中对其进行评估。
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    feature_columns = [terms_feature_column]
    print(feature_columns)

    classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        optimizer=my_optimizer,
    )

    classifier.train(
        input_fn=lambda: _input_fn([train_path]),
        steps=1000)

    evaluation_metrics = classifier.evaluate(
        input_fn=lambda: _input_fn([train_path]),
        steps=1000)
    print("Training set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")

    evaluation_metrics = classifier.evaluate(
        input_fn=lambda: _input_fn([test_path]),
        steps=1000)

    print("Test set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")

#使用深度神经网络 (DNN) 模型
def useDNNClassifier():
    # 为我们的术语构建特征列。categorical_column_with_vocabulary_list 函数可使用“字符串-特征矢量”映射来创建特征列。
    # 54 informative terms that compose our model vocabulary
    informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                         "excellent", "poor", "boring", "awful", "terrible",
                         "definitely", "perfect", "liked", "worse", "waste",
                         "entertaining", "loved", "unfortunately", "amazing",
                         "enjoyed", "favorite", "horrible", "brilliant", "highly",
                         "simple", "annoying", "today", "hilarious", "enjoyable",
                         "dull", "fantastic", "poorly", "fails", "disappointing",
                         "disappointment", "not", "him", "her", "good", "time",
                         "?", ".", "!", "movie", "film", "action", "comedy",
                         "drama", "family", "man", "woman", "boy", "girl")

    terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms",
                                                                                     vocabulary_list=informative_terms)

    my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    """
    将使用嵌入列来实现 DNN 模型。嵌入列会将稀疏数据作为输入，并返回一个低维度密集矢量作为输出
    通过将数据投射到二维空间的 embedding_column 来为模型定义特征列（如需详细了解 embedding_column 的函数签名，请参阅相关 TF 文档）。
    定义符合以下规范的 DNNClassifier：
        具有两个隐藏层，每个包含 20 个单元
        采用学习速率为 0.1 的 AdaGrad 优化方法
        gradient_clip_norm 值为 5.0
    """
    terms_embedding_column = tf.feature_column.embedding_column(terms_feature_column, dimension=2)
    feature_columns = [terms_embedding_column]
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        optimizer=my_optimizer
    )

    """classifier = tf.estimator.DNNClassifier(  #
        feature_columns=[tf.feature_column.indicator_column(terms_feature_column)],  #
        hidden_units=[20, 20],  #
        optimizer=my_optimizer,  #
    )"""

    classifier.train(
        input_fn=lambda: _input_fn([train_path]),
        steps=1000)

    evaluation_metrics = classifier.evaluate(
        input_fn=lambda: _input_fn([train_path]),
        steps=1000)
    print("Training set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")

    evaluation_metrics = classifier.evaluate(
        input_fn=lambda: _input_fn([test_path]),
        steps=1000)

    print("Test set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")

    #该模型中的张量
    print(classifier.get_variable_names())
    """
    ['dnn/hiddenlayer_0/bias',
    'dnn/hiddenlayer_0/bias/t_0/Adagrad',
    'dnn/hiddenlayer_0/kernel',
     'dnn/hiddenlayer_0/kernel/t_0/Adagrad',
     'dnn/hiddenlayer_1/bias',
     'dnn/hiddenlayer_1/bias/t_0/Adagrad',
     'dnn/hiddenlayer_1/kernel',
     'dnn/hiddenlayer_1/kernel/t_0/Adagrad',
     'dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights',
     'dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights/t_0/Adagrad',
     'dnn/logits/bias',
     'dnn/logits/bias/t_0/Adagrad',
     'dnn/logits/kernel',
     'dnn/logits/kernel/t_0/Adagrad',
     'global_step']
     
     我们可以看到这里有一个嵌入层：'dnn/input_from_feature_columns/input_layer/terms_embedding/...'。（顺便说一下，有趣的是，该层可以与模型的其他层一起训练，就像所有隐藏层一样。）
    """
    embedding_matrix = classifier.get_variable_value(
        'dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')

    for term_index in range(len(informative_terms)):
        # Create a one-hot encoding for our term.  It has 0's everywhere, except for
        # a single 1 in the coordinate that corresponds to that term.
        term_vector = np.zeros(len(informative_terms))
        term_vector[term_index] = 1
        # We'll now project that one-hot vector into the embedding space.
        embedding_xy = np.matmul(term_vector, embedding_matrix)
        plt.text(embedding_xy[0],
                 embedding_xy[1],
                 informative_terms[term_index])

    # Do a little set-up to make sure the plot displays nicely.
    plt.rcParams["figure.figsize"] = (12, 12)
    plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
    plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
    plt.show()

"""
更改超参数或使用其他优化工具，比如 Adam（通过遵循这些策略，您的准确率可能只会提高一两个百分点）。
向 informative_terms 中添加其他术语。此数据集有一个完整的词汇表文件，其中包含 30716 个术语，您可以在以下位置找到该文件：https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/terms.txt 您可以从该词汇表文件中挑选出其他术语，
也可以通过 categorical_column_with_vocabulary_file 特征列使用整个词汇表文件。
"""
def inprovementOfDNN():
    # Create a feature column from "terms", using a full vocabulary file.
    informative_terms = None
    with open("https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/terms.txt", 'r') as f:
        # Convert it to set first to remove duplicates.
        informative_terms = list(set(f.read().split()))

    terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms",
                                                                                     vocabulary_list=informative_terms)

    terms_embedding_column = tf.feature_column.embedding_column(terms_feature_column, dimension=2)
    feature_columns = [terms_embedding_column]

    my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        optimizer=my_optimizer
    )

    classifier.train(
        input_fn=lambda: _input_fn([train_path]),
        steps=1000)

    evaluation_metrics = classifier.evaluate(
        input_fn=lambda: _input_fn([train_path]),
        steps=1000)
    print("Training set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")

    evaluation_metrics = classifier.evaluate(
        input_fn=lambda: _input_fn([test_path]),
        steps=1000)

    print("Test set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")



if __name__=='__main__':

    #我们导入依赖项并下载训练数据和测试数据。tf.keras 中包含一个文件下载和缓存工具，我们可以用它来检索数据集。
    tf.logging.set_verbosity(tf.logging.ERROR)
    train_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
    train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
    test_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
    test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)

    #为了确认函数是否能正常运行，我们为训练数据构建一个 TFRecordDataset，并使用上述函数将数据映射到特征和标签。

    # Create the Dataset object
    ds = tf.data.TFRecordDataset(train_path)
    # Map features and labels with the parse function
    ds = ds.map(_parse_function)

    #从训练集中获取第一个样本
    n = ds.make_one_shot_iterator().get_next()
    sess = tf.Session()
    print(sess.run(n))

    #用LinearClassifier尝试
    #useLinearClassifier()

    #用DNNClassifier尝试
    #useDNNClassifier()

    #改进模型的效果
    inprovementOfDNN()




