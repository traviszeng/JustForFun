"""
    Python machine learning chapter 6 code
    Model evaluation and parameters tuning

"""

from IPython.display import Image
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve



def main():
    """
    main function

    :return:
    """
    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                     'machine-learning-databases'
                     '/breast-cancer-wisconsin/wdbc.data', header=None)
    print(df.head(5))


    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    #'M'-->1 'B'-->0
    y = le.fit_transform(y)

    #print(X)
    #print(y)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        stratify=y,
                                                        random_state=1)

    #使用流水线将StandardScaler、PCA以及LogisticRegression串联起来
    print("1.using pipeline with standardScaler and PCA"+20*'-')
    pipe_lr = make_pipeline(StandardScaler(),
                            PCA(n_components=2),
                            LogisticRegression(random_state=1))

    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

    #分层k折交叉验证，类别比例在每个分块中得以保持。
    print('2.K-fold cross-validation'+20*'-')

    kfold = StratifiedKFold(n_splits=10,random_state=1).split(X_train,y_train)

    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k + 1,
                                                         np.bincount(y_train[train]), score))

    print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


    #kfold交叉验证平分
    print('3.k-fold cross validation score testing'+20*'-')
    scores = cross_val_score(estimator=pipe_lr,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             n_jobs=-1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


    print('4.Testing learning curve and validation curve'+20*'-')
    #a demo of learning curve and validation curve
    pipe_lr = make_pipeline(StandardScaler(),
                            LogisticRegression(penalty='l2', random_state=1))

    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                                                            X=X_train,
                                                            y=y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            cv=10,
                                                            n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.03])
    plt.tight_layout()

    plt.show()








if __name__=='__main__':
    main()