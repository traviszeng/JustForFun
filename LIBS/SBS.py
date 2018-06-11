"""
序列后向选择算法（SBS）
created by Travis on 2018/6/8
"""
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class SBS():
    """
    构造函数
    """
    def __init__(self,
                 estimator,
                 k_features,
                 #scoring = accuracy_score,#todo 此处的accuracy_score函数用于classification 需要重新写一个评估函数 https://www.cnblogs.com/harvey888/p/6964741.html
                 scoring = mean_squared_error, #MSE 均方误差
                 test_size = 0.25,
                 random_state = 1):

        self.scoring = scoring
        self.estimator = estimator #在计算score的时候使用estimator来predict
        self.k_features = k_features
        self.random_state = random_state
        self.test_size = test_size

    def fit(self, X_train, y_train,X_test,y_test):

        #仅使用原来的test set再进一步划分
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
        #                                                    random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


"""
usage eg.


    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # selecting features
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)
    
    # plotting performance of feature subsets
    k_feat = [len(k) for k in sbs.subsets_]
    
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.02])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.tight_layout()
    # plt.savefig('images/04_08.png', dpi=300)
    plt.show()
"""

