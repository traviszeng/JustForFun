"""
序列后向选择算法（SBS）
created by Travis on 2018/6/8
"""

from itertools import combinations
import numpy as np
from sklearn.metrics import mean_squared_error
import math

class SBS():
    """
    构造函数
    """
    def __init__(self,
                 estimator,
                 element,
                 k_features,
                 #scoring = accuracy_score,此处的accuracy_score函数用于classification 需要重新写一个评估函数 https://www.cnblogs.com/harvey888/p/6964741.html
                 #hidden_layer,
                 learning_rate,
                 scoring = mean_squared_error, #MSE 均方误差
                 test_size = 0.25,
                 random_state = 1,
                 ):

        self.scoring = scoring
        self.estimator = estimator #在计算score的时候使用estimator来predict
        self.k_features = k_features
        self.random_state = random_state
        self.test_size = test_size
        self.element = element
        #self.hidden_layer = hidden_layer
        self.learning_rate = learning_rate
        #self.num = 0


    def fit(self, X_train, y_train,y_test):

        #仅使用原来的test set再进一步划分
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
        #                                                    random_state=self.random_state)
        print(X_train.shape[1])
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))

        self.subsets_ = [self.indices_]
        score,besthidden = self._calc_score(X_train, y_train,
                                 y_test, self.indices_)
        self.scores_ = [score]
        #self.hiddenlayernum_ =[besthidden]

        while dim > self.k_features:
            scores = []
            subsets = []
            #besthiddenlist = []

            for p in combinations(self.indices_, r=dim - 1):
                score,besthidden = self._calc_score(X_train, y_train,
                                          y_test, p)
                scores.append(score)
                subsets.append(p)
                #besthiddenlist.append(besthidden)


            best = np.argmin(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            #self.hiddenlayernum_.append(besthiddenlist[best])

            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, y_test, indices):
        """self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])"""
        print("X_train indices:")
        print(indices)
        print(type(indices))
        #print(X_train[:,indices])
        trainDataTemp = []
        #X_traintemp = X_train[:,indices]
        """for u in range(0,len(X_train)):
            temp = X_train[u].tolist()
            temp.append(y_train[u])
            trainDataTemp.append(temp)"""
        bestScore = 999999
        bestHiddenLayer = 0
        #for HIDDEN in range(len(indices),40):
        #使用公式求得合适的隐含层数目
        m = len(indices)
        n = 1
        HIDDEN = math.sqrt(0.43*m*n+0.12*n*n+2.54*m+0.77*n+0.35)+0.51
        y_pred = self.estimator(self.element,int(HIDDEN),self.learning_rate,5,X_train,y_train,0,indices)
        score = self.scoring(y_test, y_pred)
        if score<bestScore:
            bestScore = score
            #bestHiddenLayer = HIDDEN
        #self.num+=1
        return bestScore,bestHiddenLayer

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

