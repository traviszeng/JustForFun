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
from sklearn.model_selection import validation_curve

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC



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


    #绘制样本大小和训练准确率以及交叉验证准确率之间关系
    print('4.Testing learning curve '+20*'-')
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

    #绘制的是模型参数与准确率之间的关系
    print('5.Testing validation curve'+20*'-')
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(
        estimator=pipe_lr,
        X=X_train,
        y=y_train,
        param_name='logisticregression__C',
        param_range=param_range,
        cv=10)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    # plt.savefig('images/06_06.png', dpi=300)
    plt.show()


    #网格搜索调优超参
    print('6.Grid Search testing '+20*'-')

    pipe_svc = make_pipeline(StandardScaler(),
                             SVC(random_state=1))

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    #字典的方式指定几个超参
    param_grid = [{'svc__C': param_range,
                   'svc__kernel': ['linear']},
                  {'svc__C': param_range,
                   'svc__gamma': param_range,
                   'svc__kernel': ['rbf']}]

    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)

    clf= gs.best_estimator_
    clf.fit(X_train,y_train)

    print('Test accuracy: %3f' % clf.score(X_test,y_test))
    #也可以使用RandomizedSearchCV 抽取出随机参数组合

    #通过嵌套交叉验证选择不同机器学习算法
    print('7.Test nested cross validation'+20*'-')

    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=2)

    scores = cross_val_score(gs,X_train,y_train,scoring='accuracy',cv=5)

    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))

    #使用嵌套交叉验证方法比较SVM和简单决策树模型
    from sklearn.tree import DecisionTreeClassifier
    #只调优树的深度
    gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                      param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                      scoring='accuracy',
                      cv=2)

    scores = cross_val_score(gs, X_train, y_train,
                             scoring='accuracy', cv=5)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                          np.std(scores)))


    #混淆矩阵
    print('8.Confusion matrix'+20*'-')
    from sklearn.metrics import confusion_matrix

    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)

    #用matplotlib中的matshow标识出来
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()

    le.transform(['M', 'B'])
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)

    from sklearn.metrics import precision_score, recall_score, f1_score

    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

    #用自己定义的评分来找到最优超参组合
    from sklearn.metrics import make_scorer

    scorer = make_scorer(f1_score, pos_label=0)

    c_gamma_range = [0.01, 0.1, 1.0, 10.0]

    param_grid = [{'svc__C': c_gamma_range,
                   'svc__kernel': ['linear']},
                  {'svc__C': c_gamma_range,
                   'svc__gamma': c_gamma_range,
                   'svc__kernel': ['rbf']}]

    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring=scorer,
                      cv=10,
                      n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)

    print('9.Print a ROC'+20*'-')
    from sklearn.metrics import roc_curve, auc
    from scipy import interp

    pipe_lr = make_pipeline(StandardScaler(),
                            PCA(n_components=2),
                            LogisticRegression(penalty='l2',
                                               random_state=1,
                                               C=100.0))

    X_train2 = X_train[:, [4, 14]]

    cv = list(StratifiedKFold(n_splits=3,
                              random_state=1).split(X_train, y_train))

    fig = plt.figure(figsize=(7, 5))

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas = pipe_lr.fit(X_train2[train],
                             y_train[train]).predict_proba(X_train2[test])

        fpr, tpr, thresholds = roc_curve(y_train[test],
                                         probas[:, 1],
                                         pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr,
                 tpr,
                 label='ROC fold %d (area = %0.2f)'
                       % (i + 1, roc_auc))

    plt.plot([0, 1],
             [0, 1],
             linestyle='--',
             color=(0.6, 0.6, 0.6),
             label='random guessing')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 0, 1],
             [0, 1, 1],
             linestyle=':',
             color='black',
             label='perfect performance')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc="lower right")

    plt.tight_layout()
    # plt.savefig('images/06_10.png', dpi=300)
    plt.show()

    #多类别分类的评价标准
    print('10. The scoring metrics for multiclass classification'+20*'-')
    pre_scorer = make_scorer(score_func=precision_score,
                             pos_label=1,
                             greater_is_better=True,
                             average='micro')

    #Dealing with class imbalances
    X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
    y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

    y_pred = np.zeros(y_imb.shape[0])
    np.mean(y_pred == y_imb) * 100

    from sklearn.utils import resample

    print('Number of class 1 samples before:', X_imb[y_imb == 1].shape[0])

    X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                        y_imb[y_imb == 1],
                                        replace=True,
                                        n_samples=X_imb[y_imb == 0].shape[0],
                                        random_state=123)

    print('Number of class 1 samples after:', X_upsampled.shape[0])

    X_bal = np.vstack((X[y == 0], X_upsampled))
    y_bal = np.hstack((y[y == 0], y_upsampled))

    y_pred = np.zeros(y_bal.shape[0])
    np.mean(y_pred == y_bal) * 100





if __name__=='__main__':
    main()