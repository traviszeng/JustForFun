"""
support vector Regression implementation

see:http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html


"""
from sklearn.svm import SVR
import numpy as np

def main():
    """

    main function
    :return:
    """
    n_samples,n_features = 10,5

    np.random.seed(0)
    y = np.random.randn(n_samples)
    x = np.random.randn(n_samples,n_features)
    print('x:')
    print(x)
    print('Y =')
    print(y)

    clf = SVR(C=1.0,epsilon=0.2)
    clf.fit(x,y)

    print(clf.predict([x[0]]))

    print(clf.score(x,y))



if __name__=='__main__':
    main()