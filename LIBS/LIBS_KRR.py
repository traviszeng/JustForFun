"""

    Kernel ridge regression demo
    Created on 2018/9/16
"""

from sklearn.kernel_ridge import KernelRidge
import numpy as np

def main():

    n_samples,n_features = 10,5
    rng = np.random.RandomState(0)
    y = rng.randn(n_samples)
    x = rng.randn(n_samples,n_features)

    clf = KernelRidge(alpha=1.0)

    clf.fit(x,y)
    print(x[0])
    print(y[0])
    print(clf.predict([x[0]]))
    print(clf.score(x,y))

if __name__=='__main__':
    main()