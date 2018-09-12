"""
    Python machine learning chapter 6 code
    Model evaluation and parameters tuning

"""

from IPython.display import Image
import pandas as pd

from sklearn.preprocessing import LabelEncoder


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


if __name__=='__main__':
    main()