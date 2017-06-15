from kmeans import kmeans
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# class accuracy(object):
#     def __init__(self):
#         self.dataDF = pd.read_csv('./data.csv', header=None)

#     def get_result(self):
#         training_data, testing_data = train_test_split(self.dataDF, test_size=0.5)
#         kmeans_obj = kmeans(training_data)
#         kmeans_obj.

class visualization(object):
    def __init__(self):
        self.dataDF = pd.read_csv('./data.csv', header=None)

    # def get_result(self):
    #     self.PCA(self.dataDF)
    #     training_data, testing_data = train_test_split(self.dataDF, test_size=0.5)

    #     kmeans_obj = kmeans(training_data)
    #     kmeans_obj.

    def PCA(self):
        # print(self.dataDF.loc[:, 1:])
        pca = PCA(n_components=2)
        pca.fit(self.dataDF.loc[:, 1:])
        featReduced = pca.fit_transform(self.dataDF.loc[:, 1:])
        print(featReduced)

if __name__ == '__main__':
    obj = visualization()
    obj.PCA()    


# obj = kmeans('./data.csv')
# result = obj.get_result(3)
# print(result[2])
