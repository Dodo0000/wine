# @Author: Shane Yu  @Date Created: April 25, 2017
import numpy as np
import pandas as pd
from numpy.linalg import inv, det
import math
import operator


class BayesWine(object):
    """
    constructor:
        Training data file's path

    getDecision function:
        Input: a line of the testing data
        Output: a class number
    """
    def __init__(self, trainingDataPath):
        np.set_printoptions(precision=6, suppress=True, linewidth=1000)
        pd.set_option('display.width', 1000)
        with open(trainingDataPath, 'r') as rf:
            self.pdData = pd.read_csv(rf, header=None) # [178 rows x 14 columns]

    def getCoVarianceMatrix(self, classNum):
        return self.pdData.loc[self.pdData[0]==classNum, 1:].cov()

    def getMeanVector(self, classNum):
        return self.pdData.loc[self.pdData[0]==classNum, 1:].mean()

    def getDecision(self, inputString):
        inputVector = np.array(inputString.split(',')[1:], dtype=np.float)
        dataCount = {1:59, 2:71, 3:48} # 每個class資料筆數
        decisionValue = dict()
        for num in range(1, 4): # 計算三個class的decisionValue
            decisionValue[num] = (1/2) * np.dot(np.dot(np.subtract(inputVector, self.getMeanVector(num)).T, inv(self.getCoVarianceMatrix(num))), np.subtract(inputVector, self.getMeanVector(num))) \
                        + (1/2) * math.log(det(self.getCoVarianceMatrix(num))) \
                        - math.log(dataCount[num] / 178)
        # for key, value in decisionValue.items():
        #     print(str(key) + ' ' + str(value))
        return sorted(decisionValue.items(), key=operator.itemgetter(1))[0][0]

    def test(self, testString):
        # print(self.pdData.shape[0]) # 1:59 2:71 3:48 total:178
        # print(type(self.getCoVarianceMatrix(2))) # <class 'pandas.core.frame.DataFrame'>
        # print(type(self.getMeanVector(1))) # <class 'pandas.core.series.Series'>
        print(self.getDecision(testString))


if __name__ == '__main__':
    import sys
    obj = BayesWine('./TrainingData.txt')
    obj.test(sys.argv[1])
