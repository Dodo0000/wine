# @Author: Shane Yu  @Date Created: April 25, 2017
import numpy as np
import pandas as pd
from numpy.linalg import inv, det
import math
import operator


class NaiveBayes(object):
    """
    constructor:
        1) Training data file's path

    getDecision function:
        Input: a line of the testing data (string)
        Output: a class number (int)
    """
    def __init__(self, trainingDataPath):
        np.set_printoptions(precision=6, suppress=True, linewidth=1000)
        pd.set_option('display.width', 1000)
        with open(trainingDataPath, 'r') as rf:
            self.pdData = pd.read_csv(rf, header=None)

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

        return sorted(decisionValue.items(), key=operator.itemgetter(1))[0][0]

    def test(self, testString):
        print(self.getDecision(testString))


class DistanceToMeanVec(object):
    def __init__(self, trainingDataPath):
        np.set_printoptions(precision=6, suppress=True, linewidth=1000)
        pd.set_option('display.width', 1000)
        with open(trainingDataPath, 'r') as rf:
            self.pdData = pd.read_csv(rf, header=None)

    def getMeanVector(self, classNum):
        # return self.pdData.loc[self.pdData[0]==classNum, 1:].mean()
        print(self.pdData.loc[self.pdData[0]==classNum, 1:].mean())





if __name__ == '__main__':
    # import sys
    # obj = NaiveBayes('./Data.txt')
    # obj.test(sys.argv[1])

    obj = DistanceToMeanVec('./Data.txt')
    obj.getMeanVector(1)

