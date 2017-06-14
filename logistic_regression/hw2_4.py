# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:44:23 2017

@author: NJ
"""
from sklearn import cross_validation
import pandas as pd
import numpy as np
from numpy.linalg import inv, det
import math


class IRHW2(object):
      
    def loadData(self):
        data = pd.read_csv('wine.txt',header=None)
        self.x_train,self.x_test = cross_validation.train_test_split(data,test_size=0.7)

        self.trainCov = {1:self.x_train.loc[self.x_train[0] == 1,1:].cov(),
                    2:self.x_train.loc[self.x_train[0] == 2,1:].cov(),
                    3:self.x_train.loc[self.x_train[0] == 3,1:].cov()}

        self.trainMean = {1:self.x_train.loc[self.x_train[0] == 1,1:].mean(),
                    2:self.x_train.loc[self.x_train[0] == 2,1:].mean(),
                    3:self.x_train.loc[self.x_train[0] == 3,1:].mean()}
        
        self.trainP = {1:len(self.x_train.loc[self.x_train[0] == 1])/len(self.x_train),
                    2:len(self.x_train.loc[self.x_train[0] == 2])/len(self.x_train),
                    3:len(self.x_train.loc[self.x_train[0] == 3])/len(self.x_train)}

    def bayesClassifier(self):
        trueCount = 0
        for datanum in range(0,len(self.x_test.index)):
            resultlist = dict()
            for classnum in range(1,4):
                resultlist[classnum] = (1/2) * np.dot(np.dot(np.subtract(self.x_test.iloc[datanum][1:], self.trainMean[classnum]).T, inv(self.trainCov[classnum])), np.subtract(self.x_test.iloc[datanum][1:], self.trainMean[classnum])) \
                        + (1/2) * math.log(det(self.trainCov[classnum])) \
                        - math.log(self.trainP[classnum])

            if self.x_test.iloc[datanum][0]  ==  min(resultlist.items(), key=lambda x: x[1])[0]:
                trueCount +=1
                         
        print("Accuracy",trueCount/len(self.x_test))                 
    def main(self):
        self.loadData()
        for datanum in range(0,10):
            self.loadData()
            self.bayesClassifier()

if __name__ == '__main__':
    obj = IRHW2()
    obj.main()
