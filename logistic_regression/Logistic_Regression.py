# Author: Shane Yu  Data Created: June 7, 2017
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

class LogisticRegression(object):
    def __init__(self, dataPath):
        self.dataPath = dataPath
        np.set_printoptions(precision=6, suppress=True, linewidth=100000)

    def data_preprocessing(self):
        data = pd.read_csv(self.dataPath, header=None).values
        nestedData = list()
        for line in data:
            if line[0] == 3: continue
            line[0] -= 1
            # print(line)
            tempList = list()
            dataList = list()
            dataList.append(1.0)
            for item in line[1:]:
                dataList.append(item)
            tempList.append(dataList)
            tempList.append(line[0])
            nestedData.append(tempList)
        nestedData = np.array(nestedData)
        print(nestedData)
        return nestedData

    # sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # 計算平均梯度
    def get_gradient(self, dataset, w): # 偏微分方向
        g = np.zeros(len(w))
        for x, y in dataset: # x是訓練資料 y是該筆資料屬於的類別
            x = np.array(x)
            error = self.sigmoid(w.T.dot(x))
            g += (error - y) * x
            # print(g)
        return g / len(dataset)

    # 計算現在的權重的錯誤有多少
    def cost(self, dataset, w):
        total_cost = 0
        for x, y in dataset:
            x = np.array(x) # 訓練資料x
            error = self.sigmoid(w.T.dot(x))
            total_cost += abs(y - error)
        return total_cost

    # 演算法實作
    def logistic_regression(self, dataset):
        w = np.random.rand(14) #初始設定w為隨機的14為向量
        iterations = 100 # 更新100次後停下
        learning_rate = 0.1 # 更新幅度

        costs = [] # 紀錄每次更新權重後新的cost是多少

        for i in range(iterations):
            current_cost = self.cost(dataset, w) # 當前訓練iteration的cost
            costs.append(current_cost)
            w = w - learning_rate * self.get_gradient(dataset, w)  # w更新的方式: w - learning_rate * 梯度 (例如第一輪，試算出w在w=[rand, rand ,..., rand]時對L的偏微結果)
            learning_rate *= 0.95 # Learning Rate，逐步遞減

        # 畫出cost的變化曲線，他應該要是不斷遞減 才是正確
        # plt.plot(range(iterations), costs)
        # plt.show()
        # plt.savefig('Plot.png', format='png')
        return w

    def data_str_to_list(self, dataStr):
        return [float(item) for item in dataStr.split(',')[1:]]

    def get_classification(self, dataStr):
        data = self.data_preprocessing()
        w = self.logistic_regression(data)

        dataVector = np.array(self.data_str_to_list(dataStr))
        probability = self.sigmoid(w[1:].T.dot(dataVector) + w[0])
        if probability >= 0.5:
            return 1
        else:
            return 0


if __name__ == '__main__':
    import sys
    obj = LogisticRegression('data.csv')
    obj.get_classification([12.17,1.45,2.53,19,104,1.89,1.75,.45,1.03,2.95,1.45,2.23,355])
