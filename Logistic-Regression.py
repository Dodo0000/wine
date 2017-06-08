# Author: Shane Yu  Data Created: June 7, 2017
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

class LogisticRegression(object):
    def __init__(self, dataPath):
        self.dataPath = dataPath
        np.set_printoptions(precision=6, suppress=True, linewidth=100000)

    def dataPreprocessing(self):
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
        # print(nestedData)
        return nestedData

    # sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # 計算平均梯度
    def gradient(self, dataset, w): # 偏微分方向
        g = np.zeros(len(w))
        for x, y in dataset: # x是訓練資料 y是該筆資料屬於的類別
            x = np.array(x)
            error = self.sigmoid(w.T.dot(x))
            g += (error - y) * x
            # print(g)
        return g / len(dataset)

    # 計算現在的權重的錯誤有多少
    def cost(self, dataset, w): # Loss funcion
        total_cost = 0
        for x, y in dataset:
            x = np.array(x) # 訓練資料x
            error = self.sigmoid(w.T.dot(x))
            total_cost += abs(y - error)
        return total_cost

    # 演算法實作
    def logistic(self, dataset):
        # w = np.zeros(3) #初始設定w為(0, 0,..., 0)
        w = np.random.rand(14) #初始設定w為(0, 0,..., 0)
        iterations = 100 # 更新100次後停下
        learning_rate = 0.01 # 更新幅度

        costs = [] # 紀錄每次更新權重後新的cost是多少

        for i in range(iterations):
            current_cost = self.cost(dataset, w) # 當前訓練iteration的cost
            print('current_cost=' + str(current_cost))
            costs.append(current_cost)
            w = w - learning_rate * self.gradient(dataset, w)  # w更新的方式: w - learning_rate * 梯度 (例如第一輪，試算出w在w=[0 0 0]時對L的偏微結果)
            # print(w)
            learning_rate *= 0.95 # Learning Rate，逐步遞減

        # 畫出cost的變化曲線，他應該要是不斷遞減 才是正確
        plt.plot(range(iterations), costs)
        # plt.show()
        plt.savefig('Plot.png', format='png')
        return w

    def getWeights(self):
        data = self.dataPreprocessing()
        print(data)
        w = self.logistic(data)
        print(w)


class TempData(object):
    def __init__(self, dataPath):
        self.dataPath = dataPath
        np.set_printoptions(precision=6, suppress=True, linewidth=100000)

    def dataPreprocessing(self):
        data = pd.read_csv(self.dataPath, header=None).values
        nestedData = list()
        for line in data:
            tempList = list()
            tempList.append(line[:3])
            tempList.append(line[3])
            nestedData.append(tempList)
        nestedData = np.array(nestedData)
        # print(nestedData)
        return nestedData

    # sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # 計算平均梯度
    def gradient(self, dataset, w): # 偏微分方向
        g = np.zeros(len(w))
        for x, y in dataset: # x是訓練資料 y是該筆資料屬於的類別
            x = np.array(x)
            error = self.sigmoid(w.T.dot(x))
            g += (error - y) * x
        return g / len(dataset)

    # 計算現在的權重的錯誤有多少
    def cost(self, dataset, w): # Loss funcion
        total_cost = 0
        for x, y in dataset:
            x = np.array(x) # 訓練資料x
            error = self.sigmoid(w.T.dot(x))
            total_cost += abs(y - error)
        return total_cost

    # 演算法實作
    def logistic(self, dataset):
        # w = np.zeros(3) #初始設定w為(0, 0,..., 0)
        w = np.random.rand(3) #初始設定w為(0, 0,..., 0)
        iterations = 100 # 更新100次後停下
        learning_rate = 0.5 # 更新幅度

        costs = [] # 紀錄每次更新權重後新的cost是多少

        for i in range(iterations):
            current_cost = self.cost(dataset, w) # 當前訓練iteration的cost
            print('current_cost=' + str(current_cost))
            costs.append(current_cost)
            w = w - learning_rate * self.gradient(dataset, w)  # w更新的方式: w - learning_rate * 梯度 (例如第一輪，試算出w在w=[0 0 0]時對L的偏微結果)
            # print(w)
            learning_rate *= 0.95 # Learning Rate，逐步遞減

        # 畫出cost的變化曲線，他應該要是不斷遞減 才是正確
        plt.plot(range(iterations), costs)
        # plt.show()
        plt.savefig('tempDataPlot.png', format='png')
        return w

    def getWeights(self):
        data = self.dataPreprocessing()
        # print(data)
        w = self.logistic(data)
        print(w)


if __name__ == '__main__':
    import sys
    if sys.argv[1] == '1':
        obj = LogisticRegression('data.csv')
    else:
        obj = TempData('testData.csv')
    obj.getWeights()
    