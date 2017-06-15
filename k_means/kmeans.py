# @Author: Shane Yu  @Data Created: June 15, 2017
import numpy as np
import pandas as pd
from pprint import pprint
import random


class kmeans(object):
    """
    kmeans類別實作kmeans演算法
        - init: 資料dataFrame(預設讀取data.csv)
    get_result函式回傳分群結果
        - input: 要分幾群
        - output: 一個tuple，(分群結果dict, 中心點list, iterations次數)
    """
    def __init__(self, data=pd.read_csv('./data.csv', header=None)):
        self.dataDF = data
    
    def clustering(self, X, mu): # 傳入資料與k個mu的list，此函式作分群
        clusters_dict  = {} # {mu1: [資料1, 資料2, ...,資料N], mu2: [...]}
        for x in X: # iterates over 所有資料，每筆資料都和每個mu算一下誰的距離小，best_mu_key已是最近的centroid
            x = np.array(x)
            # best_mu_key即是最近的那個centroid的index，，所以前面list是[(centroid 0的index, x和centroid 0的歐式距離), ..., (centroid k的index, x和centroid k的歐式距離)]
            best_mu_key = min([(mu_index, np.linalg.norm(x-mu_data)) for mu_index, mu_data in enumerate(mu)], key=lambda y:y[1])[0]
            try:
                clusters_dict[best_mu_key].append(x)
            except: # 如果是第一次放，clusters_dict字典裡會沒有那個mu
                clusters_dict[best_mu_key] = [x]
        return clusters_dict
 
    def reevaluate_centroids(self, mu, clusters_dict):
        newmu = list()
        keys = sorted(clusters_dict.keys())
        for k in keys: # iterates over cluster的所有key
            newmu.append(np.mean(clusters_dict[k], axis = 0)) # 一次算一個cluster的平均(mean vector)            
        return newmu
     
    def oldmu_equals_newmu(self, mu, oldmu): # 查看新的mu和舊的mu是否相等
        return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
     
    def get_result(self, K): # 是資料 K是群數
        # load資料
        data_pool = set()
        for single_data in self.dataDF.values:
            data_pool.add(tuple(single_data[1:]))

        # 初始化oldmu和mu為list，裡面是中心點的資料
        oldmu = random.sample(data_pool, K) # 初始化K個隨機點
        oldmu = [np.array(x) for x in oldmu] # 每個element都轉成numpy array
        mu = random.sample(data_pool, K) # mu是質量中心
        mu = [np.array(x) for x in mu] # 每個element都轉成numpy array

        # converge:會合，聚集
        iterations = 0
        while not self.oldmu_equals_newmu(mu, oldmu):
            oldmu = mu
            # 把data_pool裡的所有data都放進適合的cluster
            clusters_dict = self.clustering(data_pool, mu)
            # 重新評估每個cluster內的centroid
            mu = self.reevaluate_centroids(oldmu, clusters_dict)

            iterations += 1
        return (clusters_dict, mu, iterations)

        
if __name__ == '__main__':
    obj = kmeans()
    result = obj.get_result(3)
    print(result[0])
    print('==================')
    print(result[1])
    print('==================')
    print(result[2])



 
