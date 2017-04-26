# @Author: Shane Yu  @Date Created: April 25, 2017
"""
system arguments:
1) Training data path
2) Testing data path
"""
from BayesWine import BayesWine
import sys

bayesObj = BayesWine(sys.argv[1])
correctnessDict = {'Class1':0, 'Class2':0, 'Class3':0}
testingDataCountDict = {'1':0, '2':0, '3':0}
totalDataCountDict = {1:59, 2:71, 3:48} # 每個class資料筆數


with open(sys.argv[2], 'r') as rf:
    for line in rf:
        classNum = line.split(',')[0]
        for key in testingDataCountDict.keys():
            if classNum == key:
                testingDataCountDict[key] += 1
        if bayesObj.getDecision(line) == int(classNum):
            correctnessDict['Class' + classNum] += 1

for num in range(1, 4):
    correctnessDict['Class' + str(num)] = correctnessDict['Class' + str(num)] / float(testingDataCountDict[str(num)])

for key, value in correctnessDict.items():
    print(key + ' 的正確率為' + str(value))

for key, value in testingDataCountDict.items():
    print('Class' + key + ' 的測試數量：' + str(value) + '  訓練數量: ' + str(totalDataCountDict[int(key)]-value) + '  訓練資料百分比: ' + str(round(100*(totalDataCountDict[int(key)]-value)/totalDataCountDict[int(key)], 2)) + ' %')