# @Author: Shane Yu  @Date Created: April 25, 2017
"""
system arguments:
1) Training data path
2) Testing data path
"""
from BayesWine import BayesWine
import sys
import random
import subprocess
import operator


totalDataCountDict = {1:59, 2:71, 3:48} # 每個class資料筆數
correctnessDict = {'Class1':0, 'Class2':0, 'Class3':0}
totalDataCountDict = {1:59, 2:71, 3:48} # 每個class資料筆數
# initialize classSetList
classSetDict = dict() # {'1':set(), '2':set(), '3':set()}
for x in range(1, 4):
    classSetDict[str(x)] = set()

with open('Data.txt', 'r') as rf:
    for line in rf:
        classSetDict[line.split(',')[0]].add(line)

with open('TempTrainingData.txt', 'w') as wf_Train:
    for key in classSetDict.keys(): # '1', '2', '3'
        for line in random.sample(classSetDict[key], int(totalDataCountDict[int(key)]*float(sys.argv[1]))):
            wf_Train.write(line)
            classSetDict[key].remove(line)

bayesObj = BayesWine('TempTrainingData.txt')
# 剩下的classSetDict已是測試資料
for classNum in classSetDict.keys(): # '1', '2', '3'
    for line in classSetDict[classNum]:
        if bayesObj.getDecision(line) == int(classNum):
            correctnessDict['Class' + classNum] += 1

for num in range(1, 4):
    correctnessDict['Class' + str(num)] = correctnessDict['Class' + str(num)] / float(len(classSetDict[str(num)]))

for key, value in sorted(correctnessDict.items(), key=operator.itemgetter(0)):
    print(key + ' 的正確率為' + str(value))

for key, value in sorted(classSetDict.items(), key=operator.itemgetter(0)):
    print('Class' + key + ' 的測試數量：' + str(len(value)) + '  訓練數量: ' + str(totalDataCountDict[int(key)]-len(value)) + '  訓練資料百分比: ' + str(round(100*(totalDataCountDict[int(key)]-len(value))/totalDataCountDict[int(key)], 2)) + ' %')

subprocess.call(['rm', 'TempTrainingData.txt'])
