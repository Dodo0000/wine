# @Author: Shane Yu  @Date Created: April 25, 2017
"""
system arguments:
1) Training data ratio (e.g. 0.5)
"""
from Logistic_Regression import LogisticRegression
import sys
import random
import subprocess
import operator

print('---------------------')
print('Bayes Classifier :')
totalDataCountDict = {1:59, 2:71, 3:48} # 每個class資料筆數
correctnessDict = {'Class1':0, 'Class2':0}
# 初始化classSetList
classSetDict = dict() # {'1':set(), '2':set(), '3':set()}
for x in range(1, 3):
    classSetDict[str(x)] = set()

with open('data.csv', 'r') as rf:
    for line in rf:
        CN = line.split(',')[0]
        if CN == '3': continue
        classSetDict[CN].add(line)

# 採樣訓練資料所佔比例並寫檔
with open('TempTrainingData.txt', 'w') as wf_Train:
    for key in classSetDict.keys(): # '1', '2', '3'
        for line in random.sample(classSetDict[key], int(totalDataCountDict[int(key)]*float(sys.argv[1]))):
            wf_Train.write(line)
            classSetDict[key].remove(line)
            # classSetDict中剩下的已是測試資料

# 用以計算總正確率
totalCorrectNumber = 0
# 利用NaiveBayes進行分類
LR_Obj = LogisticRegression('TempTrainingData.txt')
for classNum in classSetDict.keys(): # '1', '2', '3'
    for line in classSetDict[classNum]:
        if LR_Obj.get_classification(line) == (int(classNum)-1):
            correctnessDict['Class' + classNum] += 1
            totalCorrectNumber += 1

# 總正確率計算
number_of_data_in_TempTraining = 0
for key in classSetDict.keys():
    number_of_data_in_TempTraining += len(classSetDict[key])

totalCorrectnessRatio = float(totalCorrectNumber) / float(number_of_data_in_TempTraining)
print('***總正確率為: ' + str(totalCorrectnessRatio))

# 個別類別正確率計算
for num in range(1, 3):
    correctnessDict['Class' + str(num)] = correctnessDict['Class' + str(num)] / float(len(classSetDict[str(num)]))
# 列印
for key, value in sorted(correctnessDict.items(), key=operator.itemgetter(0)):
    print(key + ' 的正確率為' + str(value))
 
print('---------------------')

subprocess.call(['rm', 'TempTrainingData.txt'])
