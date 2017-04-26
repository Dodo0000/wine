from BayesWine import BayesWine

bayesObj = BayesWine()
correctnessDict = {'Class1':0, 'Class2':0, 'Class3':0}
dataCountDict = {'1':0, '2':0, '3':0}

with open('./TestingData_2.txt', 'r') as rf:
    for line in rf:
        classNum = line.split(',')[0]
        for key in dataCountDict.keys():
            if classNum == key:
                dataCountDict[key] += 1
        if bayesObj.getDecision(line) == int(classNum):
            correctnessDict['Class' + classNum] += 1

for num in range(1, 4):
    correctnessDict['Class' + str(num)] = correctnessDict['Class' + str(num)] / float(dataCountDict[str(num)])

for key, value in correctnessDict.items():
    print(key + '的正確率為' + str(value))

for key, value in dataCountDict.items():
    print(key + '的數量：' + str(value))