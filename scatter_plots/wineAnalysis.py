import matplotlib
matplotlib.use('Agg') # server的backend display參數設定
import matplotlib.pyplot as plt
import pprint
import sys
import numpy as np


class wineAnalysis(object):
    """
    Usage(in command line):
        e.g. 畫以class3和class7為x, y軸的Scatter圖
        - python3 wineAnalysis.py d 3 7 輸出檔名
        
        e.g. 印出class1的mean vector
        - python3 wineAnalysis.py m 1
        
        e.g. 印出class3的covariance matrix
        - python3 wineAnalysis.py c 3

    """
    def __init__(self):
        pass

    def dataStructuring(self, ft1, ft2):
        # initialize the data structure we need for draw
        drawDict = dict()
        for x in range(1, 4):
            drawDict['class'+str(x)] = list()
            drawDict['class'+str(x)].append(list())
            drawDict['class'+str(x)].append(list())

        with open('./data.txt', 'r') as rf:
            for line in rf:
                lineList = line.split(',')
                if lineList[0] == '1':
                    drawDict['class1'][0].append(lineList[ft1])
                    drawDict['class1'][1].append(lineList[ft2])
                elif lineList[0] == '2':
                    drawDict['class2'][0].append(lineList[ft1])
                    drawDict['class2'][1].append(lineList[ft2])
                else:
                    drawDict['class3'][0].append(lineList[ft1])
                    drawDict['class3'][1].append(lineList[ft2])

        return drawDict


    def draw(self, drawDict, ft1, ft2, resultFileName):
        # pprint.pprint(drawDict)
        for key in sorted(drawDict.keys()): # class1, class2, class3
            colorDict = {'class1': 'red', 'class2': 'blue', 'class3': 'green'}
            plt.scatter(drawDict[key][0], drawDict[key][1], c=colorDict[key], s=20, label=key, alpha=0.8, edgecolors='white')
        
        plt.title('Wine Analysis')
        plt.xlabel('feature' + str(ft1))
        plt.ylabel('feature' + str(ft2))
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(resultFileName+'.png', dpi=150, format='png')
                

    def arrayDictGenerator(self):
        arrayDict = dict()
        for x in range(1, 4):
            arrayDict[x] = list()

        with open('./data.txt', 'r') as rf:
            for line in rf:
                line = line.replace('\n', '')
                lineList = line.split(',')
                floatList = [float(x) for x in lineList]
                del floatList[0]
                if lineList[0] == '1':
                    arrayDict[1].append(floatList)
                elif lineList[0] == '2':
                    arrayDict[2].append(floatList)
                else:
                    arrayDict[3].append(floatList)

        return arrayDict


    def meanVector(self, arrayDict, classNum):
        mtx_mean = np.array(arrayDict[classNum])
        print('=== class' + str(classNum) + ' 的mean vector為: ===')
        np.set_printoptions(precision=6, suppress=True, linewidth=250)
        print(np.mean(mtx_mean, axis=0))


    def covarianceMatrix(self, arrayDict, classNum):
        mtx_cov = np.array(arrayDict[classNum]).T # 要轉置之後，row才會是feature向量
        print('=== class' + str(classNum) + ' 的covariance matrix為: ===')
        np.set_printoptions(precision=6, suppress=True, linewidth=250)
        print(np.cov(mtx_cov))


    def main(self):
        # pprint.pprint(dataDict)
        if sys.argv[1] == 'd': # d feature#1  feature#2  輸出檔名(不用副檔名)
            returnDict = self.dataStructuring(int(sys.argv[2]), int(sys.argv[3]))
            self.draw(returnDict, int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        elif sys.argv[1] == 'm': # m class#
            dataDict = self.arrayDictGenerator()
            self.meanVector(dataDict, int(sys.argv[2]))
        elif sys.argv[1] == 'c': # c class#
            dataDict = self.arrayDictGenerator()
            self.covarianceMatrix(dataDict, int(sys.argv[2]))



if __name__ == '__main__':
    obj = wineAnalysis()
    obj.main()
