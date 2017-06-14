import numpy as np
import pandas as pd

data = pd.read_csv('testData.csv', header=None)
# print(data)
print(len(data))
# for index, row in data.iterrows():
    # print(row[[0, 1, 2]])