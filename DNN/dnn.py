# import keras
import pandas as pd
from os.path import join

class dnn(object):
    def __init__(self, data_path=join('../', 'data.csv')):
        self.data = pd.read_csv(data_path, header=None)

    def function(self):
        print(self.data)

if __name__ == '__main__':
    dnn_obj = dnn()
    dnn_obj.function()