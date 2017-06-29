from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import pandas as pd
from os.path import join

class dnn(object):
    def __init__(self, data_path=join('../', 'data.csv')):
        self.data = pd.read_csv(data_path, header=None)

    def function(self):
        # print(self.data)
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
        # model = Sequential()
        # model.add(Dense(32, input_dim=784))

if __name__ == '__main__':
    dnn_obj = dnn()
    dnn_obj.function()