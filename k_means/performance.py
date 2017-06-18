from kmeans import kmeans
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from pprint import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class visualization(object):
    def __init__(self):
        self.dataDF = pd.read_csv('./data.csv', header=None)
        self.PCA_result = self.PCA()

    def data_organization_kmeans(self):
        kmeans_obj = kmeans(self.PCA_result)
        # pprint(kmeans_obj.get_result(3)[0]) # 是一個dictionary
        cluster_dict = kmeans_obj.get_result(3)[0]
        x_y_axis_label_dict = dict()
        for key in cluster_dict.keys(): # iterates over 1, 2, 3
            x_y_axis_label_dict[key] = list()        
            x_axis_list = list()
            y_axis_list = list()
            for item in cluster_dict[key]: # iterates over 那個群裡面的每個點 array([ 537.91916502,  -15.38834607])
                x_axis_list.append(item[0])
                y_axis_list.append(item[1])
            x_y_axis_label_dict[key].append(x_axis_list)
            x_y_axis_label_dict[key].append(y_axis_list)
        return x_y_axis_label_dict

    def data_organization_origin(self):
        label_xy_axis_dict = dict()
        print(self.PCA_result)
        for label in range(0, 3):
            label_xy_axis_dict[label] = list()
            x_axis_list = list()
            y_axis_list = list()
            label_xy_axis_dict[label].append(x_axis_list)
            label_xy_axis_dict[label].append(y_axis_list)

        for index, row in self.PCA_result.iterrows():
            label_xy_axis_dict[(row['label']-1)][0].append(row['x'])
            label_xy_axis_dict[(row['label']-1)][1].append(row['y'])
        return label_xy_axis_dict

    def kmeans_scatter_plot(self):
        data_ready_to_plot = self.data_organization_kmeans()
        colorDict = {0: 'red', 1: 'blue', 2: 'green'}
        for key in data_ready_to_plot.keys():
            plt.scatter(data_ready_to_plot[key][0], data_ready_to_plot[key][1], c=colorDict[key], s=20, label=key, alpha=0.8, edgecolors='white')

        plt.title('Kmeans Result')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid(True)
        plt.savefig('kmeans.png', format='png')

    def origin_scatter_plot(self):
        data_ready_to_plot = self.data_organization_origin()
        colorDict = {0: 'red', 1: 'blue', 2: 'green'}
        for key in data_ready_to_plot.keys():
            # print(colorDict[key])
            plt.scatter(data_ready_to_plot[key][0], data_ready_to_plot[key][1], c=colorDict[key], s=20, label=key, alpha=0.8, edgecolors='white')

        plt.title('Orginal Result')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid(True)
        plt.savefig('origin.png', format='png')

    def PCA(self):
        pca = PCA(n_components=2)
        pca.fit(self.dataDF.loc[:, 1:])
        featReduced = pca.fit_transform(self.dataDF.loc[:, 1:])
        two_dimension_data = pd.DataFrame(featReduced)
        result_plus_label = pd.concat([self.dataDF[0], two_dimension_data], axis=1)
        result_plus_label.columns = ['label', 'x', 'y']
        return result_plus_label


if __name__ == '__main__':
    obj = visualization()
    obj.origin_scatter_plot()
    