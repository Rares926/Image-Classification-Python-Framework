import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Internal framework imports

# Typing imports imports
from typing import Dict

class DataVisualization:
    def __init__(self):
        pass

    @staticmethod
    def visualizeImage(dataset:np.ndarray, labels:Dict[str,Dict[str,str]]):
        plt.figure(figsize = (5,5))
        plt.imshow(dataset[0][0])
        plt.title(labels[str(dataset[0][1])]['name'])
        plt.show()
        plt.close()

    @staticmethod
    def checkDatasetBalance(dataset:np.ndarray, labels:Dict[str,Dict[str,str]]):
        l=[]
        for i in dataset:
            l.append(labels[str(i[1])]['name'])

        sns.set_style('darkgrid')
        sns.countplot(l)
        plt.show()
        plt.close()

    @staticmethod
    def showImage(image:np.array, image_class:int,prob:float):
        plt.figure(figsize = (5,5))
        plt.imshow(image)
        plt.title("Image class {} probability {}".format(image_class,prob))
        plt.show()
        plt.close()