import matplotlib.pyplot as plt
import seaborn as sns

# Internal framework imports

# Typing imports imports


class DataVisualization:
    def __init__(self):
        pass

    @staticmethod
    def visualizeImage(dataset, labels):
        plt.figure(figsize = (5,5))
        plt.imshow(dataset[0][0])
        plt.title(labels[str(dataset[0][1])]['name'])
        plt.show()
        plt.close()

    @staticmethod
    def checkDatasetBalance(dataset, labels):
        l=[]
        for i in dataset:
            l.append(labels[str(i[1])]['name'])

        sns.set_style('darkgrid')
        sns.countplot(l)
        plt.show()
        plt.close()