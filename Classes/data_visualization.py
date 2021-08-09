import matplotlib.pyplot as plt
import seaborn as sns


def visualizeImage(dataset,labels):
    plt.figure(figsize = (5,5))
    plt.imshow(dataset[1][0])
    plt.title(labels[dataset[0][1]])
    plt.show()
    plt.close()


def checkDatasetBalance(dataset,labels):
    l=[]
    for i in dataset:
        l.append(labels[i[1]])
    sns.set_style('darkgrid')
    sns.countplot(l)
    plt.show()
    plt.close()




