import matplotlib.pyplot as plt
import seaborn as sns

def visualizeImage(train,labels):
    plt.figure(figsize = (5,5))
    plt.imshow(train[1][0])
    plt.title(labels[train[0][1]])


