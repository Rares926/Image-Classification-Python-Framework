from typing import Pattern
from numpy import testing
from numpy.core.defchararray import _translate_dispatcher
from numpy.lib.npyio import load
import tensorflow as tf
import numpy as np
import os

from Classes import DataVisualization as dv
from Classes import DataProcessing as dp


animalsDataset='C:/animalsDataset'
labels=['cat','dog']
imageSize=224

if __name__ == "__main__":
    dp.createFolders(os.getcwd())
    dp.splitData(animalsDataset,labels)

    train=dp.loadData('inputData/train')
    test=dp.loadData('inputData/test')

    dv.visualizeImage(train,labels)
    dv.checkDatasetBalance(train,labels) 

    xTrain,yTrain,xTest,yTest=dp.proccesAndNormalize(train,test,imageSize)
    print(yTrain)