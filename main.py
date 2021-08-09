# from Utils.JSON_helper import build
from typing import Pattern
from numpy import testing
from numpy.core.defchararray import _translate_dispatcher
from numpy.lib.npyio import load
import tensorflow as tf
import numpy as np
import os

from Classes import data_visualization as dv
from Classes import data_processing as dp
from Utils import json_helper as jh


animalsDataset='C:/animalsDataset'
imageSize=224

if __name__ == "__main__":

    labels=jh.build(animalsDataset)
    print(labels)
    dp.createFolders(os.getcwd())
    dp.splitData(animalsDataset,4/5,labels)

    train=dp.loadData('inputData/train',imageSize,labels)
    test=dp.loadData('inputData/test',imageSize,labels)

    dv.visualizeImage(train,labels)
    dv.checkDatasetBalance(train,labels) 

    xTrain,yTrain,xTest,yTest=dp.proccesAndNormalize(train,test,imageSize)
