from typing import Pattern
from numpy import testing
from numpy.core.defchararray import _translate_dispatcher
from numpy.lib.npyio import load
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import io
import itertools
import sklearn.metrics
import os
import shutil
import fnmatch
import cv2 as cv

animalsDataset='C:/animalsDataset'

def createFolders(root):
    print(root)
    directories = ('inputData\\train', 'inputData\\test')
    for i in range(2):
        path = os.path.join(root, directories[i])
        print(path)
        try:
            os.makedirs(path)
        except Exception:
            pass
        print("Created %s directory" %path)



def splitData():

    quotient=4/5  #80/20 split 
    directories=('cats','dogs')

    for i in range(2):
        list = os.listdir(os.path.join(animalsDataset,directories[i]))
        numberOfFiles=len(list)
        # toBeTrained=int(quotient*numberOfFiles)
        # toBeTested=numberOfFiles-toBeTrained
        toBeTrained=100
        numberOfFiles=150

        for photo in range(toBeTrained):
            source=os.path.join(animalsDataset,directories[i],list[photo])
            destination='inputData/train'
            shutil.copy(source,destination)

        for photo in range(toBeTrained,numberOfFiles):
            source=os.path.join(animalsDataset,directories[i],list[photo])
            destination='inputData/test'
            shutil.copy(source,destination)


def loadData(dataDir):

    labels=['cat','dog']
    imageSize=224

    data=[]
    
    for label in labels:
        path=os.path.join(dataDir) #nu are sens acum dar o sa aiba dupa ce schimbam putin implementarile 
        classNumber=labels.index(label)  #luam clasa in dataset ca si indicele acesteia din labels
        for image in os.listdir(path): #parcurge pe rand toate pozele din folderul dat 
            try:
                if label in image:
                    imgArr=cv.imread(os.path.join(path,image))[...,::-1] #converteste imagina din BGR in RGB 
                    arrResized=cv.resize(imgArr,(imageSize,imageSize)) #ii da resize dupa marimile dorite 
                    data.append([arrResized,classNumber])
            except Exception as e:
                print(e)

    return np.array(data)
        

if __name__ == "__main__":
    createFolders(os.getcwd())
    splitData()
    train=loadData('inputData/train')
    test=loadData('inputData/test')


