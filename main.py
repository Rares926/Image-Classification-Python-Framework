from typing import Pattern
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

animalsDataset='C:/animalsDataset'

def createFolders(root):
    print(root)
    directories = ('splittedData\\train', 'splittedData\\test')
    for i in range(2):
        path = os.path.join(root, directories[i])
        print(path)
        try:
            os.makedirs(path)
        except Exception:
            pass
        print("Created %s directory" %path)


def splitData():

    quotient=4/5
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
            destination='splittedData/train'
            shutil.copy(source,destination)

        for photo in range(toBeTrained,numberOfFiles):
            source=os.path.join(animalsDataset,directories[i],list[photo])
            destination='splittedData/test'
            shutil.copy(source,destination)


if __name__ == "__main__":
    createFolders(os.getcwd())
    splitData()
