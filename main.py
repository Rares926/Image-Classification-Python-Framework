import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import io
import itertools
import sklearn.metrics
import os

def createFolders(root):
    directories = ('train', 'test')
    for i in range(2):
        path = os.path.join(root, directories[i])
        os.mkdir(path)
        print("Created %s directory" %path)

if __name__ == "__main__":
    createFolders("C:/Users/Radu Baciu/Desktop/asdf")