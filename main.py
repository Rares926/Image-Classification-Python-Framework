import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import io
import itertools
import sklearn.metrics
import os

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

if __name__ == "__main__":
    createFolders(os.getcwd())