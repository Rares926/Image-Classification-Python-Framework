import os
import numpy as np
import cv2 as cv

# Internal framework imports
from utils.io_helper import IOHelper
from utils.json_helper import JsonHelper

# Typing imports imports
from typing import Dict


class DataProcessing:
    def __init__(self):
        pass
    
    @staticmethod
    def build_labels(path: str):
        labels = {}

        dir_list = IOHelper.get_subdirs(path)
        count = len(dir_list)
        
        for i in range(count):
            content = {}
            content['name'] = dir_list[i]
            content['uid'] = "class_" + format(i, '03d')
            labels[str(i)] = content
        
        JsonHelper.write_json('data.json', labels)
        return labels

    @staticmethod
    def createFolders(root):
        directories = ('inputData\\train', 'inputData\\test')
        for directory in directories:
            output_path = os.path.join(root, directory)
            IOHelper.create_directory(output_path, True)

    @staticmethod
    def splitData(dataset, workspace, quotient, label: Dict[str,Dict[str,str]]):
        for key in label:
            list = os.listdir(os.path.join(dataset, label[key]['name'])) #creeaza o lista cu toate imaginile dintr un folder cu path ul creat 
            #number_of_files = len(list)
            #to_be_trained = int(quotient * number_of_files)
            #to_be_tested = number_of_files - to_be_trained
            to_be_trained = 300
            number_of_files = 500

            for photo in range(to_be_trained):
                source = os.path.join(dataset, label[key]['name'], list[photo])
                destination = workspace + '/inputData/train/' + label[key]['uid'] + 'P' + str(photo) + '.jpg'
                IOHelper.copyfile(source, destination)

            for photo in range(to_be_trained, number_of_files):
                source = os.path.join(dataset, label[key]['name'], list[photo])
                destination = workspace + '/inputData/test/' + label[key]['uid'] + 'P' + str(photo) + '.jpg'
                IOHelper.copyfile(source, destination)

    @staticmethod
    def loadData(data_dir: str, image_size: float, labels: Dict[str,Dict[str,str]]) -> np.array:
        data = []
        
        for key in labels:
            path = os.path.join(data_dir) #nu are sens acum dar o sa aiba dupa ce schimbam putin implementarile 
            class_number = int(key)  #luam clasa in dataset ca si indicele acesteia din labels
            images = IOHelper.get_image_files(path)

            for image in images: #parcurge pe rand toate pozele din folderul dat 
                try:
                    if labels[key]['uid'] in image:
                        img_arr = cv.imread(os.path.join(path, image))[...,::-1] #converteste imagina din BGR in RGB 
                        arr_resized = cv.resize(img_arr, (image_size, image_size)) #ii da resize dupa marimile dorite 
                        data.append([arr_resized, class_number])
                except Exception as e:
                    print(e)

        data = np.array(data)
        np.random.shuffle(data)
        return data

    @staticmethod
    def proccesAndNormalize(train: np.ndarray, test: np.ndarray):
        x_train = []
        y_train = []
        x_test = []
        y_test = []  

        for image, label in train:
            x_train.append(image)
            y_train.append(label)
        
        for image, label in test:
            x_test.append(image)
            y_test.append(label)
        
        x_train = np.array(x_train) / 255.0
        x_test = np.array(x_test) / 255.0

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        return x_train, y_train, x_test, y_test