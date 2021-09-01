import os
from image_classification.utils.ratio import Ratio
from image_classification.utils.resize_method import ResizeMethod

import numpy as np
import cv2 as cv

# Internal framework imports
from ..utils.io_helper import IOHelper
from ..utils.json_helper import JsonHelper
from ..utils.image_shape import ImageShape
from ..utils.image_loader import ImageLoader
from ..utils.image_format import ImageFormat

# Typing imports imports
from typing import Dict


class DataProcessing:
    def __init__(self):
        pass
    
    @staticmethod
    def build_labels(dataset_root_dir: str, training_workspace_dir: str):
        labels = {}

        dir_list = IOHelper.get_subdirs(dataset_root_dir)
        count = len(dir_list)
        
        for i in range(count):
            content = {}
            content['name'] = dir_list[i]
            content['uid'] = "class_" + format(i, '03d')
            labels[str(i)] = content
        
        JsonHelper.write_json(os.path.join(training_workspace_dir, 'data.json'), labels)
        return labels

    @staticmethod
    def load_label_count(data_file_location: str):
        labels = JsonHelper.read_json(data_file_location, True, "Data.json file missing!")
        label_count = len(labels.keys())
        return label_count

    @staticmethod
    def createFolders(root):
        IOHelper.deletedirectory(os.path.join(root,'inputData'))
        directories = ('inputData\\train', 'inputData\\test')
        for directory in directories:
            output_path = os.path.join(root, directory)
            IOHelper.create_directory(output_path, True)

    @staticmethod
    def splitData(dataset, workspace:str, quotient:float, label: Dict[str,Dict[str,str]]):
        for key in label:
            list = os.listdir(os.path.join(dataset, label[key]['name'])) #creeaza o lista cu toate imaginile dintr un folder cu path ul creat 
            # number_of_files = len(list)
            # to_be_trained = int(quotient * number_of_files)
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
    def loadData(data_dir: str, image_size: ImageShape, image_format: ImageFormat, resize_method:ResizeMethod, ratios:Ratio, labels: Dict[str,Dict[str,str]]) -> np.array:
        data = []
        image_loader = ImageLoader(image_size, image_format, resize_method, ratios)
        for key in labels: 
            class_number = int(key)  #luam clasa in dataset ca si indicele acesteia din labels
            images = IOHelper.get_image_files(data_dir)

            for image in images: #parcurge pe rand toate pozele din folderul dat 
                try:
                    if labels[key]['uid'] in image:
                        image_path = os.path.join(data_dir, image)
                        image = image_loader.load_image(image_path)
                        data.append([image, class_number])
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
        
        x_train = np.array(x_train)
        x_test = np.array(x_test)

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        return x_train, y_train, x_test, y_test