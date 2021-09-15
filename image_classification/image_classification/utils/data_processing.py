import os
from ..data_structures.ratio import Ratio
from ..utils.resize_method import ResizeMethod

import numpy as np
import cv2 as cv
import albumentations as A
from numpy.lib.type_check import imag
# Internal framework imports
from ..utils.helpers.io_helper import IOHelper
from ..utils.helpers.json_helper import JsonHelper

from ..data_structures.image_shape import ImageShape
from ..network.image_loader import ImageLoader
from ..data_structures.image_format import ImageFormat

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
    def load_ground_truths(data_file_location: str, image_name: str):
        labels = JsonHelper.read_json(data_file_location, True, "Data.json file missing!")
        for key in labels:
            if labels[key]['uid'] in image_name:
                return int(key)
        raise Exception('Error: Invalid file name!')
        
    @staticmethod
    def load_label_names(data_file_location: str):
        labels = JsonHelper.read_json(data_file_location, True, "Data.json file missing!")
        label_names = []
        for i in labels.keys():
            label_names.append(labels[i]["name"])
        return label_names

    @staticmethod
    def albumentate(image,transform=None):

        transform = A.Compose([
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.0, 0.0, 0.0],std=[1.0, 1.0, 1.0],max_pixel_value=255.0,),
            ])

        transformed = transform(image=image)

        return transformed["image"]


    @staticmethod
    def createFolders(root):
        IOHelper.deletedirectory(os.path.join(root,'inputData'))
        directories = ('inputData\\train', 'inputData\\test')
        for directory in directories:
            output_path = os.path.join(root, directory)
            IOHelper.create_directory(output_path, True)

    @staticmethod
    def createResultsFolders(root, labels_path):
        IOHelper.create_directory(root) #folderul trebuie creat sau trebuie sa existe?
        folder_names = DataProcessing.load_label_names(labels_path)
        for item in folder_names:
            temp_path = os.path.join(root, item)
            IOHelper.create_directory(temp_path)

    @staticmethod
    def splitData(dataset, workspace:str, quotient:float, label: Dict[str,Dict[str,str]]):
        for key in label:
            list = os.listdir(os.path.join(dataset, label[key]['name'])) #creeaza o lista cu toate imaginile dintr un folder cu path ul creat 
            number_of_files = len(list)#1000
            to_be_trained = int(quotient * number_of_files)

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
    def proccesAndNormalize(data: np.ndarray):
        x_data = []
        y_data = []

        for image, label in data:
            x_data.append(image)
            y_data.append(label)
   
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        return x_data, y_data