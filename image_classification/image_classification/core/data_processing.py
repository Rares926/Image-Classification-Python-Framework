import os
import numpy as np
import cv2 as cv
import albumentations as A
from numpy.lib.type_check import imag
# Internal framework imports
from ..utils.io_helper import IOHelper
from ..utils.json_helper import JsonHelper

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
    def albumentate(image,transform=None):

        transform = A.Compose([
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            # A.Normalize(mean=[0.0, 0.0, 0.0],std=[1.0, 1.0, 1.0],max_pixel_value=255.0,),
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
                        #conditionam resize ul 
                        #aici se poate adauga partea de albumentation 
                        #https://albumentations.ai/docs/getting_started/image_augmentation/
                        arr_resized = cv.resize(img_arr, (image_size, image_size)) #ii da resize dupa marimile dorite 
                        # img_arr=DataProcessing.albumentate(img_arr)
                        data.append([arr_resized, class_number])
                        # data.append([img_arr, class_number])

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