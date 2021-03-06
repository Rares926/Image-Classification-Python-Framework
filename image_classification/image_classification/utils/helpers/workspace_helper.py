import os
from typing import Dict
import cv2 as cv

from image_classification.data_structures.image_loader import ImageLoader
from image_classification.utils.helpers.io_helper import IOHelper
from image_classification.utils.helpers.json_helper import JsonHelper


class WorkspaceHelper:
    def __init__(self, dataset_directory: str, workspace_directory: str, image_loader: ImageLoader):
        self.dataset_directory = dataset_directory
        self.workspace_directory = workspace_directory
        self.image_loader = image_loader
    
    def createFolders(self): #2
        IOHelper.deletedirectory(os.path.join(self.workspace_directory,'inputData'))
        directories = ('inputData\\train', 'inputData\\test')
        for item in directories:
            output_path = os.path.join(self.workspace_directory, item)
            IOHelper.create_directory(output_path, True)

    def build_labels(self): #1
        labels = {}

        dir_list = IOHelper.get_subdirs(self.dataset_directory)
        count = len(dir_list)
        
        for i in range(count):
            content = {}
            content['name'] = dir_list[i]
            content['uid'] = "class_" + format(i, '03d')
            labels[str(i)] = content

        JsonHelper.write_json(os.path.join(self.workspace_directory, 'data.json'), labels)
        return labels

    def splitData(self, labels: Dict[str,Dict[str,str]], quotient:float):
        
        log_train = {}
        log_train_path = os.path.join(self.workspace_directory, 'train_log.json')
        log_test = {}
        log_test_path = os.path.join(self.workspace_directory, 'test_log.json')
        if quotient >=1 or quotient <=0:
            raise Exception("Split quotient out of bounds!")
        for key in labels:
            list = os.listdir(os.path.join(self.dataset_directory, labels[key]['name'])) #creeaza o lista cu toate imaginile dintr un folder cu path ul creat 
            number_of_files = len(list)
            to_be_trained = int(quotient * number_of_files)

            for photo in range(to_be_trained):
                source = os.path.join(self.dataset_directory, labels[key]['name'], list[photo])
                new_name = labels[key]['uid'] + 'P' + str(photo) + '.jpg'
                destination = self.workspace_directory + '/inputData/train/' + new_name
                image = self.image_loader.load_image(source)
                log_train[list[photo]] = new_name
                cv.imwrite(destination, image)
                #IOHelper.copyfile(source, destination)

            for photo in range(to_be_trained, number_of_files):
                source = os.path.join(self.dataset_directory, labels[key]['name'], list[photo])
                new_name = labels[key]['uid'] + 'P' + str(photo) + '.jpg'
                destination = self.workspace_directory + '/inputData/test/' + new_name
                image = self.image_loader.load_image(source)
                log_test[list[photo]] = new_name
                cv.imwrite(destination, image)
                #IOHelper.copyfile(source, destination)
            JsonHelper.write_json(log_train_path, log_train)
            JsonHelper.write_json(log_test_path, log_test)
