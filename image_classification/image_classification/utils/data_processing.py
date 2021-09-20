import os
import numpy as np
from numpy.lib.type_check import imag

# Internal framework imports
from ..data_structures.ratio         import Ratio
from ..data_structures.resize_method import ResizeMethod
from ..utils.helpers.io_helper       import IOHelper
from ..utils.helpers.json_helper     import JsonHelper
from ..data_structures.image_shape   import ImageShape
from ..data_structures.image_loader  import ImageLoader
from ..data_structures.image_format  import ImageFormat

# Typing imports imports
from typing import Dict


class DataProcessing:
    def __init__(self):
        pass

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
    def createResultsFolders(root, labels_path):
        IOHelper.create_directory(root) #folderul trebuie creat sau trebuie sa existe?
        folder_names = DataProcessing.load_label_names(labels_path)
        for item in folder_names:
            temp_path = os.path.join(root, item)
            IOHelper.create_directory(temp_path)
