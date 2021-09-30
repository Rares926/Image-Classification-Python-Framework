from image_classification.data_structures.dataset_type import DatasetType
import tensorflow as tf
import numpy      as np
import os

#Internal framework imports
from ..utils.helpers.io_helper      import IOHelper
from ..data_structures.image_loader import ImageLoader

#Typing imports

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, data_path, labels, image_loader: ImageLoader, batch_size, dataset_type: DatasetType, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.image_loader = image_loader
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.image_names = IOHelper.get_image_files(data_path)
        self.image_names_size = len(self.image_names)
        self.transform=transform
        self.on_epoch_end()

    def __len__(self): #from Sequence
        a = len(self.image_names)//self.batch_size
        if self.dataset_type == DatasetType.TRAIN:
            return a
        else:
            return a+1
        

    def __getitem__(self, index:int): #from Sequence
        batch_images = self.image_names[index * self.batch_size : (index + 1) * self.batch_size]
        batch_images_size = len(batch_images)
        x, y = self.generate_X(batch_images, batch_images_size)

        if self.dataset_type == DatasetType.TEST:
            return x, y, batch_images
        else:
            return x, y

    def generate_X(self, batch_images, batch_images_size):

        x = np.empty((batch_images_size, self.image_loader.image_shape.height, self.image_loader.image_shape.width, self.image_loader.image_shape.channels))
        y = np.empty((batch_images_size))

        for index, image_name in enumerate(batch_images):
            image_path = os.path.join(self.data_path, image_name)
            image = self.image_loader.load_image(image_path)

            if self.dataset_type != DatasetType.TRAIN and self.transform!=None:
                raise Exception("It's not possible to use augmentations on validation/test data")
            
            if self.transform!=None:
                transformed = self.transform(image=image)
                image = transformed["image"]

            x[index,] = image
            
            class_id = int(image_name.split('P')[0].split('class_')[1])
            y[index] = class_id
        
        return x, y

    def on_epoch_end(self): #from Sequence
        np.random.shuffle(self.image_names)