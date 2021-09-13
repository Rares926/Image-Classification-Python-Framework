from image_classification.utils.image_loader import ImageLoader
import tensorflow as tf
import numpy as np
import os

#Internal framework imports
from ..utils.io_helper import IOHelper
from ..utils.image_loader import ImageLoader

#Typing imports

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, data_path, labels, image_loader: ImageLoader, batch_size = 32, is_train_data = False,transform=None):
        self.data_path = data_path
        self.labels = labels
        self.image_loader = image_loader
        self.batch_size = batch_size
        self.is_train_data = is_train_data
        self.image_names = IOHelper.get_image_files(data_path)
        self.transform=transform
        self.on_epoch_end()

    def __len__(self): #from Sequence
        return len(self.image_names)//self.batch_size

    def __getitem__(self, index:int): #from Sequence
        batch_images = self.image_names[index * self.batch_size : (index + 1) * self.batch_size]

        x, y = self.generate_X(batch_images)

        if self.is_train_data:
            return x, y
        else:
            return x

    def generate_X(self, batch_images):

        x = np.empty((self.batch_size, self.image_loader.image_shape.height, self.image_loader.image_shape.width, self.image_loader.image_shape.channels))
        y = np.empty((self.batch_size))

        for index, image_name in enumerate(batch_images):
            image_path = os.path.join(self.data_path, image_name)
            image = self.image_loader.load_image(image_path)

            if not self.is_train_data and self.transform!=None:
                raise Exception("It's not possible to use augmentatins on test data")

            #aici trebuie sa aplic transform pe image
            
            if self.transform!=None:
                transformed = self.transform(image=image)
                image = transformed["image"]

            x[index,] = image
            
            if self.is_train_data:
                class_id = int(image_name.split('P')[0].split('class_')[1])
                y[index] = class_id
        
        return x, y

    def on_epoch_end(self): #from Sequence
        np.random.shuffle(self.image_names)