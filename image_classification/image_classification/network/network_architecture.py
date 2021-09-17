import tensorflow     as tf
import tensorflow_hub as hub
from importlib.machinery   import SourceFileLoader
import importlib.util

# Internal framework imports
from ..data_structures.image_shape import ImageShape

# Typing imports imports

class ModelArchitecture:

    DEFAULT_INNER_MODEL = [
            tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'), #self.input.shape
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2)
                ]

    def __init__(self, input_shape: ImageShape): 
        self.input_shape = (input_shape.width, input_shape.height, input_shape.channels)
        self.inner_model=None
        self.augments=None 
        self.model = tf.keras.models.Sequential()


    def set_model(self,labels_size:int,model_path:str=None):

        if model_path!=None:
            
            spec=importlib.util.spec_from_file_location("model",model_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            
            self.DEFAULT_INNER_MODEL=mod.Model.get_model()

        self.inner_model = self.DEFAULT_INNER_MODEL

        self.model.add(tf.keras.layers.InputLayer(input_shape = self.input_shape))

        for inner_layer in self.inner_model:
            self.model.add(inner_layer)

        self.model.add(tf.keras.layers.Dense(labels_size, activation='softmax'))
    
        return self.model  


