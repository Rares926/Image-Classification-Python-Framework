import tensorflow     as tf
import importlib.util

# Internal framework imports
from ..data_structures.image_shape import ImageShape

# Typing imports imports

class ModelArchitecture:

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
            
            self.inner_model=mod.Model.get_model()
        else:
            raise Exception("No model given!")

        self.model.add(tf.keras.layers.InputLayer(input_shape = self.input_shape))

        for inner_layer in self.inner_model:
            self.model.add(inner_layer)

        self.model.add(tf.keras.layers.Dense(labels_size, activation='softmax'))
    
        return self.model  


