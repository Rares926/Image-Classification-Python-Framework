import tensorflow as tf
import tensorflow_hub as hub
# Internal framework imports
from ..utils.image_shape import ImageShape
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

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2)
                ]

    DEFAULT_AUGMENT_LAYERS = [
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                    ]

    trained_models={
    "mobilenet_v2" : "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
    "inception_v3" : "https://tfhub.dev/google/imagenet/inception_v3/classification/5"
    }

    def __init__(self, input_shape: ImageShape): #TODO: width height channels
        self.input_shape = (input_shape.width, input_shape.height, input_shape.channels)
        self.inner_model=None
        self.augments=None 
        self.model = tf.keras.models.Sequential()



    def set_model(self, labels_size:int ,use_augumentation_layer:bool=False,classifier_model=None): #inner model si aug layers cu none ca params
        if classifier_model==None:
            self.inner_model = self.DEFAULT_INNER_MODEL
            if use_augumentation_layer==True:
                self.augments = self.DEFAULT_AUGMENT_LAYERS

            self.model.add(tf.keras.layers.InputLayer(input_shape = self.input_shape))

            if self.augments is not None:
                for aug_layer in self.augments:
                    self.model.add(aug_layer)

            for inner_layer in self.inner_model:
                self.model.add(inner_layer)

            self.model.add(tf.keras.layers.Dense(labels_size, activation='softmax'))
        else:
         
            feature_extractor_layer = hub.KerasLayer(self.trained_models[classifier_model],input_shape=(224, 224),trainable=False)

            self.model = tf.keras.Sequential([
            feature_extractor_layer,
            tf.keras.layers.Dense(labels_size,activation='softmax')
                ])

    
        
        return self.model  #TODO : move to get_model method


