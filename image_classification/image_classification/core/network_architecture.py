import tensorflow as tf
# Internal framework imports

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

    def __init__(self, len:float=224, wid:float=224, ch:int=3): #TODO: width height channels
        self.input_shape=(len, wid, ch) # length,width,channels
        self.inner_model=None
        self.augments=None 
        self.model = tf.keras.models.Sequential()


    def set_model(self, labels_size:int ,use_augumentation_layer:bool=False): #inner model si aug layers cu none ca params

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

        return self.model  #TODO : move to get_model method


