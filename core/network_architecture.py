import tensorflow as tf
# Internal framework imports

# Typing imports imports


class ModelArchitecture:

    def __init__(self, len:float=224, wid:float=224, ch:int=3):
        self.input_shape=(len, wid, ch) # length,width,channels
        self.inner_model=None
        self.augments=None 
        self.model = tf.keras.models.Sequential()


    def set_model(self, labels_size:int ,augumentation_layer:int=0):

        self.inner_model = [
            tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
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
        if augumentation_layer==0:
            self.augments =[
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                            ]


        self.model.add(tf.keras.layers.InputLayer(input_shape = self.input_shape))
        if self.augments is not None:
            for aug_layer in self.augments:
                self.model.add(aug_layer)

        for inner_layer in self.inner_model:
            self.model.add(inner_layer)

        self.model.add(tf.keras.layers.Dense(labels_size, activation='softmax'))

        return self.model        


