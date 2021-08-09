import tensorflow as tf
import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers

def createAndRunModel(label,xTrain,yTrain,xTest,yTest,imageSize):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), padding="same",input_shape=(224,224,3)),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64, (5, 5), padding="same"),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(label), activation='softmax')
    ])


    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    opt = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])

    model.fit(xTrain, yTrain, epochs=10)

    test_loss, test_acc = model.evaluate(xTest,  yTest, verbose=1)
    print('\nTest accuracy:', test_acc)

