
from image_classification.core.data_generator import DataGenerator
from image_classification.utils.image_loader import ImageLoader
import tensorflow as tf
import datetime
import os
import numpy as np
import albumentations as A
# Internal framework imports
from ..utils.io_helper import IOHelper
from ..core.confusion_matrix import ConfusionMatrixCallback
from ..core.data_processing import DataProcessing
# Typing imports imports


class TrainWorker:
    
    def __init__(self, model, starting_epoch:int=0):
        self.model=model
        self.starting_epoch=starting_epoch


    def train(self, workspace:str, labels, image_loader:ImageLoader, optimizer=None, epochs:int = 10, from_checkpoint:str=None,train_metrics:list='accuracy',augmentations:list=None): # ,x_train:np.ndarray, y_train:np.array, x_test:np.ndarray, y_test:np.array
        if self.model is None:
            raise Exception("The model must be created in order to be used!")

        train_location = workspace + '/inputData/train'
        test_location = workspace + '/inputData/test'
        labels_location = workspace + '/data.json'

        checkpoint_path = os.path.join(workspace,"checkpoints")
        IOHelper.create_directory(checkpoint_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"cp-{epoch:03G}.h5",
                                                         save_best_only=False,
                                                         save_freq='epoch',
                                                         monitor='val_loss',
                                                         save_weights_only=True,
                                                         verbose=1)


        workspace=workspace+"/tensorboard/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=workspace, histogram_freq=1)
        
        self.model.compile(optimizer=optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
                    metrics=train_metrics)
                    
        if from_checkpoint!=None:
            self.model.load_weights(from_checkpoint)
                    

        transform = A.Compose(augmentations)

        self.model.summary()
        training_generator = DataGenerator(train_location, labels, image_loader, is_train_data=True, transform = transform)
        testing_generator = DataGenerator(test_location, labels, image_loader, is_train_data=True)
        self.model.fit(training_generator, epochs=epochs,initial_epoch=self.starting_epoch, callbacks=[tensorboard_callback,ConfusionMatrixCallback(self.model, testing_generator, workspace, labels_location),cp_callback])

        test_loss, test_acc = self.model.evaluate(testing_generator, verbose=1)
        
        print('\nTest loss:', test_loss)
        print('\nTest accuracy:', test_acc)