import tensorflow as tf
import datetime
import os
import numpy as np 

# Internal framework imports
from ..utils.io_helper import IOHelper
from ..core.confusion_matrix import ConfusionMatrixCallback
# Typing imports imports


class TrainWorker:
    
    def __init__(self,model):
        self.model=model


    def train(self, workspace:str, x_train:np.ndarray, y_train:np.array, x_test:np.ndarray, y_test:np.array, epochs:int = 10,from_checkpoint:str=None):
        if self.model is None:
            raise Exception("The model must be created in order to be used!")


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

        if from_checkpoint!=None:
            self.model.load_weights(from_checkpoint)
            
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
                    metrics=['accuracy'])
                    
        self.model.summary()
        
        self.model.fit(x_train, y_train, epochs=epochs, callbacks=[tensorboard_callback,ConfusionMatrixCallback(self.model,x_train,x_test,y_train,y_test,workspace),cp_callback])

        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=1)
        
        print('\nTest loss:', test_loss)
        print('\nTest accuracy:', test_acc)