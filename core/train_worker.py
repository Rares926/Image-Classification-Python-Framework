import tensorflow as tf
import datetime
import os

# Internal framework imports
from utils.io_helper import IOHelper
# Typing imports imports


class TrainWorker:
    def __init__(self,model):
        self.model=model

    def train(self, workspace, x_train, y_train, x_test, y_test, epochs = 10):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.model.compile(optimizer=optimizer,
                    loss=loss_fn,
                    metrics=['accuracy'])
                    
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

        self.model.fit(x_train, y_train, epochs=epochs, callbacks=[tensorboard_callback, cp_callback])

        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=1)
        
        print('\nTest loss:', test_loss)
        print('\nTest accuracy:', test_acc)