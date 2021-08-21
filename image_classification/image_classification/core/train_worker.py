import tensorflow as tf
import datetime
import os
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import itertools
import sklearn.metrics
import io

# Internal framework imports
from ..utils.io_helper import IOHelper
# Typing imports imports


class TrainWorker:
    
    def __init__(self,model,x_train, y_train, x_test, y_test):
        self.model=model
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test


    def train(self, workspace:str, x_train:np.ndarray, y_train:np.array, x_test:np.ndarray, y_test:np.array, epochs:int = 10,from_checkpoint:str=None):
        if self.model is None:
            raise Exception("The model must be created in order to be used!")


        def plot_confusion_matrix(cm, class_names):
            """
            Returns a matplotlib figure containing the plotted confusion matrix.

            Args:
                cm (array, shape = [n, n]): a confusion matrix of integer classes
                class_names (array, shape = [n]): String names of the integer classes
            """
            figure = plt.figure(figsize=(8, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion matrix")
            plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)

            # Compute the labels from the normalized confusion matrix.
            labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

            # Use white text if squares are dark; otherwise black.
            threshold = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            return figure


        def plot_to_image(figure):
            """Converts the matplotlib plot specified by 'figure' to a PNG image and
            returns it. The supplied figure is closed and inaccessible after this call."""
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            plt.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image

        def log_confusion_matrix(epoch, logs):
            # Use the model to predict the values from the validation dataset.
            test_pred_raw = self.model.predict(x_test)
            test_pred = np.argmax(test_pred_raw, axis=1)

            # Calculate the confusion matrix.
            cm = sklearn.metrics.confusion_matrix(y_test, test_pred)
            # Log the confusion matrix as an image summary.
            figure =plot_confusion_matrix(cm, class_names=["cat","dog"])
            cm_image =plot_to_image(figure)

            # Log the confusion matrix as an image summary.
            with file_writer_cm.as_default():
                tf.summary.image("Confusion Matrix", cm_image, step=epoch)



        checkpoint_path = os.path.join(workspace,"checkpoints")
        IOHelper.create_directory(checkpoint_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"cp-{epoch:03G}.h5",
                                                         save_best_only=False,
                                                         save_freq='epoch',
                                                         monitor='val_loss',
                                                         save_weights_only=True,
                                                         verbose=1)

#####################################################################################

        workspace=workspace+"/tensorboard/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=workspace, histogram_freq=1)
        file_writer_cm = tf.summary.create_file_writer(workspace + '/cm')
        cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)


#####################################################################################


        if from_checkpoint!=None:
            self.model.load_weights(from_checkpoint)
            
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
                    metrics=['accuracy'])
                    
        self.model.summary()
        
        self.model.fit(x_train, y_train, epochs=epochs, callbacks=[tensorboard_callback, cm_callback,cp_callback])


        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=1)
        
        print('\nTest loss:', test_loss)
        print('\nTest accuracy:', test_acc)