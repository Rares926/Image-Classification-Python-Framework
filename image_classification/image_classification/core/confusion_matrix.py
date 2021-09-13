
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import itertools
import io
import sklearn.metrics

#Internal framework imports
from image_classification.core import data_generator

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
        
    def __init__(self,model, data_generator, workspace):
        self.model=model
        self.data_generator = data_generator
        self.workspace=workspace

    def on_epoch_end(self, epoch, logs=None):
        self.log_confusion_matrix(epoch)


    def plot_confusion_matrix(self,cm, class_names):
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

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, labels[i, j] , horizontalalignment="center", color="black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("C:/Users/Radu Baciu/Desktop/workspaceTesting/test.png")
        return figure

    def plot_to_image(self,figure):
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


    def log_confusion_matrix(self,epoch):
        # Use the model to predict the values from the validation dataset.
        ground_truths = np.empty((0))
        test_prediction = np.empty((0))
        #generator length in var
        for index in range(len(self.data_generator)):
            batch_images, batch_ground_truths = self.data_generator[index]
            ground_truths = np.append(ground_truths, batch_ground_truths)
            test_prediction_raw = self.model.predict(batch_images)
            test_prediction = np.append(test_prediction, np.argmax(test_prediction_raw, axis=1))

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(ground_truths, test_prediction)
        # Log the confusion matrix as an image summary.
        figure =self.plot_confusion_matrix(cm, ["cat","dog"]) #labels parametrized
        cm_image =self.plot_to_image(figure)

        file_writer_cm = tf.summary.create_file_writer(self.workspace + '/cm')
        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)


