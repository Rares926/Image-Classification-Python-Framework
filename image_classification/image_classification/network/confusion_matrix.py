import tensorflow        as tf
import matplotlib.pyplot as plt
import numpy             as np
import io
import sklearn.metrics

#Internal framework imports
from ..utils.data_processing import DataProcessing
from .data_generator         import DataGenerator

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
        
    def __init__(self,model, data_generator, workspace, labels_location):
        self.model=model
        self.data_generator = data_generator
        self.workspace=workspace
        self.labels_location = labels_location

    def on_epoch_end(self, epoch, logs=None):
        self.log_confusion_matrix(epoch)

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
        label_names = DataProcessing.load_label_names(self.labels_location)
        # Log the confusion matrix as an image summary.
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(cmap=plt.cm.Blues)
        figure = plt.gcf() #labels parametrized
        cm_image =self.plot_to_image(figure)

        file_writer_cm = tf.summary.create_file_writer(self.workspace + '/cm')
        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)


