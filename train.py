import os
import sys
import tensorflow as tf
from jsonargparse import ArgumentParser
from jsonargparse.util import usage_and_exit_error_handler

# Internal framework imports
from core.data_visualization import DataVisualization
from core.data_processing import DataProcessing
from core.train_worker import TrainWorker

# Typing imports imports


class ClassifierTrainer():
    NETWORK_SIZE = 224

    def __init__(self):
        pass
    
    def do_train(self, dataset_root_dir: str, training_workspace_dir: str):
        labels = DataProcessing.build_labels(dataset_root_dir)
        DataProcessing.createFolders(training_workspace_dir)
        DataProcessing.splitData(dataset_root_dir, training_workspace_dir, 4/5, labels)

        train_location = training_workspace_dir + '/inputData/train'
        test_location = training_workspace_dir + '/inputData/test'

        train = DataProcessing.loadData(train_location, ClassifierTrainer.NETWORK_SIZE, labels)
        test = DataProcessing.loadData(test_location, ClassifierTrainer.NETWORK_SIZE, labels)

        DataVisualization.visualizeImage(train, labels)
        DataVisualization.checkDatasetBalance(train, labels) 

        x_train, y_train, x_test, y_test = DataProcessing.proccesAndNormalize(train, test)

        print("Starting training worker...")
        model = tf.keras.models.Sequential([
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
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(labels), activation='softmax')
        ])
        augmentations = tf.keras.models.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])
        train_worker = TrainWorker(model, augmentations)
        train_worker.create_model(labels)
        train_worker.train(training_workspace_dir, x_train, y_train, x_test, y_test)


def run():
    try:
        parser = ArgumentParser(prog="classifiertrainer",
        error_handler = usage_and_exit_error_handler,
        description="Train a custom classifier using Tensorflow framework")
        parser.add_argument("--dataset_root_dir", "-d", required=True, help="The path of the dataset root dir")
        parser.add_argument("--training_workspace_dir", "-t", required=True, help="The path of the training workspace root dir")
        args = parser.parse_args()

        trainer = ClassifierTrainer()
        trainer.do_train(args.dataset_root_dir, args.training_workspace_dir)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)

if __name__ == "__main__":
    run()