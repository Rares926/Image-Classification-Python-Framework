import os
import sys
from utils.io_helper import IOHelper
import tensorflow as tf
from jsonargparse import ArgumentParser
from jsonargparse.util import usage_and_exit_error_handler

# Internal framework imports
from core.data_visualization import DataVisualization
from core.data_processing import DataProcessing
from core.train_worker import TrainWorker
from core.network_architecture import ModelArchitecture
# Typing imports imports


class ClassifierTrainer():
    NETWORK_SIZE = 224

    def __init__(self, len, wid, ch=3):
        self.length = int(len)
        self.width = int(wid)
        self.channels = ch
    
    def do_train(self, dataset_root_dir: str, training_workspace_dir: str):
        IOHelper.create_directory(training_workspace_dir)
        labels = DataProcessing.build_labels(dataset_root_dir, training_workspace_dir)
        DataProcessing.createFolders(training_workspace_dir)
        DataProcessing.splitData(dataset_root_dir, training_workspace_dir, 4/5, labels)

        train_location = training_workspace_dir + '/inputData/train'
        test_location = training_workspace_dir + '/inputData/test'

        #TODO: loadData needs image length and width instead of ClassifierTrainer.NETWORK_SIZE
        train = DataProcessing.loadData(train_location, ClassifierTrainer.NETWORK_SIZE, labels)
        test = DataProcessing.loadData(test_location, ClassifierTrainer.NETWORK_SIZE, labels)

        DataVisualization.visualizeImage(train, labels)
        DataVisualization.checkDatasetBalance(train, labels) 

        x_train, y_train, x_test, y_test = DataProcessing.proccesAndNormalize(train, test)

        print("Starting training worker...")
        model_architurecture=ModelArchitecture(self.length,self.width,self.channels)
        model=model_architurecture.set_model(len(labels))

        train_worker = TrainWorker(model)

        train_worker.train(training_workspace_dir, x_train, y_train, x_test, y_test)


def run():
    try:
        parser = ArgumentParser(prog="classifiertrainer",
        error_handler = usage_and_exit_error_handler,
        description="Train a custom classifier using Tensorflow framework")
        parser.add_argument("--dataset_root_dir", "-d", required=True, help="The path of the dataset root dir")
        parser.add_argument("--training_workspace_dir", "-t", required=True, help="The path of the training workspace root dir")
        parser.add_argument("--image_length", "-l", required=True, help="Image length for the model")
        parser.add_argument("--image_width", "-w", required=True, help="Image width for the model")
        parser.add_argument("--image_channels", "-c", required=False, help="Number of image channels for the model (default is 3)")
        args = parser.parse_args()

        trainer = ClassifierTrainer(args.image_length, args.image_width)
        trainer.do_train(args.dataset_root_dir, args.training_workspace_dir)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)

if __name__ == "__main__":
    run()