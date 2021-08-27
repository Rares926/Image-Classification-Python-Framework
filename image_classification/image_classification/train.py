import os
import sys
from PIL.Image import Image
from jsonargparse import ArgumentParser
from jsonargparse.util import usage_and_exit_error_handler
import cv2 as cv

# Internal framework imports
from .core.data_visualization import DataVisualization
from .core.data_processing import DataProcessing
from .core.train_worker import TrainWorker
from .core.network_architecture import ModelArchitecture
from .utils.io_helper import IOHelper
from .utils.train_builder import TrainBuilder
from .utils.image_shape import ImageShape
from .utils.image_format import ImageFormat
from .utils.resize_worker import ResizeWorker
from .utils.ratio import Ratio
# Typing imports imports


class ClassifierTrainer():
    #NETWORK_SIZE = 224

    def __init__(self, image_shape: ImageShape, image_format: ImageFormat, resize_method: ResizeWorker, ratios: Ratio):
        self.image_shape = image_shape
        self.image_format = image_format
        self.resize_method = resize_method
        self.ratios = ratios
    
    def do_train(self, dataset_root_dir: str, training_workspace_dir: str):
        IOHelper.create_directory(training_workspace_dir)
        labels = DataProcessing.build_labels(dataset_root_dir, training_workspace_dir)
        DataProcessing.createFolders(training_workspace_dir)
        DataProcessing.splitData(dataset_root_dir, training_workspace_dir, 4/5, labels)

        train_location = training_workspace_dir + '/inputData/train'
        test_location = training_workspace_dir + '/inputData/test'

        train = DataProcessing.loadData(train_location, self.image_shape, self.image_format, self.resize_method, self.ratios, labels)
        test = DataProcessing.loadData(test_location, self.image_shape, self.image_format, self.resize_method, self.ratios, labels)

        #DataVisualization.visualizeImage(train, labels)
        #DataVisualization.checkDatasetBalance(train, labels) 

        x_train, y_train, x_test, y_test = DataProcessing.proccesAndNormalize(train, test)

        print("Starting training worker...")
        model_architurecture = ModelArchitecture(self.image_shape)
        model = model_architurecture.set_model(len(labels))
        #,classifier_model="mobilenet_v2"
        train_worker = TrainWorker(model)

        train_worker.train(training_workspace_dir, x_train, y_train, x_test, y_test)


def run():
    try:
        parser = ArgumentParser(prog="classifiertrainer",
        error_handler = usage_and_exit_error_handler,
        description="Train a custom classifier using Tensorflow framework")
        parser.add_argument("--training_configuration_file", "-config", required=True, help="The path of the training configuration file (must be JSON format)")
        program_args = parser.parse_args()

        trainer_args = TrainBuilder()
        trainer_args.arg_parse(program_args.training_configuration_file)
        trainer = ClassifierTrainer(trainer_args.image_shape, trainer_args.image_format, trainer_args.resize_method,trainer_args.ratios)
        trainer.do_train(trainer_args.dataset_path, trainer_args.workspace_path)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)

if __name__ == "__main__":
    run()