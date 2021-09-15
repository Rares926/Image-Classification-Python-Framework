import os
import sys

from PIL.Image import Image
from image_classification.network.image_loader import ImageLoader
from jsonargparse import ArgumentParser
from jsonargparse.util import usage_and_exit_error_handler
import cv2 as cv

# Internal framework imports
from .utils.data_processing import DataProcessing
from .core.train_worker import TrainWorker
from .network.network_architecture import ModelArchitecture
from .utils.helpers.io_helper import IOHelper
from .builders.train_builder import TrainBuilder
from .data_structures.image_shape import ImageShape
from .data_structures.image_format import ImageFormat
from .utils.resize_method import ResizeMethod
from .data_structures.ratio import Ratio
# Typing imports imports


class ClassifierTrainer():
    
    def __init__(self, image_shape: ImageShape, image_format: ImageFormat, resize_method: ResizeMethod, ratios: Ratio, checkpoint ,optimizer,metrics,augmentations):
        self.image_shape = image_shape
        self.image_format = image_format
        self.resize_method = resize_method
        self.ratios = ratios
        self.checkpoint=checkpoint
        self.optimizer=optimizer
        self.metrics=metrics
        self.augmentations=augmentations
    
    def do_train(self, dataset_root_dir: str, training_workspace_dir: str):
        IOHelper.create_directory(training_workspace_dir)
        labels = DataProcessing.build_labels(dataset_root_dir, training_workspace_dir)
        DataProcessing.createFolders(training_workspace_dir)
        DataProcessing.splitData(dataset_root_dir, training_workspace_dir, 0.9, labels)

        train_location = training_workspace_dir + '/inputData/train'
        test_location = training_workspace_dir + '/inputData/test'

        train = DataProcessing.loadData(train_location, self.image_shape, self.image_format, self.resize_method, self.ratios, labels)
        ##test = DataProcessing.loadData(test_location, self.image_shape, self.image_format, self.resize_method, self.ratios, labels)

        #DataVisualization.visualizeImage(train, labels)
        #DataVisualization.checkDatasetBalance(train, labels) 

        ##x_train, y_train, x_test, y_test = DataProcessing.proccesAndNormalize(train, test)

        print("Starting training worker...")
        model_architurecture = ModelArchitecture(self.image_shape)
        model = model_architurecture.set_model(len(labels))
        #,classifier_model="mobilenet_v2"

        if self.checkpoint:
            starting_epoch=IOHelper.get_epoch_from_checkpoint_path(self.checkpoint)
        else: starting_epoch=0

        train_worker = TrainWorker(model,starting_epoch)

        #asta ar putea fi implementate in alta parte
        #din self.checkpoint trebuie sa iau doar epoca
        image_loader = ImageLoader(self.image_shape, self.image_format, self.resize_method, self.ratios)
        train_worker.train(training_workspace_dir, labels ,image_loader, self.optimizer, from_checkpoint=self.checkpoint,train_metrics=self.metrics,augmentations=self.augmentations) #aici mai trebuie adaugat ceva de train metrics


def run():
    try:
        parser = ArgumentParser(prog="classifiertrainer",
        error_handler = usage_and_exit_error_handler,
        description="Train a custom classifier using Tensorflow framework")
        parser.add_argument("--training_configuration_file", "-config", required=True, help="The path of the training configuration file (must be JSON format)")
        parser.add_argument("--checkpoint_path", "-checkpoint", required = False, help="The path of the checkpoint file")
        program_args = parser.parse_args()

        trainer_args = TrainBuilder()
        trainer_args.arg_parse(program_args.training_configuration_file)
        trainer = ClassifierTrainer(trainer_args.image_shape, trainer_args.image_format, trainer_args.resize_method,trainer_args.ratios,program_args.checkpoint_path,trainer_args.optimizer,trainer_args.metrics,trainer_args.augumentations)
        trainer.do_train(trainer_args.dataset_path, trainer_args.workspace_path)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)

if __name__ == "__main__":
    run()