import os
import sys
from image_classification.core.data_processing import DataProcessing
from image_classification.utils.image_shape import ImageShape
from image_classification.utils.test_builder import TestBuilder
from jsonargparse import ArgumentParser
from jsonargparse.util import usage_and_exit_error_handler

# Internal framework imports
from .core.test_worker import TestWorker
from .core.network_architecture import ModelArchitecture
from .utils.test_builder import TestBuilder
# Typing imports imports


class ModelTester():

    def __init__(self, image_shape: ImageShape, labels_path):
        self.image_shape = image_shape
        self.labels_path = labels_path
    
    def do_test(self, checkpoint_root_dir: str, image_root_dir: str):

        # model_architurecture=ModelArchitecture(self.length,self.width,self.channels)
        model_architurecture=ModelArchitecture(self.image_shape)
        label_count = DataProcessing.load_label_count("C:/Users/Radu Baciu/Desktop/Dev/Image-Classification-Python-Framework/image_classification/image_classification/data.json")
        model=model_architurecture.set_model(label_count)
        #,classifier_model="mobilenet_v2"
        testWorker=TestWorker(model)
        testWorker.load_checkpoint(checkpoint_root_dir)
        testWorker.test_image(image_root_dir, self.image_shape)


def run():
    try:
        parser = ArgumentParser(prog = "classifiertrainer",
        error_handler = usage_and_exit_error_handler,
        description="Test a model saved from a checkpoint")
        parser.add_argument("--test_configuration_file", "-config", required=True, help="The path to the test config file")
        program_args = parser.parse_args()

        tester_args = TestBuilder()
        tester_args.arg_parse(program_args.test_configuration_file)
        tester = ModelTester(tester_args.image_shape, tester_args.labels_path)
        tester.do_test(tester_args.network_path, tester_args.images_path)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)

if __name__ == "__main__":
    run()