import os
import sys
from jsonargparse import ArgumentParser
from jsonargparse.util import usage_and_exit_error_handler

# Internal framework imports
from .utils.data_processing import DataProcessing
from .data_structures.image_shape import ImageShape
from .builders.test_builder import TestBuilder
from .core.test_worker import TestWorker
from .network.network_architecture import ModelArchitecture
from .builders.test_builder import TestBuilder
# Typing imports imports


class ModelTester():

    def __init__(self, image_shape: ImageShape, labels_path: str, results_path: str):
        self.image_shape = image_shape
        self.labels_path = labels_path
        self.results_path = results_path
    
    def do_test(self, checkpoint_root_dir: str, image_root_dir: str):
        DataProcessing.createResultsFolders(self.results_path, self.labels_path)
        model_architurecture=ModelArchitecture(self.image_shape)
        label_count = DataProcessing.load_label_count(self.labels_path)
        model=model_architurecture.set_model(label_count)

        #,classifier_model="mobilenet_v2"
        testWorker=TestWorker(model, self.labels_path)
        testWorker.load_checkpoint(checkpoint_root_dir)
        testWorker.test_image(image_root_dir, self.image_shape, self.results_path)


def run():
    try:
        parser = ArgumentParser(prog = "classifiertrainer",
        error_handler = usage_and_exit_error_handler,
        description="Test a model saved from a checkpoint")
        parser.add_argument("--test_configuration_file", "-config", required=True, help="The path to the test config file")
        parser.add_argument("--checkpoint_path", "-checkpoint", required = True, help="The path of the checkpoint file")
        program_args = parser.parse_args()

        tester_args = TestBuilder()
        tester_args.arg_parse(program_args.test_configuration_file)
        tester = ModelTester(tester_args.image_shape, tester_args.labels_path, tester_args.results_path)
        tester.do_test(program_args.checkpoint_path, tester_args.images_path)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)

if __name__ == "__main__":
    run()