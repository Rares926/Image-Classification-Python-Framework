import os
import sys
from jsonargparse import ArgumentParser
from jsonargparse.util import usage_and_exit_error_handler

# Internal framework imports
from core.test_worker import TestWorker

# Typing imports imports


class ModelTester():
    NETWORK_SIZE = 224

    def __init__(self):
        pass
    
    def do_test(self, checkpoint_root_dir: str, image_root_dir: str):

        testWorker=TestWorker()
        testWorker.create_model(2)
        testWorker.load_checkpoint(checkpoint_root_dir)
        testWorker.test_image(image_root_dir,ModelTester.NETWORK_SIZE)


def run():
    try:
        parser = ArgumentParser(prog="classifiertrainer",
        error_handler = usage_and_exit_error_handler,
        description="Test a model saved from a checkpoint")
        parser.add_argument("--checkpoint_root_dir", "-d", required=True, help="The path to the checkpoint root")
        parser.add_argument("--images_root_dir", "-p", required=True, help="The path of the image to be tested")
        args = parser.parse_args()

        tester = ModelTester()
        tester.do_test(args.checkpoint_root_dir, args.images_root_dir)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)

if __name__ == "__main__":
    run()