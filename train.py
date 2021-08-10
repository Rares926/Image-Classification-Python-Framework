import os
import sys
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
        DataProcessing.createFolders(os.getcwd())
        DataProcessing.splitData(dataset_root_dir, 4/5, labels)

        train = DataProcessing.loadData('inputData/train', ClassifierTrainer.NETWORK_SIZE, labels)
        test = DataProcessing.loadData('inputData/test', ClassifierTrainer.NETWORK_SIZE, labels)

        DataVisualization.visualizeImage(train, labels)
        DataVisualization.checkDatasetBalance(train, labels) 

        x_train, y_train, x_test, y_test = DataProcessing.proccesAndNormalize(train, test)

        print("Starting training worker...")
        train_worker = TrainWorker()
        train_worker.create_model(labels)
        train_worker.train(x_train, y_train, x_test, y_test)


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