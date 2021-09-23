import os
import sys
from image_classification.utils.helpers.workspace_helper import WorkspaceHelper
from jsonargparse                  import ArgumentParser
from jsonargparse.util             import usage_and_exit_error_handler

# Internal framework imports
from .data_structures.image_loader import ImageLoader
from .builders.network_builder     import NetworkBuilder
from .utils.data_processing        import DataProcessing
from .core.train_worker            import TrainWorker
from .network.network_architecture import ModelArchitecture
from .utils.helpers.io_helper      import IOHelper
from .builders.train_builder       import TrainBuilder
from .data_structures.resize_method import ResizeMethod


# Typing imports imports


class ClassifierTrainer():
    
    def __init__(self, network:NetworkBuilder, checkpoint):

        self.checkpoint=checkpoint
        self.network=network
    
    def do_train(self, dataset_root_dir: str, training_workspace_dir: str, dont_generate_dataset):
        
        
        IOHelper.create_directory(training_workspace_dir)
        image_loader = ImageLoader(self.network.image_shape, self.network.image_format, self.network.resize_method, self.network.ratios, self.network.resize_after_crop, normalize=False)
        workspace_creator = WorkspaceHelper(dataset_root_dir, training_workspace_dir, image_loader)
        labels = workspace_creator.build_labels()
        if dont_generate_dataset == False:
            workspace_creator.createFolders()
            workspace_creator.splitData(labels, self.network.split_percentage)
        else:
            temp_path = os.path.join(training_workspace_dir, 'inputData')
            IOHelper.check_if_dir_exists(temp_path, "inputData folder does not exist")
            temp_train_path = os.path.join(temp_path, 'train')
            temp_test_path = os.path.join(temp_path, 'test')
            IOHelper.check_if_dir_exists(temp_train_path, "inputData/train folder does not exist")
            IOHelper.check_if_dir_exists(temp_test_path, "inputData/test folder does not exist")

        print("Starting training worker...")
        model_architurecture = ModelArchitecture(self.network.image_shape)
        model = model_architurecture.set_model(len(labels),model_path=self.network.model_path)

        if self.checkpoint:
            starting_epoch=IOHelper.get_epoch_from_checkpoint_path(self.checkpoint)
        else: starting_epoch=0


        train_worker = TrainWorker(model,self.network,starting_epoch)

        image_loader.normalize = True
        image_loader.resize_method = ResizeMethod.NONE
        train_worker.train(training_workspace_dir, labels ,image_loader, from_checkpoint=self.checkpoint) 


def run():
    try:
        parser = ArgumentParser(prog="classifiertrainer",
        error_handler = usage_and_exit_error_handler,
        description="Train a custom classifier using Tensorflow framework")
        parser.add_argument("--training_configuration_file", "-config", required=True, help="The path of the training configuration file (must be JSON format)")
        parser.add_argument("--checkpoint_path", "-checkpoint", required = False, help="The path of the checkpoint file")
        parser.add_argument('--dont_generate_dataset', action='store_true', default=False, help='Skip workspace dataset generation')
        program_args = parser.parse_args()

        trainer_args = TrainBuilder()
        trainer_args.arg_parse(program_args.training_configuration_file)
        trainer = ClassifierTrainer(trainer_args.network,program_args.checkpoint_path)
        trainer.do_train(trainer_args.dataset_path, trainer_args.workspace_path, program_args.dont_generate_dataset)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)

if __name__ == "__main__":
    run()