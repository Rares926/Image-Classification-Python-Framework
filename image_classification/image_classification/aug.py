import os
import sys
import albumentations as A
import cv2
import numpy as np 
from image_classification.utils.helpers.json_helper import JsonHelper
from jsonargparse                  import ArgumentParser
from jsonargparse.util             import usage_and_exit_error_handler

# Internal framework imports
from .builders.aug_builder         import AugBuilder
from .utils.data_processing        import DataProcessing
from .utils.helpers.io_helper      import IOHelper
# Typing imports imports


class AugTester():
    #https://albumentations.ai/docs/faq/#how-can-i-find-which-augmentations-were-applied-to-the-input-data-and-which-parameters-they-used
    def __init__(self, transform, path: str ,steps: int):

        self.transform =A.Compose(transform)
        self.steps     =int(steps)

        #creez doua liste una cu pozele citite si una cu numele acestora si le dau zip 
        self.img_names=[]
        self.img_path=[]

        #lista cu numele pozelor indiferent daca e una sau mai multe
        if IOHelper.is_image_file(path):
            self.img_names.append(IOHelper.get_filename(path))
            image = cv2.imread(path)
            self.img_path.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            self.img_names = IOHelper.get_image_files(path)
            for idx in self.img_names:
                tmp_path=os.path.join(path,idx)
                image=cv2.imread(tmp_path)
                self.img_path.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))





    def do_augmentation_test(self,output_path=None):

            output_location=output_path if output_path!=None else os.getcwd()
            directory=DataProcessing.createFolder(output_location,"augm_test")

            for idx in range(len(self.img_path)):
                for step in range(self.steps):
                    transformed = self.transform(image=self.img_path[idx])["image"]
                    cv2.imwrite(os.path.join(directory,self.img_names[idx]+"_aug_"+str(step)+".jpg"),transformed)




def run():
    try:
        parser = ArgumentParser(prog="augtester",
                                error_handler = usage_and_exit_error_handler,
                                description="Test the augmentation pipeline")

        parser.add_argument("--augmentation_pipeline_file", "-config", required=True, help="The path of the training configuration file (must be JSON format)")
        parser.add_argument("--image_path", "-img", required = True, help="The path of the image to be augmented")
        parser.add_argument("--nr_of_steps", "-steps", required =True, help="Number of augmentations")
        parser.add_argument("--output_path", "-out", required =False, help="Number of augmentations")

        program_args = parser.parse_args()

        trainer_args = AugBuilder() #aici trebuie sa mai fac niste verificari si sa arunc exceptii 
        trainer_args.arg_parse(program_args.augmentation_pipeline_file)
        
        trainer = AugTester(trainer_args.augmentation,program_args.image_path,program_args.nr_of_steps)
        trainer.do_augmentation_test(program_args.output_path)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)

if __name__ == "__main__":
    run()