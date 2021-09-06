from ..utils.io_helper import IOHelper
from ..utils.dict_helper import DICTHelper

class AugmentHelper:
    
    POSSIBLE_AUGUMENTATIONS=["randomcrop","horizontalflip","verticalflip","rotate"]

    DEFAULT_AUGUMENTATIONS={
        "horizontalflip":{
            "p":0.5
        },

        "randomcrop":{
            "height":None,
            "width":None,
            "p":1
        }


    }
    def __init__(self):
        pass
    
    @staticmethod
    def check_name(name:str):
        return name.lower() in AugmentHelper.POSSIBLE_AUGUMENTATIONS

    def get_params(aug:dict,name):

        aug=IOHelper.set_dictionary_keys_to_lower(aug)
        aug_tmp=DICTHelper.combine_dict_params(AugmentHelper.DEFAULT_AUGUMENTATIONS,aug,name)

        return aug_tmp


