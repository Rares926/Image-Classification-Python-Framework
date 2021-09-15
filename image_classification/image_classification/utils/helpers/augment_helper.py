from .io_helper import IOHelper
from .dict_helper import DICTHelper

class AugmentHelper:
    
    POSSIBLE_AUGUMENTATIONS=["horizontalflip","randomcrop","flip","randombrightnesscontrast","rotate"]

    DEFAULT_AUGUMENTATIONS={
        "horizontalflip":{
            "p":0.5
        },

        "randomcrop":{
            "height":None,
            "width":None,
            "p":1
        },

        "flip":{
                "p":0.5
        },

        "randombrightnesscontrast":{
                "brightness_limit":0.2,
                "contrast_limit":0.2,
                "brightness_by_max":True,
                "p":0.5
        },

        "rotate":{
            "limit":90,
            "interpolation":1,
            "border_mode":"4",
            "value":None,
            "mask_value":None,
            "p":0.5

        }

        }

    def __init__(self):
        pass
    
    @staticmethod
    def check_name(name:str):
        return name.lower() in AugmentHelper.POSSIBLE_AUGUMENTATIONS

    def get_params(aug:dict,name:str):

        aug=IOHelper.set_dictionary_keys_to_lower(aug)
        aug_tmp=DICTHelper.combine_dict_params(AugmentHelper.DEFAULT_AUGUMENTATIONS,aug,name)

        return aug_tmp


