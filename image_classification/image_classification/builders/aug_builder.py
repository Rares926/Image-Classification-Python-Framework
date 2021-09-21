
#Internal framework imports
from ..network.augmentations     import Augment
from ..utils.helpers.json_helper import JsonHelper

#Typing imports

class AugBuilder:
    def __init__(self):
        self.augmentation=None
        

    def arg_parse(self, path: str):

        raw_data = JsonHelper.read_json(path)

        if not {'augmentations'} <= raw_data.keys():
            raise Exception("No augmentations are made")

        augument_params=Augment()
        augument_params.build_augmentation_params(raw_data['augmentations'])
        self.augmentation=augument_params.get_aug_list()
        