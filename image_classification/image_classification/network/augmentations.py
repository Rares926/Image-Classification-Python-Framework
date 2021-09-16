import albumentations as A
from .helpers.augment_helper import AugmentHelper 

class Augment:

    def __init__(self):
        self.aug_list=[]

    def build_augmentation_params(self,aug_data : dict):

        for A in aug_data:
            A["name"]=A["name"].lower()

            if not AugmentHelper.check_name(A["name"]):

                raise "Augmentation does not exist"

            A["params"]=AugmentHelper.get_params(A["params"],A["name"])

            self.aug_list.append(self.get_aug(A))

    def get_aug_list(self):
        return self.aug_list

    def get_aug(self,augument:dict):
        method_name = 'aug_' + augument["name"]
        method = getattr(self, method_name, lambda: "Invalid optimizier")

        return method(augument["params"])

    def aug_horizontalflip(self,params:dict):
        return A.HorizontalFlip(p=params["p"])

    def aug_randomcrop(self,params:dict):
        return A.RandomCrop(width=params["width"], height=params["height"])

    def aug_rotate(self,params:dict):
        return A.Rotate (limit=params["limit"],
                        interpolation=params["interpolation"],
                        border_mode=params["border_mode"],
                        value=params["value"],
                        mask_value=params["mask_value"],
                        p=params["p"])

