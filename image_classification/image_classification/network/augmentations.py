import albumentations as A


#Internal framework inputs
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
        #de bagat o exceptie in caz ca nu gaseste functia 
        return method(augument["params"])

    def aug_horizontalflip(self,params:dict):
        return A.HorizontalFlip(p=params["p"])

    def aug_randomcrop(self,params:dict):
        return A.RandomCrop(width=params["width"], height=params["height"])

    def aug_centercrop(self,params:dict):
        return A.CenterCrop(width=params["width"], height=params["height"],p=params["p"])

    def aug_rotate(self,params:dict):
        return A.Rotate (limit=params["limit"],
                        interpolation=params["interpolation"],
                        border_mode=params["border_mode"],
                        value=params["value"],
                        mask_value=params["mask_value"],
                        p=params["p"])

    def aug_randombrightnesscontrast(self,params:dict):
        return A.RandomBrightnessContrast(brightness_limit=params["brightness_limit"],
                                          contrast_limit=params["contrast_limit"],
                                          brightness_by_max=params["brightness_by_max"],
                                          p=params["p"]
                                           )

    def aug_flip(self,params:dict):
        return A.Flip(p=params["p"])

