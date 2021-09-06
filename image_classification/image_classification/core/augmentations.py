import albumentations as A
from ..utils.augment_helper import AugmentHelper 
from ..utils.io_helper import IOHelper

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
        #returneaza augmentarile create in forma de lista
        #lista asta se poate transmite ca parametru in compose 

    def get_aug(self,augument):
        method_name = 'aug_' + augument["name"]
        method = getattr(self, method_name, lambda: "Invalid optimizier")

        return method(augument["params"])

    def aug_horizontalflip(self,params):
        return A.HorizontalFlip(p=params["p"])

    def aug_randomcrop(self,params):
        return A.RandomCrop(width=params["width"], height=params["height"])

