

#Internal framework imports

#Typing imports

class NetworkBuilder:

    def __init__(self):

        self.image_shape   = None
        self.image_format  = None
        self.resize_method = None
        self.ratios        = None
        self.resize_after_crop = None

        self.model_path    = None
        
        self.optimizer     = None
        self.augmentations = None
        self.metrics       = None

        self.epochs        = 10
        self.batch_size    = 32




