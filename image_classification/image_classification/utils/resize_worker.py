from enum import Enum

#Internal framework imports

#Typing imports

class ResizeWorker:

    class ResizeMethod(Enum):
        UNDEFINED = 1
        CROP = 2
        STRETCH = 3
        LETTERBOX = 4

        @classmethod
        def str2enum(cls, resize_method_string, error_if_undefined = False):
            resize_method_string = resize_method_string.lower()

            if resize_method_string == "crop":
                return cls.CROP
            elif resize_method_string == "stretch":
                return cls.STRETCH
            elif resize_method_string == "letterbox":
                return cls.LETTERBOX
            elif error_if_undefined:
                raise Exception("Error: Undefined resize method!")
            return cls.UNDEFINED

    def __init__(self, resize_method_data: str):
        self.resize_method = self.ResizeMethod.str2enum(resize_method_data)
        a=0
    
    def __str__(self) -> str:
        print ("The resize method is {}".format(self.resize_method))

        