

#Internal framework imports

#Typing imports

class ImageShape:
    def __init__(self, shape_data: dict):
        self.width = int(shape_data['width'])
        self.length = int(shape_data['height'])
        self.channels = int(shape_data['depth'])
    def __str__(self):
        print("The image resolution is {} by {} with {} channels".format(self.width, self.length, self.channels))