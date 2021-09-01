

#Internal framework imports
from .image_shape import ImageShape

#Typing imports


class Ratio:
    def __init__(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right
    
    def size_calculator(self, image_shape: ImageShape):
        top_left_x = int(self.top_left * image_shape.width)
        top_left_y = int(self.top_left * image_shape.height)
        bottom_right_x = int(image_shape.width - self.bottom_right * image_shape.width)
        bottom_right_y = int(image_shape.height - self.bottom_right * image_shape.height)
        return top_left_x, top_left_y, bottom_right_x, bottom_right_y