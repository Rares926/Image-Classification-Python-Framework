import os
from typing import final
import cv2 as cv
from cv2 import data
import numpy as np

#Internal framework imports
from .image_shape import ImageShape
from .image_format import ImageFormat
from .io_helper import IOHelper

#Typing imports

class ImageLoader:
    def __init__(self, image_shape: ImageShape, image_format: ImageFormat):
        self.image_shape = image_shape
        self.image_format = image_format

    def load_image(self, image_path: str):
        image = cv.imread(image_path)

        if self.image_format.channels_format == ImageFormat.ChannelsFormat.RGB:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif self.image_format.channels_format == ImageFormat.ChannelsFormat.GRAYSCALE:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        if self.image_format.data_format == ImageFormat.DataType.FLOAT:
            image = image.astype(np.float)
        final_image = image
        #final_image = cv.resize(image, (self.image_shape.width, self.image_shape.length))
            
        return final_image

    