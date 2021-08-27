import cv2 as cv
import numpy as np

#Internal framework imports
from .image_shape import ImageShape
from .image_format import ImageFormat
from .resize_worker import ResizeWorker
from .image_preprocessing import ImageProcessing
from .ratio import Ratio

#Typing imports

class ImageLoader:
    def __init__(self, image_shape: ImageShape, image_format: ImageFormat, resize_method: ResizeWorker, ratios: Ratio):
        self.image_shape = image_shape
        self.image_format = image_format
        self.resize_method = resize_method
        self.ratios = ratios

    def load_image(self, image_path: str):
        image = cv.imread(image_path)

        if self.image_format.channels_format == ImageFormat.ChannelsFormat.RGB:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif self.image_format.channels_format == ImageFormat.ChannelsFormat.GRAYSCALE:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        if self.image_format.data_format == ImageFormat.DataType.FLOAT:
            image = image.astype(np.float)
        
        if self.resize_method.strategy == ResizeWorker.ResizeMethod.CROP:
            image = ImageProcessing.crop(image, self.image_shape, self.ratios)
        if self.resize_method.strategy == ResizeWorker.ResizeMethod.STRETCH:
            image = ImageProcessing.stretch(image, self.image_shape)
        if self.resize_method.strategy == ResizeWorker.ResizeMethod.LETTERBOX:
            image = ImageProcessing.letterbox(image, self.image_shape)

        image = ImageProcessing.normalize(image)

        return image

    