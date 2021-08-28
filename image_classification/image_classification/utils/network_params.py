

#Internal framework imports
from image_classification.utils.image_format import ImageFormat
from .json_helper import JsonHelper
from .image_shape import ImageShape

#Typing imports

class NetworkParams:
    def __init__(self):
        self.image_shape = None
        self.image_format = None
        self.checkpoint=None

    def build_network_params(self, network_data: dict):

        if not {'input_shape', 'input_format','checkpoint'} <= network_data.keys():
            raise Exception("Invalid network params format")
        if not {'width', 'height', 'depth'} <= network_data['input_shape'].keys():
            raise Exception("Invalid input shape params")
        if not {'channels', 'data_type'} <= network_data['input_format'].keys():
            raise Exception("Invalid input format params")

        self.image_shape = ImageShape(network_data['input_shape'])
        self.image_format = ImageFormat(network_data['input_format'])
        if network_data['checkpoint']!="None":
            self.checkpoint=network_data['checkpoint']

    def get_network_params(self):
        return self.image_shape, self.image_format,self.checkpoint