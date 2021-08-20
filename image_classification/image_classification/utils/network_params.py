

#Internal framework imports
from image_classification.utils.image_format import ImageFormat
from .json_helper import JsonHelper
from .image_shape import ImageShape

#Typing imports

class NetworkParams:
    def __init__(self):
        self.image_shape = None
        self.image_format = None

    def build_network_params(self, path:str):
        raw_data = JsonHelper.read_json(path)
        if not {'network'} <= raw_data.keys():
            raise Exception("Config has no network params")
        if not {'input_shape', 'input_format'} <= raw_data['network'].keys():
            raise Exception("Invalid network params format")
        if not {'width', 'height', 'depth'} <= raw_data['network']['input_shape'].keys():
            raise Exception("Invalid input shape params")
        if not {'channels', 'data_type'} <= raw_data['network']['input_format'].keys():
            raise Exception("Invalid input format params")
        self.image_shape = ImageShape(raw_data['network']['input_shape'])
        self.image_format = ImageFormat(raw_data['network']['input_format'])

    def get_network_params(self):
        return self.image_shape, self.image_format