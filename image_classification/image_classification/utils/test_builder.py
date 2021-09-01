

#Internal framework imports
from .json_helper import JsonHelper
from .network_params import NetworkParams

#Typing imports

class TestBuilder:
    def __init__(self):
        self.images_path = None
        self.network_path = None
        self.results_path = None
        self.topK = None
        self.image_shape = None
        self.image_format = None

    def arg_parse(self, path: str):
        raw_data = JsonHelper.read_json(path)
        if not {'network_path', 'images_path','results_path','top_k', 'network'} <= raw_data.keys():
            raise Exception("Invalid config file format")
        self.network_path = raw_data['network_path']
        self.images_path = raw_data['images_path']
        self.results_path =raw_data['results_path']
        self.topK = raw_data['top_k']
        network_params = NetworkParams()
        network_params.build_network_params(raw_data['network'])
        self.image_shape, self.image_format,_ = network_params.get_network_params()


