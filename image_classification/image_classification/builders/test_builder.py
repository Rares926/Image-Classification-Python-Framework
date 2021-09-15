

#Internal framework imports
from ..utils.helpers.json_helper import JsonHelper
from ..network.network_params import NetworkParams

#Typing imports

class TestBuilder:
    def __init__(self):
        self.images_path = None
        self.labels_path = None
        self.results_path = None
        self.topK = None
        self.image_shape = None
        self.image_format = None
        self.resize_method = None
        self.ratios = None

    def arg_parse(self, path: str):
        raw_data = JsonHelper.read_json(path)
        if not {'images_path','results_path','top_k', 'network'} <= raw_data.keys():
            raise Exception("Invalid config file format")
        self.images_path = raw_data['images_path']
        self.labels_path = raw_data['labels_path']
        self.results_path = raw_data['results_path']
        self.topK = raw_data['top_k']
        network_params = NetworkParams()
        network_params.build_network_params(raw_data['network'])
        self.image_shape, self.image_format, self.resize_method, self.ratios = network_params.get_network_params()


