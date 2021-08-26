

#Internal framework imports
from .json_helper import JsonHelper
from .network_params import NetworkParams

#Typing imports

class TrainBuilder:
    def __init__(self):
        self.dataset_path = None
        self.workspace_path = None
        self.image_shape = None
        self.image_format = None
        self.ratios = None
        self.resize_method = None

    def arg_parse(self, path: str):
        raw_data = JsonHelper.read_json(path)
        if not {'dataset_path', 'workspace_path','network'} <= raw_data.keys():
            raise Exception("Invalid config file format")
        self.dataset_path = raw_data['dataset_path']
        self.workspace_path = raw_data['workspace_path']
        network_params = NetworkParams()
        network_params.build_network_params(raw_data['network'])
        self.image_shape, self.image_format, self.resize_method, self.ratios = network_params.get_network_params()



