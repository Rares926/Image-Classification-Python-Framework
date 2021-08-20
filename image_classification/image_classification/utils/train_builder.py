

#Internal framework imports
from .json_helper import JsonHelper
from .network_params import NetworkParams
from .image_shape import ImageShape
from .image_format import ImageFormat

#Typing imports

class TrainBuilder:
    def __init__(self):
        self.dataset_path = None
        self.workspace_path = None
        self.image_shape = None
        self.image_format = None

    def arg_parse(self, path: str):
        raw_data = JsonHelper.read_json(path)
        if not {'dataset_path', 'workspace_path','network'} <= raw_data.keys():
            raise Exception("Invalid config file format")
        self.dataset_path = raw_data['dataset_path']
        self.workspace_path = raw_data['workspace_path']
        network_params = NetworkParams()
        network_params.build_network_params(path)
        self.image_shape, self.image_format = network_params.get_network_params()

if __name__ == "__main__":
    a = TrainBuilder()
    a.arg_parse("C:\\Users\\Radu Baciu\\Desktop\\configs\\train.json")


