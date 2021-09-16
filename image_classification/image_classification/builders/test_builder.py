

#Internal framework imports
from ..utils.helpers.json_helper import JsonHelper
from ..network.network_params    import NetworkParams
from .network_builder            import NetworkBuilder
#Typing imports

class TestBuilder:
    def __init__(self):
        self.images_path = None
        self.labels_path = None
        self.results_path = None
        self.topK = None
        self.network=NetworkBuilder()

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
        self.network.image_shape, self.network.image_format, self.network.resize_method, self.network.ratios = network_params.get_network_params()


