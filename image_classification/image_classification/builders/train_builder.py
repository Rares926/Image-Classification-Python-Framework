
#Internal framework imports
from ..network.augmentations     import Augment
from ..utils.helpers.json_helper import JsonHelper
from ..network.network_params    import NetworkParams
from ..network.optimizer         import Optimizer
from ..builders.network_builder  import NetworkBuilder

#Typing imports

class TrainBuilder:
    def __init__(self):
        self.dataset_path   = None
        self.workspace_path = None
        self.network        = NetworkBuilder()
        

    def arg_parse(self, path: str):

        raw_data = JsonHelper.read_json(path)

        if not {'dataset_path', 'workspace_path','model_path','network','augmentations'} <= raw_data.keys():
            raise Exception("Invalid config file format")

        self.dataset_path = raw_data['dataset_path']
        self.workspace_path = raw_data['workspace_path']
        self.network.model_path=raw_data['model_path']
        
        if {'epochs'} <=raw_data.keys():
            self.network.epochs=raw_data['epochs']
        if {'batch_size'} <=raw_data.keys():
            self.network.batch_size=raw_data['batch_size']

        network_params = NetworkParams()
        network_params.build_network_params(raw_data['network'])
        self.network.image_shape, self.network.image_format, self.network.resize_method, self.network.ratios, self.network.resize_after_crop = network_params.get_network_params()

        optimizer_params=Optimizer()
        optimizer_params.build_optimizer_params(raw_data['optimizer'])
        self.network.optimizer=optimizer_params.get_opt()

        augument_params=Augment()
        augument_params.build_augmentation_params(raw_data['augmentations'])
        self.network.augmentations=augument_params.get_aug_list()

        self.network.metrics=raw_data['metrics']