

#Internal framework inputs
from ...utils.helpers.io_helper import IOHelper
from ...utils.helpers.dict_helper import DICTHelper

class LRScheduleHelper:

    STR_TO_BOOL={
        "True":True,
        "False":False
    }

    DEFAULT_PARAMS_VALUES={

        "ExponentialDecay":
    	{
        "decay_steps":100000,
        "decay_rate":0.96,
        "staircase":True
		},

        "PolynomialDecay":
        {
        "decay_steps":100000,
        "end_learning_rate":0.0001,
        "power":1.0,
        "cycle":False,
        },
    }

    def __init__(self):
        pass
    
    @staticmethod
    def set_learning_rate_schedule(name:str,params=None):
        params_tmp={}
        
        #daca params nu apare sau daca apare dar nu are nici un element se transmit valorile de baza 
        if params==None or len(params)==0:
            if not {name}<=LRScheduleHelper.DEFAULT_PARAMS_VALUES.keys():
                raise Exception("This learning rate schedule is not implemented")

            return LRScheduleHelper.DEFAULT_PARAMS_VALUES[name]
        else:
            params=IOHelper.set_dictionary_keys_to_lower(params)

            params_tmp=DICTHelper.combine_dict_params(LRScheduleHelper.DEFAULT_PARAMS_VALUES,params,name)

        return params_tmp

    


