



class DICTHelper:

    STR_TO_BOOL={
        "True":True,
        "False":False
    }

    def __init__(self):
        pass

    @staticmethod
    def combine_dict_params(const_dict,config_dict,name):

        tmp_dict={}

        for key in const_dict[name]:

            if key in config_dict:
                if isinstance(config_dict[key],str):
                    tmp_dict[key]=DICTHelper.STR_TO_BOOL[config_dict[key]]
                else: tmp_dict[key]=config_dict[key]
            else: tmp_dict[key]=const_dict[name][key]

        return tmp_dict


