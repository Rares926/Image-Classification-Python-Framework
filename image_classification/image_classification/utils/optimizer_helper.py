

#de facut niste verificari in caz ca nu exista name ul samd
# de verificat si daca scrie cu litere mari sa translatesze in lowercase 
class OptimizerHelper:

# la punctrul cu hiperparametrii le adaug in config direct langa path uri 

    STR_TO_BOOL={
        "True":True,
        "False":False
    }

    DEFAULT_OPTIMIZER_VALUES={
        "Adam":
    	{
			"lr":0.001,
			"grad_clip":0.5,
			"beta_1":0.9, 
			"beta_2":0.999, 
			"epsilon":1e-07, 
			"decay":0.0, 
			"amsgrad":False
		},

        "SGD":
        {
            "lr":0.01,
            "momentum":0.0,
            "nesterov":"False"
        },

        "Adadelta":
        {
            "lr":0.001,
            "rho":0.95,
            "epsilon":1e-07
        },

        "Adagrad":
        {
            "lr":0.001,
            "initial_accumulator_value":0.1,
            "epsilon":1e-07
            
        },

        "RMSprop":
        {
            "lr":0.001,
            "rho":0.9,
            "momentum":0.0,
            "epsilon":1e-07,
            "centered":"False"
        }


    }

    def __init__(self):
        pass
    
    @staticmethod
    def set_optimizer_value(optimizer,name):
        opt_tmp={}

        for par in OptimizerHelper.DEFAULT_OPTIMIZER_VALUES[name]:
            if par in optimizer:
                if isinstance(optimizer[par],str):
                    opt_tmp[par]=OptimizerHelper.STR_TO_BOOL[optimizer[par]]
                else : opt_tmp[par]=optimizer[par]
            else: opt_tmp[par]=  OptimizerHelper.DEFAULT_OPTIMIZER_VALUES[name][par]
            
        return opt_tmp