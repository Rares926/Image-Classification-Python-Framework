import tensorflow as tf 
from ..utils.optimizer_helper import OptimizerHelper

#parametrii o sa aibe valori de 0 daca nu trebuie transmise 

class Optimizer:
    def __init__(self):
        self.name=None
        self.params=None
    
    def build_optimizer_params(self,optimizer_data :dict):

        if not {'name','params'} <=optimizer_data.keys():
            raise Exception("Invalid optimizer format")
        
        self.name=optimizer_data["name"]
        self.params=OptimizerHelper.set_optimizer_value(optimizer_data["params"],self.name)
        

    def get_opt(self):
        """Dispatch method"""
        method_name = 'opt_' + str(self.name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid optimizier")
        # Call the method as we return it
        return method()

    def opt_Adam(self):
        return tf.keras.optimizers.Adam(learning_rate=self.params["lr"],
                                        beta_1=self.params["beta_1"],
                                        beta_2=self.params["beta_2"],
                                        epsilon=self.params["epsilon"],
                                        decay=self.params["decay"],
                                        amsgrad=self.params["amsgrad"]
                                        )

    def opt_SGD(self):
        tf.keras.optimizers.SGD(learning_rate=0.01,
                                momentum=0.0,
                                nesterov=False,
                                name='SGD',
                                )
