import tensorflow as tf 
from ..utils.optimizer_helper import OptimizerHelper


#parametrii o sa aibe valori de 0 daca nu trebuie transmise 

class Optimizer:
    def __init__(self):
        self.name=None
        self.params=None
        self.lr=None
    
    def build_optimizer_params(self,optimizer_data :dict):

        if not {'name','params'} <=optimizer_data.keys():
            raise Exception("Invalid optimizer format")

        self.name=optimizer_data["name"]

        if {'lr'} <= optimizer_data.keys():
            self.lr=OptimizerHelper.set_learning_rate(self.name,optimizer_data["lr"])
        else: self.lr=OptimizerHelper.set_learning_rate(self.name)

        self.params=OptimizerHelper.set_optimizer_value(optimizer_data["params"],self.name)
        

    def get_opt(self):
        """Dispatch method"""
        method_name = 'opt_' + str(self.name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid optimizier")
        # Call the method as we return it
        return method()

    def opt_Adam(self):
        return tf.keras.optimizers.Adam(learning_rate=self.lr,
                                        beta_1=self.params["beta_1"],
                                        beta_2=self.params["beta_2"],
                                        epsilon=self.params["epsilon"],
                                        decay=self.params["decay"],
                                        amsgrad=self.params["amsgrad"]
                                        )

    def opt_SGD(self):
        return tf.keras.optimizers.SGD(learning_rate=self.lr,
                                momentum=self.params["momentum"],
                                nesterov=self.params["nesterov"],
                                )

    def opt_Adadelta(self):
        return tf.keras.optimizers.Adadelta(learning_rate=self.lr,
                                     rho=self.params["rho"],
                                     epsilon=self.params["epsilon"],
                                     )

    def opt_Adagrad(self):
        return tf.keras.optimizers.Adagrad(learning_rate=self.lr,
                                     initial_accumulator_value=self.params["initial_accumulator_value"],
                                     epsilon=self.params["epsilon"],
                                     )

    def opt_RMSprop(self):
        return tf.keras.optimizers.RMSprop(learning_rate=self.lr,
                                     rho=self.params["rho"],
                                     momentum=self.params["momentum"],
                                     epsilon=self.params["epsilon"],
                                     centered=self.params["centered"]
                                     )



                               