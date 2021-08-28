import tensorflow as tf 


#parametrii o sa aibe valori de 0 daca nu trebuie transmise 

class Optimizer:
    def __init__(self,optimizer_name="Adam", lr=0.01, clip=0.5):
        self.lr=lr
        self.clip = clip
        self.optimizer_name=optimizer_name

    def get_opt(self):
        """Dispatch method"""
        method_name = 'opt_' + str(self.optimizer_name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid optimizier")
        # Call the method as we return it
        return method()

    def opt_Adam(self):
        return tf.keras.optimizers.Adam(lr=self.lr, clipvalue=self.clip)

    def opt_example(self):
        return  tf.keras.optimizers.example(lr=self.lr, clipvalue=self.clip)