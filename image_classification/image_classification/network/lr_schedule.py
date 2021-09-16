import tensorflow as tf

#Internal framework imports
from .helpers.lr_schedule_helper import LRScheduleHelper

class LearningRateSchedule():

    def __init__(self):

        self.name=None
        self.params=None
        self.lr=None

    def build_learning_rate_params(self,lr:dict):

        if not {"value","schedule"}<=lr.keys():
            raise Exception("Invalid lossing rate format")

        self.lr=lr["value"]
        self.name=lr["schedule"]

        if not {"params"}<=lr.keys():
            self.params=LRScheduleHelper.set_learning_rate_schedule(self.name)
        else: 
            self.params=LRScheduleHelper.set_learning_rate_schedule(self.name,lr["params"])

    def get_lr(self):
        """Dispatch method"""
        method_name = 'opt_' + str(self.name)
        method = getattr(self, method_name, lambda: "Invalid optimizier")

        return method()

    def opt_ExponentialDecay(self):

        return tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr,
                                                             decay_steps=self.params["decay_steps"],
                                                             decay_rate=self.params["decay_rate"],
                                                             staircase=self.params["staircase"])

    def opt_PolynomialDecay(self):

        return tf.keras.optimizers.schedules.PolynomialDecay(
                            initial_learning_rate=self.lr,
                            decay_steps=self.params["decay_steps"],
                            end_learning_rate=self.params["end_learning_rate"],
                            power=self.params["power"],
                            cycle=self.params["cycle"],)