from tensorflow.keras import backend as K
import tensorflow.contrib.eager as tfe
import tensorflow as tf
import numpy as np
import pandas as pd



class WeightAnnealer_epoch(tf.keras.callbacks.Callback):
    '''Weight of variational autoencoder scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            weight for the VAE (float).
        Currently just adjust kl weight, will keep xent weight constant
    '''

    def __init__(self, schedule, weight, weight_orig, weight_name):
        super(WeightAnnealer_epoch, self).__init__()
        self.schedule = schedule
        self.weight_var = weight
        self.weight_orig = weight_orig
        self.weight_name = weight_name

    def on_epoch_begin(self, epoch, logs=None):

        
        if logs is None:
            logs = {}
        new_weight = self.schedule(epoch)
        new_value = new_weight * self.weight_orig
        print("Current {} annealer weight is {}".format(self.weight_name, new_value))
        assert type(
            new_weight) == float, 'The output of the "schedule" function should be float.'
        K.set_value(self.weight_var, new_value)

# Schedules for VAEWeightAnnealer
def no_schedule(epoch_num):
    return float(1)


def sigmoid_schedule(time_step, slope=1., start=None):
    return float(1 / (1. + np.exp(slope * (start - float(time_step)))))

