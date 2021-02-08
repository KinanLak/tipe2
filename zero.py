from keras_common import *
from matplotlib import pyplot as plt
import numpy as np
from random import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([ #Définition du modèle, assez arbitraire pour l'instant
        
])
model.compile(optimizer="rmsprop", loss="mean_squared_error")
model.build(input_shape=(1,1000,1)) #input shape : 1 entrée à 1000 caractéristiques : une courbe d'infectés
return model