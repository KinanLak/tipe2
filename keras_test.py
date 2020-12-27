from keras_core import *

from scipy.integrate import odeint
from matplotlib import pyplot as plt
import numpy as np
from random import random

import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("model")

arange = (1e-5,5e-5)
brange = (0.01,0.1)
pop = 10000
n = 1000
T = 365

a = arange[0] + (arange[1]-arange[0])*random()
b = brange[0] + (brange[1]-brange[0])*random()
x = np.array(odeintI(a,b,pop,n,T)[0])
y = model.predict(x)
print(x.shape, y.shape)
#print("a = {}, b = {}, prediction = {}".format(a,b,y))
