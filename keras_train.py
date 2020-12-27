from keras_core import *

from scipy.integrate import odeint
from matplotlib import pyplot as plt
import numpy as np
from random import random

import tensorflow as tf
from tensorflow import keras

def gen_dataset(arange,brange,pop,n,T,size):
    data = []
    labels = []
    for i in range(size):
        a = arange[0] + (arange[1]-arange[0])*random()
        b = brange[0] + (brange[1]-brange[0])*random()
        data.append(np.array([odeintI(a,b,pop,n,T)[0]]))
        labels.append((a,b))
    data = np.array(data)
    data = np.squeeze(data)
    labels = np.array(labels)
    labels = np.squeeze(labels)
    print(data.shape, labels.shape)
    return tf.data.Dataset.from_tensor_slices((data,labels))

arange = (1e-5,5e-5)
brange = (0.01,0.1)
pop = 10000
n = 1000
T = 365

x = keras.Input(shape=(1,n))
y = keras.layers.Dense(100,activation="sigmoid")(x)
y = keras.layers.Dense(100,activation="sigmoid")(y)
y = keras.layers.Dense(2,activation="sigmoid")(y)
model = keras.Model(inputs=x,outputs=y)
model.compile(optimizer="rmsprop", loss="mean_squared_error")
print("MODEL READY")
print(model.summary())
train_ds = gen_dataset(arange,brange,pop,n,T,100)
val_ds = gen_dataset(arange,brange,pop,n,T,5)
test_ds = gen_dataset(arange,brange,pop,n,T,5)
print("DATA READY")
results1 = model.evaluate(val_ds)
model.fit(train_ds,batch_size=16,epochs=10,validation_data=val_ds)
results2 = model.evaluate(val_ds)
print("Evaluation before training : {}".format(results1))
print("Evaluation after training : {}".format(results2))
model.save("model")