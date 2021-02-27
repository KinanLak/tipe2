from keras_common import *
from matplotlib import pyplot as plt
import numpy as np
from random import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("model")
print(model.summary())

arange = (2e-5,5e-5) #Intervalle de valeurs de alpha pour la génération des courbes
brange = (0.05,0.1) #Intervalle pour beta
pop = 10000 #Population
n = 1000 #nombre de points sur une courbe
T = 365 #durée représentée sur une courbe

data,labels = gen_data(arange,brange,pop,n,T,100)
x1,y1 = [l[0] for l in labels],[l[1] for l in labels]
ds = tf.data.Dataset.from_tensor_slices((data,labels))

predictions = [model.predict(d) for d in data]
#print(predictions)
x2,y2 = [l[0,0] for l in predictions],[l[0,1] for l in predictions]

fig, (ax1,ax2) = plt.subplots(1,2)
#ax1.scatter([l[0] for l in labels],[[l[1] for l in labels]])
ax1.scatter(x1,y1)
ax1.set_title("Data labels")
ax2.scatter(x2,y2)
plt.show()