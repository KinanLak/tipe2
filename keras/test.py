from keras_common import *
from keras_train import *
from matplotlib import pyplot as plt
import numpy as np
from random import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

def get_predictions(data, model):
    predictions = [model.predict(d) for d in data]
    return ([p[0,0] for p in predictions], [p[0,1] for p in predictions])

arange = (2e-5,5e-5) #Intervalle de valeurs de alpha pour la génération des courbes
brange = (0.05,0.1) #Intervalle pour beta
pop = 10000 #Population
n = 1000 #nombre de points sur une courbe
T = 365 #durée représentée sur une courbe
size = 100

model = make_model()
train_data, train_labels = gen_data(arange,brange,pop,n,T,size)
train_ds = tf.data.Dataset.from_tensor_slices((train_data,train_labels))
val_data, val_labels = gen_data(arange,brange,pop,n,T,20)
val_ds = tf.data.Dataset.from_tensor_slices((val_data,val_labels))
test_data,test_labels = gen_data(arange,brange,pop,n,T,32)
arr = []
for i in range(5):
    model.fit(train_ds,batch_size=16,epochs=1,validation_data=val_ds)
    arr.append(get_predictions(test_data,model))


fig, axs = plt.subplots(2,6,sharey=True)
x,y = [l[0] for l in test_labels],[l[1] for l in test_labels]
axs[0][0].scatter(x,y)
axs[0][0].set_title("Test data labels")
x,y = [l[0] for l in val_labels],[l[1] for l in val_labels]
axs[1][0].scatter(x,y)
axs[1][0].set_title("Val. data labels")

for i in range(5):
    x, y = arr[i]
    ax = axs[(i+1)//6][i%5+1]
    ax.scatter(x,y)
    ax.set_title("Epoch {}/{}".format(i+1,len(arr)))
x,y = [l[0] for l in train_labels],[l[1] for l in train_labels]
axs[1][1].scatter(x,y)
axs[1][1].set_title("Train data labels")
plt.show()

