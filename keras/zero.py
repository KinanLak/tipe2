#from keras_common import *
from matplotlib import pyplot as plt
import numpy as np
from random import random
from keras_common import odeintI
#import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import keras

def gen_data(arange,brange,pop,n,T,size):
    """Génération d'une de size courbes d'infectés, chacune à n valeurs sur une durée T, avec des coefficients dans arange et brange"""
    data = []
    labels = []
    for i in range(size):
        a = arange[0] + (arange[1]-arange[0])*random()
        b = brange[0] + (brange[1]-brange[0])*random()
        dp = np.array(odeintI(a,b,pop,n,T)[0]).reshape(1,1000)
        data.append(dp)
        labels.append((a,b))
    data = np.array(data)
    labels = np.array(labels)
    labels = np.squeeze(labels)
    return (data,labels)

def gen_dataset(arange,brange,pop,n,T,size):
    """Génération d'un dataset tensorflow à partir de gen_data"""
    data,labels = gen_data(arange,brange,pop,n,T,size)
    data = preprocessing(data)
    for i in range(len(labels)):
        a,b = labels[i]
        a = (a-arange[0])/(arange[1]-arange[0])
        b = (b-brange[0])/(brange[1]-brange[0])
        labels[i] = (a,b)
    return tf.data.Dataset.from_tensor_slices((data,labels))

def preprocessing(data):
    processed = []
    data = data.reshape(len(data),1000)
    for dp in data:
        mx = np.max(dp)
        avg = np.average(dp)
        mxd = 0
        last = dp[0]
        for i in range(len(dp)):
            if dp[i] == mx:
                mxp = i 
            delta = dp[i]-last
            if delta > mxd:
                mxd = delta
            last = dp[i]
        processed.append([mx,mxp,mxd,avg])
    return np.array(processed).reshape(len(processed),1,4)        


model = keras.Sequential([ #Définition du modèle, assez arbitraire pour l'instant
    keras.layers.Dense(20,activation="sigmoid"),
    keras.layers.Dense(2,activation="sigmoid")
])
opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(opt, loss="mean_absolute_percentage_error")
model.build(input_shape=(1,4)) #input shape : 1 entrée à n caractéristiques

"""
model = keras.models.load_model("model0")
keras.backend.set_value(model.optimizer.learning_rate, 1e-7)
"""

arange = (2e-5,5e-5) #Intervalle de valeurs de alpha pour la génération des courbes
brange = (0.05,0.1) #Intervalle pour beta
pop = 10000 #Population
n = 1000 #nombre de points sur une courbe
T = 365 #durée représentée sur une courbe
train_ds = gen_dataset(arange,brange,pop,n,T,2000) #Dataset d'apprentissage
val_ds = gen_dataset(arange,brange,pop,n,T,10) #Validation
test_ds = gen_dataset(arange,brange,pop,n,T,10)
arr = list(test_ds.as_numpy_iterator())

model.summary()

"""
keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
model.fit(train_ds,epochs=20,validation_data=val_ds,verbose=1,shuffle=True)
model.save("model0")
"""


data,labels = gen_data(arange,brange,pop,n,T,10)
processed = preprocessing(data)
x1,y1 = [l[0] for l in labels],[l[1] for l in labels]
ds = tf.data.Dataset.from_tensor_slices((processed,labels))

predictions = [model.predict(d) for d in processed]
#print(predictions)
x2,y2 = [l[0,0] for l in predictions],[l[0,1] for l in predictions]



fig, axs = plt.subplots(2,10)
"""
#ax1.scatter([l[0] for l in labels],[[l[1] for l in labels]])
ax1.scatter(x1,y1)
ax1.set_title("Data labels")
ax1.scatter(x2,y2)
"""

predictions = []
for i in range(20):
    model.fit(train_ds,epochs=1,validation_data=val_ds,verbose=1,shuffle=True)
    predictions.append(model.predict(arr[i][0]))
    x1,y1 = [l[0,0] for l in predictions],[l[0,1] for l in predictions]
    x2,y2 = [l[0,0] for l in arr[i][1]],[l[0,1] for l in arr[i][1]]
    x = i%10
    y = i//10
    axs[x][y].scatter(x1,y1)
    axs[x][y].scatter(x2,y2)
plt.show()


"""
data = data.reshape(10,1000)
for dp in data:
    plt.plot(dp)
plt.show()
"""