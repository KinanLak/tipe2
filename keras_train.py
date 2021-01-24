from keras_common import *

from scipy.integrate import odeint
from matplotlib import pyplot as plt
import numpy as np
from random import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def gen_dataset(arange,brange,pop,n,T,size):
    """Génération d'un dataset tensorflow à partir de gen_data"""
    data,labels = gen_data(arange,brange,pop,n,T,size)
    return tf.data.Dataset.from_tensor_slices((data,labels))

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
    
def make_model():
    """Création du modèle keras"""
    model = keras.Sequential([ #Définition du modèle, assez arbitraire pour l'instant
    layers.Dense(100,activation="sigmoid"),
    layers.Dense(2,activation="sigmoid")
    ])
    model.compile(optimizer="rmsprop", loss="mean_squared_error")
    model.build(input_shape=(1,1000)) #input shape : 1 entrée à 1000 caractéristiques : une courbe d'infectés
    return model

def train_model(model, params = ((2e-5,5e-5),(0.05,0.1),10000,1000,365,1000)):
    arange, brange, pop, n, T, size = params 
    train_ds = gen_dataset(arange,brange,pop,n,T,size) #Dataset d'apprentissage
    val_ds = gen_dataset(arange,brange,pop,n,T,5) #Validation
    model.fit(train_ds,batch_size=16,epochs=5,validation_data=val_ds)
    return model




"""
#VALEURS
arange = (2e-5,5e-5) #Intervalle de valeurs de alpha pour la génération des courbes
brange = (0.05,0.1) #Intervalle pour beta
pop = 10000 #Population
n = 1000 #nombre de points sur une courbe
T = 365 #durée représentée sur une courbe
"""

"""Script apparentissage complet
#GENERATION DES DATASETS
train_data, train_labels = gen_data(arange,brange,pop,n,T,1000) #Dataset d'apprentissage
train_ds = tf.data.Dataset.from_tensor_slices((train_data,train_labels))
val_data, val_labels = gen_data(arange,brange,pop,n,T,5) #Validation
val_ds = tf.data.Dataset.from_tensor_slices((val_data,val_labels))
test_ds = gen_dataset(arange,brange,pop,n,T,5) #Test
print("DATA READY")

a,b = random_ab(arange,brange)
x = np.array(odeintI(a,b,pop,n,T)[0]).reshape(1,1000) #Pour comparer avant et après l'apprentissage



#APPRENTISSAGE ET EVALUTAION
print(model.summary())
results1 = model.evaluate(val_ds) #Première évaluation de la performance du modèle (avant l'apprentissage)
y1 = model.predict(x)
model.fit(train_ds,batch_size=16,epochs=5,validation_data=val_ds)
results2 = model.evaluate(val_ds) #Deuxième évalution (après l'apprentissage)
y2 = model.predict(x)
print("Evaluation before training : {}".format(results1)) #Comparaison
print("Evaluation after training : {}".format(results2))
model.save("model") #Sauvegarde du modèle dans un fichier
print("a = {}, b = {}, prediction1 = {}, prediction2 = {}".format(a,b,y1,y2))
"""