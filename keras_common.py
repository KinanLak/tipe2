from scipy.integrate import odeint
from matplotlib import pyplot as plt
import numpy as np
from random import random

import tensorflow as tf
from tensorflow import keras

def odeintI(a,b,pop,n,T):
    t = np.linspace(0,T,n)
    solution = odeint(fct,[0,pop-1,1],t,args=(a,b))
    return solution[:,2],t

def fct(y, t, a, b):
    R,S,I = y
    return [b*I,-a*S*I,a*S*I-b*I]

def random_ab(arange,brange):
    a = arange[0] + (arange[1]-arange[0])*random()
    b = brange[0] + (brange[1]-brange[0])*random()
    return (a,b)

def gen_dataset(arange,brange,pop,n,T,size):
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
    print(data.shape, labels.shape)
    return tf.data.Dataset.from_tensor_slices((data,labels))