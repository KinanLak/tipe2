<<<<<<< HEAD
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential([
    keras.layers.Dense(6),
    keras.layers.Dense(2)
])
x = tf.ones((1,4))
y = model(x)
data = [[1,2,3,4]]
data = np.array(data)
model(data)
print(model.summary())
=======
from multiprocessing import Pool

def f(x):
    return x*x

with Pool(5) as p:
    print(p.map(f, [1, 2, 3]))
>>>>>>> d37076e8868beb34413104f699a3e048b9f8dc7a
