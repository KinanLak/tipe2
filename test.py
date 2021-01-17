from keras_common import *

from matplotlib import pyplot as plt


arange = (1e-5,5e-5)
brange = (0.01,0.5)
size = 5
bs = t = np.linspace(brange[0],brange[1],size)
pop = 10000
n = 1000
T = 365

data = []
labels = []
a = arange[0] + (arange[1]-arange[0])*random()
for i in range(size):
    b = brange[0] + (brange[1]-brange[0])*random()
    dp = odeintI(a,b,pop,n,T)
    data.append(dp)
    labels.append((a,b))
data = np.array(data)
labels = np.array(labels)

colors = ["red","green","blue","orange","black"]
print(["{} : {}".format(colors[i], labels[i]) for i in range(size)])
for i in range(size):
    plt.plot(data[i][1], data[i][0], color=colors[i])
plt.show()
