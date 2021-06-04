from matplotlib import pyplot as plt
import numpy as np
from math import exp

res = 100

xs = np.linspace(-0.5,1.5,res)
ys = np.linspace(-1.1,1.1,res)

def local_minima(x,y,x0,y0):
    return -exp(-(x-x0)**2-(y-y0)**2)

minimas = [[0,0],[0,1],[1,1/2]]

X = np.zeros((res,res))
Y = np.zeros((res,res))
Z = np.zeros((res,res))
for i in range(len(xs)):
    x = xs[i]
    for j in range(len(ys)):
        y = ys[j]
        z = 0
        a = 1-exp(-(x+y)**2)
        b = 1-exp(-(x-y)**2)
        Z[i,j] = a*b
        X[i,j] = x
        Y[i,j] = y

"""
x0,y0 = 0.2,0.5
rate = 1
gdxs, gdys = [x0],[y0]
for k in range(10):
    i,j = int(x0//(xs[1]-xs[0])),int(y0//(ys[1]-ys[0]))
    gx = Z[i+1][j]-Z[i][j]
    gy = Z[i][j+1]-Z[i][j]
    x0 += gx*rate
    y0 += gy*rate
    gdxs.append(x0)
    gdys.append(y0)"""

x0,y0 = 1,0
i,j = int((x0+0.5)//(xs[1]-xs[0])),int((y0+1)//(ys[1]-ys[0]))
gdis, gdjs = [i],[j]
for k in range(100):
    mini = np.max(Z)
    mini_i,mini_j = 0,0
    offsets = [-1,0,1]
    for oi in offsets:
        for oj in offsets:
            if Z[i+oi][j+oj] <= mini:
                mini = Z[i+oi][j+oi]
                mini_i = i+oi
                mini_j = j+oj
    i = mini_i+1-1
    j = mini_j+1-1
    gdis.append(mini_i)
    gdjs.append(mini_j)

gxs,gys,gzs = [],[],[]
for k in range(len(gdis)):
    i = gdis[k]
    j = gdjs[k]
    gxs.append(X[i][j])
    gys.append(Y[i][j])
    gzs.append(Z[i][j]+0.1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(gxs,gys,gzs,color="r",linewidth=2,alpha=1)
ax.plot_surface(X,Y,Z,cmap="RdYlBu_r")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()
        
