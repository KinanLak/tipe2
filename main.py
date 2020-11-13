from matplotlib import pyplot as plt
from random import random
from math import log,exp

pop = 10000

def Iab(a,b,T):
    S = pop-1
    I = [1]
    R = 0
    for i in range(T-1):
        I.append(I[-1] + a*S*I[-1] - b*I[-1])
        S -= a*S*I[-2]
        R += b*I[-2]
    return I

def Nk(a,b,I0,k=2):
    I = Iab(a,b,len(I0))
    s = 0
    for t in range(len(I)):
        s += abs(I0[t] - I[t])**k
    return s**(1/k)

def Nab(I0,amin,amax,bmin,bmax,n,k=2):
    mat = []
    X = []
    Y = []
    amin,amax = log(amin),log(amax)
    bmin,bmax = log(bmin),log(bmax)
    da,db = (amax-amin)/n,(bmax-bmin)/n
    m = [0,0,0,0,0]
    a = amin + da/2
    for i in range(n):
        mat.append([])
        X.append([])
        Y.append([])
        b = bmin + db/2
        for j in range(n):
            X[i].append(exp(a))
            Y[i].append(exp(b))
            try:
                N = Nk(exp(a),exp(b),I0,k)
            except:
                print(exp(a),exp(b))
            if N > m[0]:
                m = [N,exp(a),exp(a+da),exp(b),exp(b+db)]
            mat[i].append(N)
            b += db
        a += da

    plt.figure(figsize=(n-1, n-1), dpi=90)
    plt.pcolormesh(X,Y,mat, cmap="RdYlBu")
    plt.show()
    return mat
    

def I0al(a,b,r,T):
    I = Iab(a,b,T)
    for i in range(T):
        I[i] += I[i]*random()*r
    return I

"""
Is = []
for i in range(10):
    a = 0.000001*i
    Is.append(Iab(a,0.01,1000))
    plt.plot(Is[-1])
plt.show()
"""
a = 0.00009
b = 0.01
I = I0al(a,b,0.01,1000)
Nab(I,0.00001,0.0003,0.005,0.02,51,2)

