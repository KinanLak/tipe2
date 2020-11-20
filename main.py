from matplotlib import pyplot as plt
from random import random
from math import log,exp

pop = 10000

def Iab(a,b,T,n):
    S = pop-1
    I = [1]
    for i in range(n-1):
        new = min(a/T*S*I[-1], S)
        I.append(I[-1] - b/T*I[-1] + new)
        S -= new
    return I

def Nk(a,b,I0,T,k=2):
    #print("N({},{},I0)".format(a,b))
    I = Iab(a,b,T,len(I0))
    s = 0
    for t in range(len(I)):
        try:
            s += abs(I0[t] - I[t])**k
        except:
            print("Except : N({},{},I0,{}) : s = {}, t = {}".format(a,b,k,s,t))
            f = open("tmp.txt", "w")
            f.write("\n".join( [str(I[t]) + " " + str(I0[t]) for t in range(len(I)) ] ))
            f.close()
            return 0
    return s**(1/k)

def Nab(I0,T,amin,amax,bmin,bmax,n,k=2):
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
            N = Nk(exp(a),exp(b),I0,T,k)
            if N > m[0]:
                m = [N,exp(a),exp(a+da),exp(b),exp(b+db)]
            mat[i].append(N)
            b += db
        a += da
    return mat,X,Y
    
def I0al(a,b,r,T):
    I = Iab(a,b,T,T)
    for i in range(T):
        I[i] += I[i]*random()*r
    return I

def writearr(arr, filename):
    f = open(filename, "w")
    f.write("\n".join([str(v) for v in arr]))
    f.close()

def dichobof(I,amin,amax,bmin,bmax,n,p):
    T = len(I)
    for i in range(3):
        mat,X,Y = Nab(I,amin,amax,bmin,bmax,50,2)
        n = len(mat)
        m = [mat[0][0],0,0]
        for x in range(n):
            for y in range(n):
                if mat[x][y] < m[0]:
                    m = [mat[x][y],x,y]
        x,y = m[1],m[2]
        Is.append(Iab(X[x][y],Y[x][y],T))
        print(m[0])
        width,height = (amax-amin)/20,(bmax-bmin)/20
        amin,amax = max(amin,X[x][y]-width/2), min(amax,X[x][y]+width/2)
        bmin,bmax = max(bmin,Y[x][y]-height/2), min(bmax,Y[x][y]+height/2)
    return (X[x][y],Y[x][y])

"""
a = 3e-3
b = 3
n = 360
T = 1000

#I = I0al(a,b,0.05,100)
I0 = Iab(a,b,T,T)
I = I0[:n]
amin = 1e-3
amax = 1e-2
bmin = 0.1
bmax = 10
writearr(I,"I.txt")
Is = [I]
for i in range(3):
    mat,X,Y = Nab(I,T,amin,amax,bmin,bmax,50,2)
    n = len(mat)
    m = [mat[0][0],0,0]
    for x in range(n):
        for y in range(n):
            if mat[x][y] < m[0]:
                m = [mat[x][y],x,y]
    x,y = m[1],m[2]
    #Is.append(Iab(X[x][y],Y[x][y],T,n))
    print(m[0])
    width,height = (amax-amin)/20,(bmax-bmin)/20
    amin,amax = max(amin,X[x][y]-width/2), min(amax,X[x][y]+width/2)
    bmin,bmax = max(bmin,Y[x][y]-height/2), min(bmax,Y[x][y]+height/2)
If = Iab(X[x][y],Y[x][y],T,T)
plt.plot(I0)
plt.plot(If)
plt.plot(I)
plt.show()
"""

"""
Is = []
for i in range(10):
    a = 0.000001*i
    Is.append(Iab(a,0.01,1000))
    plt.plot(Is[-1])
plt.show()
"""
"""
n = 50
mat,X,Y = Nab(I,amin,amax,bmin,bmax,n,2)
plt.figure(figsize=(n-1, n-1), dpi=90)
plt.pcolormesh(X,Y,mat, cmap="RdYlBu")
plt.show()
"""
"""
I = Iab(0.00001,0.02,1000)
plt.plot(I)
writearr(I,"Iab.txt")
plt.show()
"""
