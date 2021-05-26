from matplotlib import pyplot as plt
from random import random
from math import log,exp
import numpy as np

def Iab(a,b,n,T):
    """"Génération d'une courbe d'infections en fonctions des coefficients a et b : méthode d'Euler"""
    S = pop-1
    I = [1]
    dt = T/n
    for i in range(n-1):
        new = min(a*dt*S*I[-1], S) #on ne peut pas infecter plus d'individus qu'il n'y en a de sains
        I.append(I[-1] - b*dt*I[-1] + new)
        S -= new
    return I

def Nk(I0,a,b,T,k=2):
    """ 'norme k' : erreur entre la courbe générée en utilisant a,b et la courbe I0 donnée, en calculant [la somme de (l'écart en chaque point, exposant k)], exposant 1/k"""
    I = Iab(a,b,len(I0),T)
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

def Nab(I0,amin,amax,bmin,bmax,T,k,size):
    """ évalue l'erreur Nk entre I0 et la courbe générée par a,b pour 50 valeurs de a, 50 valeurs de b, variant "logarithmiquement" sur amin->amax et bmin->bmax """
    mat = []
    X = []
    Y = []
    n = len(I0)
    #amin,amax = log(amin),log(amax)
    #bmin,bmax = log(bmin),log(bmax)
    da,db = (amax-amin)/size,(bmax-bmin)/size
    m = [0,0,0,0,0]
    a = amin + da/2
    for i in range(size):
        mat.append([])
        X.append([])
        Y.append([])
        b = bmin + db/2
        for j in range(size):
            X[i].append(a)
            Y[i].append(b)
            N = Nk(I0,a,b,T,k)
            """if N > m[0]:
                m = [N,a,a+da,b,b+db]"""
            mat[i].append(N)
            b += db
        a += da
    return mat,X,Y
    
def I0al(a,b,r,n,T):
    """Génère la courbe d'infections en fonction de a et b, en rajoutant une erreur aléatoire entre 0 et 5% sur chaque point"""
    I = Iab(a,b,n,T)
    for i in range(T):
        I[i] += I[i]*random()*r
    return I

def nab_iter(I,amin,amax,bmin,bmax,n,p,k=3):
    """mesure Nab sur amin->amax, bmin->bmax, séléctionne des intervalles plus précis où les valeurs sont faible, et recalcule Nab dessus, k fois"""
    T = len(I)
    for i in range(k):
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

def writearr(arr, filename):
    f = open(filename, "w")
    f.write("\n".join([str(v) for v in arr]))
    f.close()

"""
a = 3e-3
b = 3
n = 360
T = 1000

I1 = Iab(a,b,T,n)
I2 = Iab(a,b,T,2*n)
plt.plot(I2)
plt.plot(I1)
plt.show()

#Projection de la suite de la courbe d'infectés avec T points, à partir de n<T points
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

Is = []
for i in range(10):
    a = 0.000001*i
    Is.append(Iab(a,0.01,T,T))
    plt.plot(Is[-1])
plt.show()
"""

n = 1000
a = 1.2e-5
b = 0.1
T = 365
pop = 10000

amin = 1e-5
amax = 3e-5
bmin = 0.05
bmax = 0.1

I = Iab(a,b,T,n)

size = 50
mat,X,Y = Nab(I,amin,amax,bmin,bmax,T,2,size)
plt.figure(figsize=(size-1, size-1), dpi=90)
plt.pcolormesh(X,Y,mat, cmap="RdYlBu")
plt.xlabel("a")
plt.ylabel("b")
plt.title("Rouge : écart plus important")
plt.show()
"""

I = Iab(0.00001,0.02,1000)
plt.plot(I)
writearr(I,"Iab.txt")
plt.show()
"""
