import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from random import random

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

def erreurab(I0,a,b,T):
    I = Iab(a,b,len(I0),T)
    return erreur2(I,I0)

def erreur2(I1,I2):
    s = 0
    assert len(I1) == len(I2)
    for t in range(len(I1)):
        s += (I1[t]-I2[t])**2
    return sqrt(s)

def randomizer(arr,r):
    return [v*random()*r for v in arr]

def naive(I0,arange,brange,T,p):
    """Essaie p*p valeurs de (a,b) dans les intervalles donnés et évalue l'erreur par rapport à I0 en chaque point"""
    mat = []
    xs = np.linspace(arange[0],arange[1],p)
    ys = np.linspace(brange[0],brange[1],p)
    n = len(I0)
    for a in xs:
        mat.append([])
        for b in ys:
            I = Iab(a,b,n,T)
            err = erreur2(I0,I)
            mat[-1].append(err)
    return np.array(mat),xs,ys


n = 1000
a = 1.5e-5
b = 0.1
T = 365
pop = 10000

I = Iab(a,b,n,T)

arange = [a*0.8, a*1.2]
brange = [b*0.5, b*1.5]
p = 25
mat,xs,ys = naive(I,arange,brange,T,p)

plt.figure(figsize=(p,p),dpi=90)
plt.pcolormesh(xs,ys,mat,cmap="RdYlBu")
plt.xlabel("a")
plt.ylabel("b")
plt.title("Bleu : erreur plus grande \n (a,b)=({},{})".format(a,b))
plt.show()
    
    
