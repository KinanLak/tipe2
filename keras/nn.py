from random import random
from matplotlib import pyplot as plt
from main import Nk, Iab, I0al
from time import time

from multiprocessing import Pool

class Worker_drond():
    def __init__(self,W,V,Is,eps,wv):
        self.W = W
        self.V = V
        self.Is = Is
        self.eps = eps
        self.wv = wv
    def __call__(self,k):
        return drond(self.W,self.V,k,self.wv,self.Is,self.eps)

class Worker_drond:
    """pour la parallélisation du calcul de gradient"""
    def __init__(self,W,V,eps,wv):
        self.W = W
        self.V = V
        self.eps = eps
        self.wv = wv
    def __call__(self,k):
        global Is
        return drond(self.W,self.V,Is,k,self.eps,self.wv)

def drond(W,V,k,Is,eps,wv): 
    """
    dérivée partielle de erreurTotale(Is,W,V) en changeant le k-ieme coefficient de W ou V
        0 <= k < N
        wv = "w" ou "v"
        eps = epsilon : on change le coefficient étudié en multipliant par (1+eps) et (1-eps)
    """
    if wv == "w":
        W1,W2 = [W[i]*(1+eps) if i==k else W[i] for i in range(N)],[W[i]*(1-eps) if i==k else W[i] for i in range(N)]
        e1, e2 = erreurTotale(Is,W1,V), erreurTotale(Is,W2,V)
        return (e2-e1)/(W2[k]-W1[k])
    else:
        V1,V2 = [V[i]*(1+eps) if i==k else V[i] for i in range(N)],[V[i]*(1-eps) if i==k else V[i] for i in range(N)]
        e1, e2 = erreurTotale(Is,W,V1), erreurTotale(Is,W,V2)
        return (e2-e1)/(V2[k]-V1[k])

def gen_test(n,amin,amax,bmin,bmax):
    """génère des courbes pour des a et b aléatoires, avec un erreur aléatoire par point (max 5%)"""
    Is = []
    for i in range(n):
        a = amin+(amax-amin)*random()
        b = bmin+(bmax-bmin)*random()
        Is.append(I0al(a,b,0.05,N))
    return Is

def model(I,W,V):
    """"application du réseau de neurones : I = entrée, W = poids pour a, V = poids pour b"""
    a = 0
    b = 0
    for k in range(N):
        a += I[k]*W[k]
        b += I[k]*V[k]
    return [a,b]

def erreur(I,V,W):
    """erreur quadratique entre la courbe donnée I et la courbe générée par a,b donnés par le modèle en fonction de I et W,V"""
    a,b = model(I,W,V)
    return Nk(a,b,I,N)

def erreurTotale(Is,W,V):
    """Erreur moyenne du modèle pour une liste de courbes Is"""
    s = 0
    for I in Is:
        s += erreur(I,W,V)
    return s/len(Is)

<<<<<<< HEAD
=======
def erreur(I,V,W):
    a,b = model(I,W,V)
    return Nk(a,b,I,N)

def drond(W,V,k,wv,Is,eps):
    if wv == "w":
        W1,W2 = [W[i]*(1-eps) if i == k else W[i] for i in range(N)], [W[i]*(1+eps) if i == k else W[i] for i in range(N)]
        e1 = erreurTotale(Is,W1,V)
        e2 = erreurTotale(Is,W2,V)
        return (e2-e1)/(W2[k]-W1[k])
    else:
        V1,V2 = [V[i]*(1-eps) if i == k else V[i] for i in range(N)], [V[i]*(1+eps) if i == k else V[i] for i in range(N)]
        e1 = erreurTotale(Is,W,V1)
        e2 = erreurTotale(Is,W,V2)
        return (e2-e1)/(V2[k]-V1[k])

def gradmp(W,V,Is,eps):
    with Pool(50) as p:
        dpw = p.map(Worker_drond(W,V,Is,eps,"w"), range(N))
        dpv = p.map(Worker_drond(W,V,Is,eps,"v"), range(N))
        print(dpw,dpv)
        return (dpw,dpv)

>>>>>>> d37076e8868beb34413104f699a3e048b9f8dc7a
def grad(W,V,Is,eps):
    """gradient de l'erreur totale sur la liste de courbes Is évalué en W,V, eps sert au calcul des dérivées partielles"""
    gw = []
    gv = []
    for k in range(N):
        t1 = time()
        print("w{}/{}".format(k,N))
        W1,W2 = [W[i]*(1-eps) if i == k else W[i] for i in range(N)], [W[i]*(1+eps) if i == k else W[i] for i in range(N)]
        e1 = erreurTotale(Is,W1,V)
        e2 = erreurTotale(Is,W2,V)
        dp = (e2-e1)/(W2[k]-W1[k])
        gw.append(dp)
        t2 = time()
        print(round((t2-t1)*1000))
        t1 = t2
    for k in range(N):
        print("v{}/{}".format(k,N))
        V1,V2 = [V[i]*(1-eps) if i == k else V[i] for i in range(N)], [V[i]*(1+eps) if i == k else V[i] for i in range(N)]
        e1 = erreurTotale(Is,W,V1)
        e2 = erreurTotale(Is,W,V2)
        dp = (e2-e1)/(V2[k]-V1[k])
        gv.append(dp)
    return (gw,gv)

<<<<<<< HEAD
def grad_mp(W,V,Ips,eps):
    """équivalent à la fonction grad, avec multiprocessing"""
    with Pool() as p:
        p.map(Worker_drond(W,V,eps,"w"), range(N))
        p.map(Worker_drond(W,V,eps,"v"), range(N))
=======

def model(I,W,V):
    a = 0
    b = 0
    for k in range(N):
        a += I[k]*W[k]
        b += I[k]*V[k]
    return [a,b]
>>>>>>> d37076e8868beb34413104f699a3e048b9f8dc7a

def opti(Is,W,V,n):
    """optimisation des coefficients de W et V en utilisant le gradient de l'erreur"""
    eps = 1/100
    for i in range(n):
        print("OPTI " + str(i))
        gw,gv = grad(W,V,Is,eps)
        for k in range(n):
            W[k] += gw[k]*eps
            V[k] += gv[k]*eps
    return W,V

"""
def norme_gdt(gw,gv):
    s = 0
    for k in range(N):
        s += gw**2 + gv**2
    return s**1/2
"""


count = 0
N = 1000
if __name__ == "__main__":
    W,V = [0]*1000, [0]*1000
    for k in range(N):
        W[k] = 1e-7*(-1+2*random())
        V[k] = 1e-7*(-1+2*random())

    amin,amax = 1e-3, 1e-2
    bmin,bmax = 0.1, 10

    Is0 = gen_test(100,amin,amax,bmin,bmax)
    Is = Is0[1:]
    I = Is0[0]
    #W,V = opti(Is,W,V,5)
    print("2")
    gradmp(W,V,Is,1/100)
    """
    plt.plot(I)
    plt.plot(model(I,W,V))
    plt.show()
    """