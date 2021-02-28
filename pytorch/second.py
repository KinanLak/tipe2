from scipy.integrate import odeint
from random import random
import numpy as np

def odeintI(a,b,pop,n,T):
    """Génération d'une courbe d'infectés solution du système différentiel SIR avec odeint"""
    t = np.linspace(0,T,n)
    solution = odeint(fct,[0,pop-1,1],t,args=(a,b))
    return solution[:,2],t

def fct(y, t, a, b): #pour odeint
    R,S,I = y
    return [b*I,-a*S*I,a*S*I-b*I]

def random_ab(arange,brange):
    """Choix aléatoire de coefficients a et b dans des intervalles arange=(amin,amax) brange=(bmin,bmax)"""
    a = arange[0] + (arange[1]-arange[0])*random()
    b = brange[0] + (brange[1]-brange[0])*random()
    return (a,b)

def normalize_labels(labels,arange,brange):
    r = []
    agap = arange[1]-arange[0]
    bgap = brange[1]-brange[0]
    for a,b in labels:
        a = (a-arange[0])/agap
        b = (b-brange[0])/bgap
        r.append((a,b))
    return r

def gen_data(arange=(2e-5,5e-5)
            ,brange=(0.05,0.1)
            ,pop=10000
            ,n=1000
            ,T=365
            ,size=100):
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

#VALEURS
arange = (2e-5,5e-5) #Intervalle de valeurs de alpha pour la génération des courbes
brange = (0.05,0.1) #Intervalle pour beta
pop = 10000 #Population
n = 1000 #nombre de points sur une courbe
T = 365 #durée représentée sur une courbe