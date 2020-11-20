from random import random
from main import Nk, Iab, I0al

def gen_test(n,amin,amax,bmin,bmax):
    Is = []
    for i in range(n):
        a = amin+(amax-amin)*random()
        b = bmin+(bmax-bmin)*random()
        Is.append(I0al(a,b,0.05,N))
    return Is

def erreurTotale(Is,W,V):
    s = 0
    for I in Is:
        s += erreur(I,W,V)
    return s/len(Is)

def erreur(I,V,W):
    a,b = model(I,W,V)
    return Nk(a,b,I,N)

def grad(W,V,Is,eps):
    gw = []
    gv = []
    for k in range(N):
        print("w{}/{}".format(k,N))
        W1,W2 = W[::],W[::] 
        W1[k],W2[k] = W1[k]*(1-eps), W2[k]*(1+eps)
        e1 = erreurTotale(Is,W1,V)
        e2 = erreurTotale(Is,W2,V)
        dp = (e2-e1)/(W2[k]-W1[k])
        gw.append(dp)
    for k in range(N):
        print("v{}/{}".format(k,N))
        V1,V2 = V[::],V[::] 
        V1[k],V2[k] = V1[k]*(1-eps), V2[k]*(1+eps)
        e1 = erreurTotale(Is,W,V1)
        e2 = erreurTotale(Is,W,V2)
        dp = (e2-e1)/(V2[k]-V1[k])
        gv.append(dp)
    return (gw,gv)

def model(I,W,V):
    a = 0
    b = 0
    for k in range(N):
        a += I[k]*W[k]
        b += I[k]*V[k]
    return [a,b]

def opti(Is,W,V,n):
    eps = 1/100
    for i in range(n):
        print("OPTI " + str(i))
        gw,gv = grad(W,V,Is,eps)
        for k in range(n):
            W[k] += gw[k]*eps
            V[k] += gv[k]*eps
    return W,V

def norme_gdt(gw,gv):
    s = 0
    for k in range(N):
        s += gw**2 + gv**2
    return s**1/2

N = 1000
W,V = [0]*1000, [0]*1000
for k in range(N):
    W[k] = 1e-7*(-1+2*random())
    V[k] = 1e-7*(-1+2*random())

amin,amax = 1e-3, 1e-2
bmin,bmax = 0.1, 10

Is0 = gen_test(100,amin,amax,bmin,bmax)
Is = Is0[1:]
I = Is0[0]
W,V = opti(Is,W,V,5)
print("2")
plt.plot(I)
plt.plot(model(I,W,V))
plt.show()