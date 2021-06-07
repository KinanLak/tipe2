from functions import *
import numpy as np
from math import log,floor

def sci_round(x):
    tenpower = floor(log(x)/log(10))
    norm = x/10**(tenpower)
    roundx = round(norm*100)/100
    return str(roundx)+"e"+str(int(tenpower))

b = (brange[0]+brange[1])/2
for a in np.linspace(arange[0], arange[1],6):
    I, t = odeintI(a,b,pop,n,T)
    plt.plot(t, I, label="a="+sci_round(a))

a = (arange[0]+arange[1])*0.3
for b in np.linspace(brange[0], brange[1],6):
    I, t = odeintI(a,b,pop,n,T)
    plt.plot(t, I, label="b="+sci_round(b))

plt.xlabel("Temps")
plt.ylabel("Population infectée")
plt.title("b = " + sci_round(b))
plt.title("a = " + sci_round(a))
plt.legend()
plt.show()
