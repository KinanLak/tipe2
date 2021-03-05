from second import *
from matplotlib import pyplot as plt
from matplotlib import colors

N = 20

aa = np.linspace(arange[0],arange[1],N)
bb = np.linspace(brange[0],brange[1],N)
mat1 = []
mat2 = []
mat3 = []

for i in range(N):
    a = aa[i]
    mat1.append([])
    mat2.append([])
    mat3.append([])
    for j in range(N):
        b = bb[j]
        I,x = odeintI(a,b,pop,n,T)
        m1 = 0
        m2 = 0
        m3 = 0
        for k in range(n):
            if I[k] > m1:
                m1 = I[k]
                m2 = k
            if k>0 and I[k]-I[k-1]>m3:
                m3 = I[k]-I[k-1]
        mat1[i].append(m1)
        mat2[i].append(m2)
        mat3[i].append(m3)
        

fig, axs = plt.subplots(3,2)
axs[0][0].pcolormesh(mat1,cmap="RdBu")
axs[1][0].pcolormesh(mat2,cmap="RdBu")
axs[2][0].pcolormesh(mat3,cmap="RdBu")

for i in range(N):
    axs[0][1].plot(np.linspace(0,N,N),mat1[i])
    axs[1][1].plot(np.linspace(0,N,N),mat2[i])
    axs[2][1].plot(np.linspace(0,N,N),mat3[i])

axs[0][0].set_title("Maximum infection as function of a and b")
axs[0][1].set_title("Maximum infection as function of b, colored by a")

axs[1][0].set_title("Maximum infection time as function of a and b")
axs[1][1].set_title("Maximum infection time as function of b, colored by a")


plt.show()
