import torch
import numpy as np

from net import Net
from functions import arange, brange, n, pop, T, preprocessing, gen_data, odeintI

from matplotlib import pyplot as plt
from matplotlib import gridspec

net = Net()
net.load_state_dict(torch.load("model"))
net.eval()

#VALEURS
arange = (2e-5,1e-4) #Intervalle de valeurs de alpha pour la génération des courbes
brange = (0.05,0.1) #Intervalle pour beta
pop = 10000 #Population
n = 1000 #nombre de points sur une courbe
T = 365 #durée représentée sur une courbe

#TEST SET
test_data,test_labels = gen_data(arange,brange,pop,n,T,50)
p_test_data,p_test_labels = preprocessing(test_data,test_labels)
inputs, targets = torch.tensor(p_test_data).float(), torch.tensor(p_test_labels).float()

fig = plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(1,3,width_ratios=[1,1,2])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])

#A, B PREDICTION/LABEL GRAPHS
predictions = net(inputs).detach().numpy().reshape(len(targets),2)

x1,y1 = [l[0] for l in targets],[l[1] for l in targets]
x2,y2 = [l[0] for l in predictions],[l[1] for l in predictions]

#ax1.axis([0,1,0,1])
ax1.set_title("prediction/label : a")
ax1.scatter(x1,x2,marker="+")

#ax2.axis([0,1,0,1])
ax2.set_title("prediction/label : b")
ax2.scatter(y1,y2,marker="+")

#LOSS COLORMESH GRAPH
mat1 = []
N = 30
aa = np.linspace(arange[0],arange[1],N)
bb = np.linspace(brange[0],brange[1],N)
criterion = torch.nn.MSELoss()

for i in range(N):
    a = aa[i]
    mat1.append([])
    for j in range(N):
        b = bb[j]
        I,x = odeintI(a,b,pop,n,T)
        p = preprocessing([I],[(a,b)])
        p_data,p_labels = p[0], p[1]
        inputs, targets = torch.tensor(p_data).float(), torch.tensor(p_labels).float()
        loss = criterion(net(inputs),targets)
        mat1[i].append(loss.item())

ax3.pcolormesh(mat1,cmap="RdBu")
ax3.set_title("Prediction loss (red is lower)")
ax3.set_xlabel("b")
ax3.set_ylabel("a")
#ax3.axis([brange[0],brange[1],arange[0],arange[1]])

plt.show()

print("LABELS")
print(np.array(p_test_labels).reshape(50,2))
print("PREDICTIONS")
print(predictions)
plt.show()

print("PARAMETERS")
for param in net.parameters():
    print(param.data)
