from second import *
from net import *

import torch
from matplotlib import pyplot as plt
from matplotlib import colors

N = 20

aa = np.linspace(arange[0],arange[1],N)
bb = np.linspace(brange[0],brange[1],N)
mat1 = []

net = Net()
net.load_state_dict(torch.load("model"))
net.eval()
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

fig, ax = plt.subplots(1,1)

ax.pcolormesh(mat1,cmap="RdBu")
ax.set_title("Prediction loss (red is lower)")
ax.set_xlabel("b")
ax.set_ylabel("a")

plt.show()
