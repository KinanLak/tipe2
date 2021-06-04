import torch
import numpy as np

from net import Net
from functions import  preprocessing, gen_data, odeintI, random_ab, data_preprocessing, get_params

from matplotlib import pyplot as plt

net = Net()
net.load_state_dict(torch.load("model"))
net.eval()

#VALEURS
params = get_params()
arange,brange,pop,n,T = params

Is=[]
for i in range(1):
    a,b = random_ab(arange,brange)
    I,t = odeintI(a,b,pop,n,60)
    Is.append(I)
p_data = data_preprocessing(Is,params)
inputs = torch.tensor(p_data).float()

#A, B PREDICTION
out_a,out_b = net(inputs).detach().numpy().reshape(2,len(Is))
p_a,p_b = [],[]
for i in range(len(out_a)):
    p_a.append(arange[0] + (arange[1]-arange[0])*out_a[i])
    p_b.append(brange[0] + (brange[1]-brange[0])*out_b[i])

fig,ax = plt.subplots(1,1)
for i in range(len(p_a)):
    I,t2 = odeintI(p_a[i],p_b[i],pop,2*n,120)
    ax.plot(t,Is[i])
    ax.plot(t2,I)

plt.show()
