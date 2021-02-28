import torch
import torch.nn.functional as tnf
import torch.nn as nn

from second import *

from matplotlib import pyplot as plt

def RMSELoss(output,target):
    return torch.sqrt(torch.mean((output-target)**2))

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(1000,32)
        self.lin2 = nn.Linear(32,2) 

        self.lin1.weight.data.fill_(0)
        self.lin1.weight.data.fill_(0)

    def forward(self, x):
        x = tnf.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def num_flat_feature(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

data,labels = gen_data(arange,brange,10000,1000,365,500)
val_data,val_labels = gen_data(arange,brange,10000,1000,365,10)
val_data,val_labels = torch.tensor(val_data).float(), torch.tensor(val_labels).float()
inputs = torch.tensor(data).float()
targets = torch.tensor(labels).float()

criterion = RMSELoss

n = 500
for i in range(n):
    lr = (1e-8)*(n-i)/n + (1e-9)*i/n
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    opt.zero_grad()
    outputs=net(inputs)
    loss = criterion(outputs,targets)
    loss.backward()
    opt.step()
    print(criterion(net(val_data),val_labels))

labels = val_labels.detach().numpy()
predictions = net(val_data).detach().numpy().reshape(10,2)
print(labels.shape)
print(predictions.shape)
x1,y1 = [l[0] for l in predictions],[l[1] for l in labels]
x2,y2 = [l[0] for l in predictions],[l[1] for l in predictions]
fig,(ax1,ax2) = plt.subplots(1,2)
ax1.scatter(x1,y1)
ax1.scatter(x2,y2)
plt.show()
print(labels)
print(predictions)