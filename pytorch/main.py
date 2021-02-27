import torch
import torch.nn.functional as tnf
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(4,32)
        self.lin2 = nn.Linear(32,2) 

    def forward(self, x):
        x = tnf.relu(self.lin1(x))
        x = tnf.relu(self.lin2(x))
        return x

    def num_flat_feature(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)