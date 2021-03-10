import torch
#import torch.nn.functional as tnf
#import torch.nn as nn

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(4,4)
        self.lin2 = torch.nn.Linear(4,2)
        self.bn1 = torch.nn.BatchNorm1d(4)

    def forward(self, x):
        x = self.bn1(x)
        x = torch.sigmoid(self.lin1(x))
        x = self.lin2(x)
        return x

    def num_flat_feature(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
