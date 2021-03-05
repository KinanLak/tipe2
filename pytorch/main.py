import torch
import torch.nn.functional as tnf
import torch.nn as nn

from second import *

from matplotlib import pyplot as plt

def RMSELoss(output,target):
    return torch.sqrt(torch.mean(((output-target)/target)**2))

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(4,4)
        self.lin2 = nn.Linear(4,2)
        self.bn1 = nn.BatchNorm1d(4)

        #self.lin1.weight.data.fill_(1e-9)
        #self.lin1.weight.data.fill_(1e-9)

    def forward(self, x):
        x = self.bn1(x)
        x = tnf.sigmoid(self.lin1(x))
        x = self.lin2(x)
        return x

    def num_flat_feature(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

#TRAINING SET
train_data,train_labels = gen_data(arange,brange,pop,n,T,5000) #data,labels = infected curve(array), (a,b)
p_train_data, p_train_labels = preprocessing(train_data,train_labels) #p_data, p_labels = preprocesed data/labels : (max,max position,max delta,average) and normalized a and b
train_inputs, train_targets = torch.tensor(p_train_data).float(), torch.tensor(p_train_labels).float() #inputs, labels = pytroch tensors for p_data and p_labels

#VALIDATION SET
val_data,val_labels = gen_data(arange,brange,10000,1000,365,50)
p_val_data, p_val_labels = preprocessing(val_data,val_labels)
val_inputs, val_targets = torch.tensor(p_val_data).float(), torch.tensor(p_val_labels).float()

#TEST SET
test_data,test_labels = gen_data(arange,brange,10000,1000,365,50)
p_test_data,p_test_labels = preprocessing(test_data,test_labels)
test_inputs, test_targets = torch.tensor(p_test_data).float(), torch.tensor(p_test_labels).float()

criterion = torch.nn.MSELoss() #Loss function

n = 5000
lr = 0.1
opt = torch.optim.SGD(net.parameters(), lr=lr)
for i in range(n):
    opt.zero_grad()
    outputs=net(train_inputs)
    loss = criterion(outputs,train_targets)
    loss.backward()
    opt.step()
    print(criterion(net(val_inputs),val_targets).item())

torch.save(net.state_dict(), "model")
        
predictions = show_test(net,test_inputs,test_targets)
          
print("LABELS")
print(np.array(p_test_labels).reshape(50,2))
print("PREDICTIONS")
print(predictions)
plt.show()

print("PARAMETERS")
for param in net.parameters():
    print(param.data)
