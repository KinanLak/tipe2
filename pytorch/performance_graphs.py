import torch

from net import Net
from second import arange, brange, n, pop, T, preprocessing, gen_data, show_test
from matplotlib import pyplot as plt

import numpy as np

net = Net()
net.load_state_dict(torch.load("model"))
net.eval()

#TEST SET
test_data,test_labels = gen_data(arange,brange,10000,1000,365,50)
p_test_data,p_test_labels = preprocessing(test_data,test_labels)
test_inputs, test_targets = torch.tensor(p_test_data).float(), torch.tensor(p_test_labels).float()

predictions = show_test(net,test_inputs,test_targets)
          
print("LABELS")
print(np.array(p_test_labels).reshape(50,2))
print("PREDICTIONS")
print(predictions)
plt.show()

print("PARAMETERS")
for param in net.parameters():
    print(param.data)
