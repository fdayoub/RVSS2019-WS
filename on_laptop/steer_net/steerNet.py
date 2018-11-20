import numpy as np 
import torch as tr
import torch.nn as nn
import torch.nn.functional as F



class SteerNet(nn.Module):
    def __init__(self):
        super(SteerNet,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        
        self.fc1   = nn.Linear(16*18*18,120)
        self.fc2   = nn.Linear(120,10)
        self.fc3   = nn.Linear(10,1)
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)        
        x = F.max_pool2d(x,(2,2))
        
        x = self.conv2(x)
        x = F.relu(x)        
        x = F.max_pool2d(x,(2,2))
                       
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features  

def test():
    mynet = SteerNet()
    print(mynet)
    #mynet = tr.load("steerNet.pt")

if __name__ == "__main__":
    test()  
        
        
