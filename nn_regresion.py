import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13,30)
        self.fc2 = nn.Linear(30,10)
        self.fc3 = nn.Linear(10,1)
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# input = Variable(torch.randn(1,1,1))
# out = net(input)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss()

# Load Data
x_full,y_full = load_boston(return_X_y=True)
# Split the data for testing
x_train, x_test, y_train, y_test = train_test_split(
    x_full, y_full, test_size=0.3, random_state=42)
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))
# Normalization and preprocessing here

# Make the data into tensors
train_tensor = torch.tensor(np.concatenate((x_train,y_train), axis=1)).float()
test_tensor = torch.tensor(np.concatenate((x_test,y_test), axis=1)).float()

for epoch in range(100):
    for i, data2 in enumerate(train_tensor):
        X=data2[0:13]
        Y=data2[13]
        X, Y = Variable(X, requires_grad=True), Variable(Y, requires_grad=False)
        optimizer.zero_grad()
        y_pred = net(X)
        output = criterion(y_pred, Y)
        output.backward()
        optimizer.step()
    if (epoch % 20 == 0.0):
        print("Epoch {} - loss: {}".format(epoch, output))

