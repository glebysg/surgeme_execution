import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np


def simple_gradient():
    # print the gradient of 2x^2 + 5x
    x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
    z = 2 * (x * x) + 5 * x
    # run the backpropagation
    z.backward(torch.ones(2, 2))
    print(x.grad)


def create_nn(batch_size=20, learning_rate=0.01, epochs=10,
              log_interval=10):
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

    # Normalization and preprocessing hnto the dataset manager
    train_loader = torch.utils.data.DataLoader(train_tensor,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_tensor,
        batch_size=batch_size, shuffle=True)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(13, 20)
            self.fc2 = nn.Linear(20, 20)
            self.fc3 = nn.Linear(20, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    print(net)

    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.MSELoss()

    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, full_data in enumerate(train_loader):
            data, target = Variable(full_data[:,0:13]), Variable(full_data[:,13])
            data = data.view(-1, 13)
            optimizer.zero_grad()
            net_out = net(data)
            print "TARGET:", target.view(20,1)
            loss = criterion(net_out, target.view(20,1))
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item()))

    # run a test loop
    test_loss = 0
    correct = 0
    epsilon = 0.1
    for data, target in test_loader:
        data, target = Variable(data, requires_grad=False), Variable(target)
        data = data.view(-1, 13)
        net_out = net(data)
        # sum up batch loss
        print "TARGET:", target.view(20,1)
        test_loss += criterion(net_out, target.view(20,1))
        # if the difference is less than the threshold, consider it an acurate response.
        # else, consider it a failed example
        pred = 1 if abs(target.data.item()-net_out.data.item()) < epsilon else 0
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    run_opt = 2
    if run_opt == 1:
        simple_gradient()
    elif run_opt == 2:
        create_nn()
