import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 6)
        self.conv2 = nn.Conv1d(128, 256, 6)
        self.fc1 = nn.Linear(256*73, 256)
        self.fc2 = nn.Linear(256, 15)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

model = Net()
print(model)
