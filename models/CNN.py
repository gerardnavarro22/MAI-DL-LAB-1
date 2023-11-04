import torch.nn as nn


class Net(nn.Module):

    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=131072, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.max_pooling(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.max_pooling(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.max_pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out
