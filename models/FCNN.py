import torch.nn as nn


class FCNN1Layers(nn.Module):

    def __init__(self, num_classes, p=0.2, p_conv=0.1):
        super(FCNN1Layers, self).__init__()

        self.p = p
        self.p_conv = p_conv
        self.fc1 = nn.Linear(in_features=256 * 256 * 3, out_features=num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)

        return out

    def __str__(self):
        return f'FCNN1Layers'


class FCNN2Layers(nn.Module):

    def __init__(self, num_classes, p=0.2, p_conv=0.1):
        super(FCNN2Layers, self).__init__()

        self.p = p
        self.p_conv = p_conv
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.fc1 = nn.Linear(in_features=256 * 256 * 3, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def __str__(self):
        return f'FCNN2Layers_drop{self.p}'


class FCNN3Layers(nn.Module):

    def __init__(self, num_classes, p=0.2, p_conv=0.1):
        super(FCNN3Layers, self).__init__()

        self.p = p
        self.p_conv = p_conv
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.fc1 = nn.Linear(in_features=256 * 256 * 3, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)

        return out

    def __str__(self):
        return f'FCNN3Layers_drop{self.p}'
