import torch.nn as nn


class CNN1Conv(nn.Module):

    def __init__(self, num_classes, p=0.2, p_conv=0.1):
        super(CNN1Conv, self).__init__()

        self.p = p
        self.p_conv = p_conv
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.dropout_conv = nn.Dropout(p_conv)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, padding='same')       # 256/2 = 128
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=327680, out_features=512)                                  # 128*128*32
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.max_pooling(out)
        out = self.dropout_conv(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def __str__(self):
        return f'CNN1Conv_drop{self.p}_dropconv{self.p_conv}'


class CNN2Conv(nn.Module):

    def __init__(self, num_classes, p=0.2, p_conv=0.1):
        super(CNN2Conv, self).__init__()

        self.p = p
        self.p_conv = p_conv
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.dropout_conv = nn.Dropout(p_conv)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding='same')       # 256/2 = 128
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')      # 128/2 = 64
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=262144, out_features=512)                                  # 64*64*64
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.max_pooling(out)
        out = self.dropout_conv(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.max_pooling(out)
        out = self.dropout_conv(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def __str__(self):
        return f'CNN2Conv_drop{self.p}_dropconv{self.p_conv}'


class CNN3Conv(nn.Module):

    def __init__(self, num_classes, p=0.2, p_conv=0.1):
        super(CNN3Conv, self).__init__()

        self.p = p
        self.p_conv = p_conv
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)
        self.dropout_conv = nn.Dropout(p_conv)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding='same')       # 256/2 = 128
        self.batch_norm_conv_1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')      # 128/2 = 64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')     # 64/2 = 32
        self.batch_norm_conv_2 = nn.BatchNorm2d(128)
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=131072, out_features=512)                                  # 32*32*128
        self.batch_norm_fcnn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.max_pooling(out)
        out = self.batch_norm_conv_1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.max_pooling(out)
        out = self.dropout_conv(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.max_pooling(out)
        out = self.batch_norm_conv_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch_norm_fcnn(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def __str__(self):
        return f'CNN3Conv_drop{self.p}_dropconv{self.p_conv}'
