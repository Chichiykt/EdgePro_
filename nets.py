from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)  ----(32, 1, 28, 28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2 ----(32, 6, 28, 28)
            nn.ReLU(),  # input_size=(6*28*28) ----(32, 6, 28, 28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14) ----(32, 6, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # ----(32, 16, 10, 10)
            nn.ReLU(),  # input_size=(16*10*10) ----(32, 16, 10, 10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5) ----(32, 16, 5, 5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # ---- (32, 120)
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),  # ----(32, 84)
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)  # ----(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class LeNet5_rank(nn.Module):
    def __init__(self):
        super(LeNet5_rank, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)  ----(200, 1, 28, 28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2 ----(200, 6, 28, 28)
            nn.ReLU(),  # input_size=(6*28*28) ----(32, 6, 28, 28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14) ----(200, 6, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # ----(200, 16, 10, 10)
            nn.ReLU(),  # input_size=(16*10*10) ----(200, 16, 10, 10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5) ----(200, 16, 5, 5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # ---- (200, 120)
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),  # ----(200, 84)
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)  # ----(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        self.CONV1 = x  # ----(200, 6, 14, 14)
        x = self.conv2(x)
        self.CONV2 = x  # ----(200, 16, 5, 5)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        self.FC1 = x  # ---- (200, 120)
        x = self.fc2(x)
        self.FC2 = x  # ----(200, 84)
        x = self.fc3(x)
        return x


class LeNet_AZ(nn.Module):  # net(inputs, perm, a1, a2, a3, a4)
    def __init__(self):
        super(LeNet_AZ, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)  ----(32, 1, 28, 28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2 ----(32, 6, 28, 28)
            nn.ReLU(),  # input_size=(6*28*28) ----(32, 6, 28, 28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14) ----(32, 6, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # ----(32, 16, 10, 10)
            nn.ReLU(),  # input_size=(16*10*10) ----(32, 16, 10, 10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5) ----(32, 16, 5, 5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # ----(32, 120)
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),  # ----(32, 84)
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)  # ----(32, 10)

    def forward(self, x, a1, a2, a3, a4):
        x = self.conv1(x)
        x = x * a1
        self.CONV1 = x
        x = self.conv2(x)
        x = x * a2
        self.CONV2 = x
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = x * a3
        self.FC1 = x
        x = self.fc2(x)
        x = x * a4
        self.FC2 = x
        x = self.fc3(x)
        return x
