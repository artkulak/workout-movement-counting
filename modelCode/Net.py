import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    '''
    Class contains CNN model architecture
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64 * 27 * 27, 512)
        self.bnorm1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 128)
        self.bnorm2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)
        self.bnorm3 = nn.BatchNorm1d(64)

        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.pool(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 64 * 27 * 27)
        x = self.dropout(x)
        x = F.relu(self.bnorm1(self.fc1(x)))
        x = F.relu(self.bnorm2(self.fc2(x)))
        x = F.relu(self.bnorm3(self.fc3(x)))
        x = self.fc4(x)
        return x


if __name__ == '__main__':
    pass