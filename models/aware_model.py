'''
本部分是光照感知模块，用来计算输入的可见光图像的光照强度
'''
from torch.autograd import Variable  # 这一步还没有显式用到variable，但是现在写在这里也没问题，后面会用到
import torch.nn as nn
import torch.nn.functional as F
from models.common import Focus, Conv


# define
class AwareModel(nn.Module):
    def __init__(self):
        super(AwareModel, self).__init__()
        self.focus = Focus(3, 16, 5) # /2

        # convolutional layer
        # self.conv1 = nn.Conv2d(3, 16, 5)
        # # max pooling layer
        self.pool = nn.MaxPool2d(2, 2) # /4
        self.conv2 = Conv(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 32)  # 线性层: 256长度 -> 84长度
        self.fc3 = nn.Linear(32, 2)  # 线性层：84长度 -> 2长度
        self.softmax = nn.Softmax(dim=1)  # Softmax

    def forward(self, x):
        # print("x.shape",x.shape)
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.focus(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        # print("x.shape", x.shape)
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        # print(x)
        return x


# create a complete CNN
model = AwareModel()
