import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset_2", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Duck(nn.Module):
    def __init__(self,):  # ctrl+o
        super(Duck, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=
            2),  # 没有默认值的参数一般都是必填，卷积提取特征
            MaxPool2d(2),  # 池化层减小尺寸，降低过拟合
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5, padding=2),
            MaxPool2d(2),
            Flatten(),  # 全连接层需要一维的输入
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


duck = Duck()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(duck.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = duck(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()  # 每次反向传播（每个batchsize）之前要清零梯度，否则会累加
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)




