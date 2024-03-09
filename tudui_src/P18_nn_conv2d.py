import torch
import torchvision
from torch import nn
from torch.nn import Conv2d  # torch和torch.nn各是各
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset_2", train=False, transform=torchvision.transforms.ToTensor()
                                       , download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Duck(nn.Module):
    def __init__(self,):  # ctrl+o
        super(Duck, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

duck = Duck()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = duck(imgs)
    # print(imgs.shape)
    # print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30]) -> [128, 3, 30, 30] 不是3个chanel会报错
    output = torch.reshape(output, (-1, 3, 30, 30))  # 重新组织为新的形状而不改变其元素的数量或值，第一个不知道就写-1自动计算
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
