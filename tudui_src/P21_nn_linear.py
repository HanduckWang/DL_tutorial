import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset_2", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)  # 此处需要把drop_last设置为true否则最后一个矩阵对应不上


class Duck(nn.Module):
    def __init__(self):
        super(Duck, self).__init__()
        self.linear1 = Linear(196608, 10)
    def forward(self, input):
        output = self.linear1(input)
        return output

duck = Duck()
writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, label = data
    print(imgs.shape)
    imgs = torch.reshape(imgs, (1, 1, 1, -1))
    print(imgs.shape)
    output = duck(imgs)
    writer.add_images("output_linear", output, step)
    step += 1
writer.close()