import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset_2", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Duck(nn.Module):
    def __init__(self):
        super(Duck, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        output_relu = self.relu1(input)
        return output, output_relu

duck = Duck()
writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, label = data
    writer.add_images("input_sig", imgs, step)
    output, output_relu= duck(imgs)
    writer.add_images("output_sig", output, step)
    writer.add_images("output_relu", output_relu, step)
    step += 1

writer.close()
