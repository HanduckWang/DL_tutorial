import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset_2", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloder = DataLoader(dataset, batch_size=64)
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)  # 否则会报错，输入整数默认成long类型了
# print(input.shape)
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)

# 池化有点像1080p视频用720p观看一样，信息保留空间缩小
class Duck(nn.Module):
    def __init__(self):
        super(Duck, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

duck = Duck()
# output = duck(input)
# print(output)

writer = SummaryWriter("../logs")
step = 0
for data in dataloder:
    imgs, targets = data
    writer.add_images("input_maxpool", imgs, step)
    output = duck(imgs)
    writer.add_images("output_maxpool", output, step)
    step = step + 1

writer.close()
