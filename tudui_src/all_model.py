import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Sequential, Flatten, Linear


class Duck(nn.Module):
    def __init__(self):
        super(Duck, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),  # 没有默认值的参数一般都是必填，卷积提取特征
            MaxPool2d(2),  # 池化层减小尺寸，降低过拟合
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),  # 全连接层需要一维的输入
            Linear(64*4*4, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output


if __name__ == '__main__':
    duck = Duck()
    input = torch.ones((64, 3, 32, 32))
    output = duck(input)
    print(output.shape)