import torch
import torchvision
from torch import nn
from torch.nn import Conv2d

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1.模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2.模型参数 官方推荐
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


# 陷阱1
class Duck(nn.Module):
    def __init__(self):
        super(Duck, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2)
        )
    def forward(self, x):
        x = self.model1(x)
        return x


duck = Duck()
torch.save(duck, "xian_jin.pth")