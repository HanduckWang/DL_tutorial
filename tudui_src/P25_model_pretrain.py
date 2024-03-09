import torchvision

# 此模型是针对ImageNet数据集训练
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

# 1000个输出，可以print seq的结构
print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("../dataset_3", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)

# 调整vgg16适合cifar10数据集的10个输出
vgg16_true.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

# 替换方法
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
