import torch
import torchvision

# 方法1
model1 = torch.load("vgg16_method1.pth")
print(model1)

# 方法2 官方推荐
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict("vgg16_method2.pth")
# model2 = torch.load("") 这样只输出参数
print(vgg16)

# 陷阱1，直接加载会报错，需要把网络辅助过来，或者import过来
# class Duck(nn.Module):
#     def __init__(self):
#         super(Duck, self).__init__()
#         self.model1 = Sequential(
#             Conv2d(3, 32, 5, padding=2)
#         )
#     def forward(self, x):
#         x = self.model1(x)
#         return x
model = torch.load("xian_jin.pth")