import torch
import torchvision
from PIL import Image
import os

from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

# # 文件夹路径
# folder_path = "../test_dataset"
#
# # 遍历文件夹中的图片文件
# for filename in os.listdir(folder_path):
#     if filename.endswith(".png") or filename.endswith(".jpg"):  # 确保文件是图片文件
#         # 图片路径
#         img_path = os.path.join(folder_path, filename)
#
#         # 读取图片并进行预处理
#         image = Image.open(img_path)
#         image = transform(image).to(device)
#         image = torch.reshape(image, (1, 3, 32, 32))  # 调整形状
#
#         # 使用模型进行识别
#         with torch.no_grad():
#             output = model(image)
#
#         # 解析输出结果
#         predicted_class = class_mapping[output.argmax(1).item()]
#         print("图片 '{}' 中的物体为：{}".format(filename, predicted_class))

img_path = "../test_dataset/beibei.png"
image = Image.open(img_path)

# resize输入pil格式最佳
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()])  # 注意多重小中括号
image = transform(image)
# cuda上训练的模型，需要把数据也转到cuda上才行
device = torch.device("cuda")
image = image.to(device)


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


# 创建数字到类型的映射字典
class_mapping = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

model = torch.load("duck_final.pth")
print(model)

image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)

predicted_class = class_mapping[output.argmax(1).item()]
print("该图片中的东西为：{}".format(predicted_class))

# resize和reshape,vision.resize用于调整图片大小然后totensor，reshape用于调整tensor大小
# 先使用vision.resize调整图像尺寸并将pil转化为tensor，reshape可以改变tensor形状
# cuda数据和普通数据转化
# with用法
"""
with用于创建上下文管理器，可以确保在进入和退出指定代码块时执行特定的操作
用于资源的管理和释放，例如文件操作打开关闭、数据库连接断开
with open('example.txt', 'r') as file:
    data = file.read()
    print(data)
在with语句的范围内，可以自由地读取文件内容
离开这个范围时，文件会被自动关闭

在该程序中不使用with就需要手动管理no_grad的生命周期
model.eval()  
torch.set_grad_enabled(False)  # 禁用梯度计算
output = model(image)
torch.set_grad_enabled(True)  

如果不禁用梯度计算会导致额外的计算开销和内存占用可能对模型的行为产生微弱影响，而且这些梯度在测试过程中不会被使用。
在测试阶段显式地禁用梯度计算是一个良好的做法可以确保推断过程的高效性和准确性，避免不必要的计算和内存消耗。
"""