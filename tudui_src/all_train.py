import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from all_model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10("../fianl_dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)  # 按住shift+字母也可大写
test_data = torchvision.datasets.CIFAR10("../fianl_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 定义训练设备，以下面较为完备方式训练，需要找到三个东西：网络模型，数据（输入，标签），损失函数，使用.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用GPU训练" if torch.cuda.is_available() else "使用CPU训练")

# 数据集length长度
train_data_length = len(train_data)
test_data_length = len(test_data)
print("训练数据集长度为：{}".format(train_data_length))
print("训练数据集长度为：{}".format(test_data_length))

# DataLoader加载数据集，将数据分为多个batch，注意drop_last，此处容易忽略报错
train_data_loader = DataLoader(train_data, batch_size=64, drop_last=True)
test_data_loader = DataLoader(test_data, batch_size=64, drop_last=True)

# 创建网络模型
duck = Duck()
duck = duck.to(device)

# 损失函数，使用交叉熵
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)


# 优化器
learning_rate = 0.01  # 把学习率单独拿出来，方便搜索调整
optimizer = torch.optim.SGD(duck.parameters(), lr=learning_rate)  # 参数中=...表示参数可选

# 设置训练参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数，每次iteration迭代为一次batch，一个epoch为过完所有batch即所有数据一遍
epoch = 20

# 添加TensorBoard，可视化损失函数变化情况
writer = SummaryWriter("../logs")

for i in range(epoch):
    print("-------第{}轮训练开始------".format(i))

    # 训练开始
    duck.train()  # 加上最好，某些情况不会报错
    for data in train_data_loader:  # 注意这里是loader，写成dataset了一直报错
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = duck(imgs)
        loss = loss_fn(outputs, targets)

        # 设置优化器
        optimizer.zero_grad()  # 梯度清零，非常关键
        loss.backward()
        optimizer.step()

        # 展示设置，避免过多，满百展示，注意loss为
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, loss：{}".format(total_train_step, loss.item()))  # .item()获得张量中的标量
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试开始
    duck.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = duck(imgs)

            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss：{}，正确率：{}".format(total_test_loss, total_accuracy/test_data_length))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)

# 保存模型
torch.save(duck, "duck_final.pth")

writer.close()