from torch.utils.tensorboard import SummaryWriter
import cv2
'''
sm是用于将训练过程中数据可视化到特别的工具
'''

# 初始化sm，定义一个对象，指定保存数据的目录
writer = SummaryWriter("logs")

img_path = "dataset/train/ants/0013035.jpg"
img = cv2.imread(img_path)
writer.add_image("show", img, 1, dataformats='HWC' ) # 点进去看参数

for epoch in range(100):
    loss = 100 - epoch
    writer.add_scalar("training_loss", loss , global_step=epoch) # 保存一个标量
# 关闭sm
writer.close()
# tensorboard --logdir=logs

