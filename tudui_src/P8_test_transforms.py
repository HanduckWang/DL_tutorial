from torchvision import transforms # 用于把其他格式转换为tensor
import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


img_path = "dataset/train/ants/0013035.jpg"
img = cv2.imread(img_path)  # 此时img是numpy数组
# 可以使用Image为python内置库，得到PIL格式

writer = SummaryWriter("logs")  # 可以先写Summarywriter再通过报错去加import

tensor_trans = transforms.ToTensor()  # 可numpy可PIL格式
tensor_img = tensor_trans(img)

writer.add_image("Tensor_show", tensor_img, 2)  # 此处为tensor不用再指定dataformats

writer.close()
print(tensor_img)
