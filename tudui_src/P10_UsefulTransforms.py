from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants/0013035.jpg")

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Tensor", img_tensor, 0)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_tensor[0][0][0])
writer.add_image("Normalize", img_norm, 1)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))  # 在代码补全中取消勾选分大小写
# PIL —> resize -> PIL
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 2)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL _> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])  # 注意先后顺序，后面所需输入应该和前面所需输出匹配
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 3)

# RandomCrop随机裁剪
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)


writer.close()