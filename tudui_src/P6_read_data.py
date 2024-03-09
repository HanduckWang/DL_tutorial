from torch.utils.data import Dataset
import os
import cv2


class MyData(Dataset):  # 类名要采用驼峰命名法且避免使用下划线

    def __init__(self, root_dir, label_dir):  # 函数要使用小写字母且用下划线链接
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path_list = os.listdir(self.path)  # 列出目录中的文件和子目录输出为列表

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]
        img_item = os.path.join(self.root_dir, self.label_dir, img_name)
        img = cv2.imread(img_item)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path_list)


root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
img, label = ants_dataset[1]

train_dataset = ants_dataset + bees_dataset
cv2.imshow('Image', img)
cv2.waitKey(5000)
cv2.destroyAllWindows()




















































































