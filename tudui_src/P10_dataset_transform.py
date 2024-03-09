import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])



train_set = torchvision.datasets.CIFAR10(root="./dataset_2", transform=dataset_transform, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset_2", transform=dataset_transform, train=False, download=True)
# img, target = test_set[0]
# img.show()
writer = SummaryWriter("p1")
for i in range(10):
    img, target = test_set[1]
    writer.add_image("test_set", img, i)