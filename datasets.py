import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
import os
from PIL import Image
import random

class ImageDataset(Dataset):
    def __init__(self, root = "", transform = None, mode = "train"):
        super(ImageDataset, self).__init__()
        self.path = root
        self.pathA = os.path.join(root, mode+"A")
        self.pathB = os.path.join(root, mode+"B")
        self.listA = os.listdir(self.pathA)
        self.listB = os.listdir(self.pathB)
        self.transform = torchvision.transforms.Compose(transform)

    def __getitem__(self, index):
        img_nameA = self.listA[index % len(self.listA)]
        img_nameB = random.choice(self.listB)
        img_A = self.transform(Image.open(os.path.join(self.pathA, img_nameA)))
        img_B = self.transform(Image.open(os.path.join(self.pathB, img_nameB)))
        return {"A":img_A, "B":img_B}

    def __len__(self):
        return max(len(self.listA), len(self.listB))

if __name__ == '__main__':
    transform_ = [torchvision.transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                  torchvision.transforms.ToTensor()]
    train_set = ImageDataset("datasets/apple2orange", transform_)
    train_set_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
    for i, batch in enumerate(train_set_loader):
        print(i)