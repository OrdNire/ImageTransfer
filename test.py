import time

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode

from models import Generator, Discriminator, SRResNet
from utils import ReplayBuffer, LambdaLR, weights_init_normal, convert_image
from datasets import ImageDataset
import itertools
import tensorboardX
from torchvision.utils import save_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

size = 256

epoch = 13

# 超分模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 4      # 放大比例
srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
srresnet = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor).to(device)

# 加载权重
netG_A2B.load_state_dict(torch.load("checkpoints/netG_A2B{}.pth".format(epoch)))
netG_B2A.load_state_dict(torch.load("checkpoints/netG_B2A{}.pth".format(epoch)))
srresnet.load_state_dict(torch.load("checkpoints/checkpoint_srresnet.pth")['model'])


input_A = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
input_B = torch.ones([1, 3, size, size], dtype=torch.float).to(device)



def test_all_test():
    data_root = "datasets/vangogh2photo"

    # 数据增强
    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    dataloader = DataLoader(ImageDataset(data_root, transforms_, mode="test"), batch_size=1,
                            shuffle=False, num_workers=4)
    if not os.path.exists("outputs/A"):
        os.makedirs("outputs/A")
    if not os.path.exists("outputs/B"):
        os.makedirs("outputs/B")

    for i, batch in enumerate(dataloader):
        real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float).to(device)
        real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        save_image(fake_A, "outputs/A/{}.png".format(i))
        save_image(fake_B, "outputs/B/{}.png".format(i))

# 风景 -> 梵高
def test_image_B2A():
    data_root = "datasets/imagesB"
    if not os.path.exists("outputs/imagesA"):
        os.makedirs("outputs/imagesA")

    list_imgs = os.listdir(data_root)
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    for i, img_name in enumerate(list_imgs):
        img = transform(Image.open(os.path.join(data_root, img_name)))
        print(img.shape)
        real_B = torch.tensor(input_B.copy_(img), dtype=torch.float).to(device)

        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        fake_A = SRTransfer(fake_A)
        # print(fake_A.shape)
        save_image(fake_A, "outputs/imagesA/{}.png".format(i))

def test_image_A2B():
    data_root = "datasets/imagesA"
    if not os.path.exists("outputs/imagesB"):
        os.makedirs("outputs/imagesB")

    list_imgs = os.listdir(data_root)
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    for i, img_name in enumerate(list_imgs):
        img = transform(Image.open(os.path.join(data_root, img_name)))
        print(img.shape)
        real_A = torch.tensor(input_A.copy_(img), dtype=torch.float).to(device)

        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)

        fake_B = transforms.Resize((2000, 2000), interpolation=InterpolationMode.BILINEAR)(fake_B)

        save_image(fake_B, "outputs/imagesB/{}.png".format(i))

def SRTransfer(img):
    img = convert_image(img, source="[0, 1]", target="imagenet-norm")
    # img.unsqueeze_(0)
    print(img.shape)

    # 记录时间
    start = time.time()

    # 转移数据至设备
    img = img.to(device)  # (1, 3, w, h ), imagenet-normed

    # 模型推理
    with torch.no_grad():
        sr_img = srresnet(img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = convert_image(sr_img, source='[-1, 1]', target='[0, 1]')

    print('用时  {:.3f} 秒'.format(time.time()-start))

    return sr_img

if __name__ == '__main__':
    # test_all_test()
    test_image_B2A()
    # test_image_A2B()