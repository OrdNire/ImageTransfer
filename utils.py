import random
import torch
import numpy as np
import torchvision.transforms.functional as FT

# 常量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456,
                                   0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224,
                                  0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor(
    [0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor(
    [0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)

# 判别器训练时利用前面记录的顺序，非最近生成器生成的数据，参考论文
# 目的：稳定训练过程
# 分拆图像的通道数据，随机取近max_size中三个通道组成图像
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0),'Empty buffer'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0) #图像升维
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                # 均匀分布判 0.5，表示取概率1/2
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

# 学习率衰减
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay_start_epoch must in range of n_epochs"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

# 权重初始化，参考论文
# 初始化正态分布N(0, 0.02)
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1,0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def convert_image(img, source, target):
    """
    转换图像格式.

    :参数 img: 输入图像
    :参数 source: 数据源格式, 共有3种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]'
    :参数 target: 数据目标格式, 共5种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]'
                   (4) 'imagenet-norm' (由imagenet数据集的平均值和方差进行标准化)
                   (5) 'y-channel' (亮度通道Y，采用YCbCr颜色空间, 用于计算PSNR 和 SSIM)
    :返回: 转换后的图像
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'
                      }, "无法转换图像源格式 %s!" % source
    assert target in {
        'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm', 'y-channel'
    }, "无法转换图像目标格式t %s!" % target

    # 转换图像数据至 [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)   #把一个取值范围是[0,255]的PIL.Image 转换成形状为[C,H,W]的Tensor，取值范围是[0,1.0]

    elif source == '[0, 1]':
        pass  # 已经在[0, 1]范围内无需处理

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # 从 [0, 1] 转换至目标格式
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # 无需处理

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                           rgb_weights) / 255. + 16.

    return img

if __name__ == '__main__':
    data = torch.ones((3, 256, 256))

    output = ReplayBuffer().push_and_pop(data)
    print(len(data.data))


