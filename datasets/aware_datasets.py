import os
import torch as t
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms as T
import cv2
from torch.utils.data import DataLoader
import glob
from pathlib import Path
import torch
from PIL import Image, ImageStat
import random


class AwareData(data.Dataset):

    def __init__(self, root="/home/shw/data/uveAwareData", transforms=None, mode="train"):
        '''
        获取所有图片的地址，并根据训练，测试，验证三类对数据进行划分
        '''
        self.root = os.path.join(root, mode)
        self.imgs = glob.glob(str(Path(self.root) / '**' / '*.*'), recursive=True)  # 获取root路径下所有图片的地址
        self.smooth = False
        print(self.imgs)
        # print imgs

    def __getitem__(self, index):
        """
        返回图像数据和标签，0代表白天1代表晚上
        """
        img_path = self.imgs[index]

        if self.smooth:
            label = brightness(img_path) / 255.0
            # print(label)
        else:
            if str(img_path).__contains__("day"):
                label = 0
            else:
                label = 1

        data = cv2.imread(img_path)  # BGR
        data = data[..., ::-1]  # BGR->RGB
        data = cv2.resize(data, (128, 128)).transpose(2, 0, 1)

        return torch.from_numpy(data), label

    def __len__(self):
        """
        返回数据集中所有图片的个数
        """
        return len(self.imgs)


def brightness(im_file):
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.rms[0]


if __name__ == '__main__':

    train_dataset = AwareData("/home/shw/data/uveAwareData", mode="train")  # 训练集
    # val_data = AwareData("/home/shw/data/uveAwareData", mode="val")  # 验证集

    train_dataloader = DataLoader(train_dataset, 1, shuffle=True, num_workers=1)

    for ii, (data, label) in enumerate(train_dataloader):
        # 训练模型参数
        print(data.shape, label)
