from torch.utils import data
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class ICASDataset(data.Dataset):
    def __init__(self, root="dataset\\ICAS", split="train", transform=None):
        """
        适配ICAS二分类数据集的加载类
        :param root: 数据集根目录
        :param split: 划分类型："train"/"val"/"test"
        :param transform: 数据预处理/增强变换
        """
        # oversample_factor = 2  # 扩充少数类到原来的2倍
        # new_samples = []
        # new_labels = []
        # for sample, label in zip(self.datas, self.labels):
        #     new_samples.append(sample)
        #     new_labels.append(label)
        #     if label == 1:  # 少数类
        #         for _ in range(oversample_factor - 1):
        #             new_samples.append(sample)
        #             new_labels.append(label)
        # self.datas = new_samples
        # self.labels = new_labels
        # super(ICASDataset, self).__init__()
        self.root = root
        # self.split = split
        self.transform = transform
        
        # 加载划分文件
        split_file = f"icas_{split}.txt"
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        self.samples = []
        for line in lines:
            img_path, label = line.strip().split()
            self.samples.append((img_path, int(label)))
        
        # 若未指定transform，使用默认变换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 适配net网络输入
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                                     std=[0.229, 0.224, 0.225])   # ImageNet标准差
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        # 读取RGB图像（使用PIL确保与transform兼容）
        img = Image.open(f"{self.root}/{img_path}").convert("RGB")
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        # 注意：net网络的forward需要输入格式为 (batch_size, 2*C, H, W)
        # 这里将图像复制为双通道（参考特征+测试特征，ICAS任务中可视为相同输入的增强）
        # 若需要更合理的双通道设计，可替换为其他增强方式（如原图+反转图）
        img = torch.cat([img, img], dim=0)  # 形状变为 (6, H, W)，适配net的输入要求
        
        return img, float(label)

# 测试代码
if __name__ == "__main__":
    dataset = ICASDataset(split="train")
    print(f"样本数量：{len(dataset)}")
    img, label = dataset[0]
    print(f"图像形状：{img.shape}（适配net网络的双通道输入）")
    print(f"标签：{label}（0=non_icas, 1=icas）")