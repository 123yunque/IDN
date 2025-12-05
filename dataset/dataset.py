import os
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import random

class ThermalDataset(Dataset):
    """热图像分类数据集（ICAS/Non-ICAS）"""
    def __init__(self, data_root, split='train', transform=None, oversample=False):
        """
        Args:
            data_root (str): 数据集根目录，包含'icas'和'non_icas'子文件夹
            split (str): 数据集划分，可选'train'/'val'/'test'
            transform (callable): 图像变换函数
            oversample (bool): 是否对少数类进行过采样（仅对训练集有效）
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.oversample = oversample and (split == 'train')  # 仅训练集过采样
        
        # 类别映射
        self.classes = ['non_icas', 'icas']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 加载所有样本路径和标签
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(data_root, cls)
            if not os.path.exists(cls_dir):
                raise ValueError(f"类别文件夹不存在: {cls_dir}")
            
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls]))
        
        # 划分数据集（70%训练，15%验证，15%测试）
        self.samples = self._split_dataset()
        
        # 处理类别不平衡（过采样少数类）
        if self.oversample:
            self.samples = self._oversample_minority()
        
        # 打印数据集信息
        self._print_dataset_info()

    def _split_dataset(self):
        """划分训练/验证/测试集"""
        # 固定随机种子确保划分一致
        random.seed(42)
        np.random.seed(42)
        
        # 按类别分层抽样，确保分布一致
        all_indices = np.arange(len(self.samples))
        all_labels = np.array([label for _, label in self.samples])
        
        # 先划分训练集（70%）和临时集（30%）
        train_indices, temp_indices, _, _ = train_test_split(
            all_indices, all_labels, test_size=0.3, stratify=all_labels, random_state=42
        )
        
        # 从临时集划分验证集（15%总数据）和测试集（15%总数据）
        val_indices, test_indices, _, _ = train_test_split(
            temp_indices, all_labels[temp_indices], test_size=0.5, stratify=all_labels[temp_indices], random_state=42
        )
        
        # 根据当前split选择对应索引
        split_indices = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }[self.split]
        
        return [self.samples[i] for i in split_indices]

    def _oversample_minority(self):
        """过采样少数类（ICAS）以平衡数据集"""
        # 分离多数类和少数类样本
        majority_samples = [s for s in self.samples if s[1] == 0]  # non_icas
        minority_samples = [s for s in self.samples if s[1] == 1]   # icas
        
        # 计算过采样倍数（使少数类数量接近多数类）
        oversample_factor = len(majority_samples) // len(minority_samples)
        if oversample_factor < 1:
            return self.samples  # 无需过采样
        
        # 过采样少数类
        oversampled_minority = []
        for _ in range(oversample_factor):
            oversampled_minority.extend(minority_samples)
        
        # 合并并打乱
        balanced_samples = majority_samples + oversampled_minority
        random.shuffle(balanced_samples)
        return balanced_samples

    def _print_dataset_info(self):
        """打印数据集统计信息"""
        labels = [label for _, label in self.samples]
        class_counts = {cls: labels.count(idx) for cls, idx in self.class_to_idx.items()}
        
        print(f"[{self.split}] 数据集加载完成 - 总样本数: {len(self.samples)}")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} 样本 ({count/len(self.samples):.2%})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像（RGB模式）
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"无法加载图像 {img_path}: {e}")
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


# 测试数据集加载
if __name__ == "__main__":
    # 示例用法
    data_root = "dataset\\ICAS"  # 对应任务中的数据集路径
    
    # 定义变换
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 初始化数据集
    train_dataset = ThermalDataset(
        data_root, split='train', transform=train_transform, oversample=True
    )
    val_dataset = ThermalDataset(
        data_root, split='val', transform=val_transform, oversample=False
    )
    test_dataset = ThermalDataset(
        data_root, split='test', transform=val_transform, oversample=False
    )
    
    # 测试数据加载
    print("\n测试样本加载...")
    img, label = train_dataset[0]
    print(f"样本形状: {img.shape}, 标签: {label} (类别: {train_dataset.classes[int(label)]})")