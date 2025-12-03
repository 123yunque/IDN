"""
ICAS Dataset Binary Classification using Homework Components
整合现有作业代码实现ICAS数据集二分类
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

# 导入现有作业模块
from test_data_loading import ThermalDataset  # 数据加载类
from instructor_solution_guide import (  # 指导方案中的工具函数
    get_data_transforms_solution,
    CustomCNN,
    ResNetClassifier,
    train_model_solution,
    evaluate_model_solution,
    plot_training_history_solution,
    plot_confusion_matrix_solution
)

# 配置参数
DATA_PATH = './dataset/ICAS'  # 数据集路径
BATCH_SIZE = 32
NUM_EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def main():
    # 1. 数据准备（使用test_data_loading中的ThermalDataset）
    train_transform, val_transform = get_data_transforms_solution()
    
    # 创建完整数据集
    full_dataset = ThermalDataset(DATA_PATH, transform=train_transform)
    
    # 划分数据集（70%训练，15%验证，15%测试）
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 应用验证集变换
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. 模型初始化（使用指导方案中的模型）
    model = CustomCNN(num_classes=2).to(DEVICE)
    # 可选：使用迁移学习模型
    # model = ResNetClassifier(num_classes=2).to(DEVICE)
    
    # 3. 训练模型（使用指导方案中的训练函数）
    print("Starting model training...")
    model, history = train_model_solution(
        model, train_loader, val_loader, NUM_EPOCHS, DEVICE
    )
    
    # 4. 可视化训练过程
    plot_training_history_solution(history)
    
    # 5. 评估模型（使用指导方案中的评估函数）
    print("Evaluating model on test set...")
    metrics = evaluate_model_solution(model, test_loader, DEVICE)
    
    # 打印评估指标
    print("\nTest Set Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # 绘制混淆矩阵
    plot_confusion_matrix_solution(metrics['confusion_matrix'], ['Non-ICAS', 'ICAS'])
    
    # 保存最终模型
    torch.save(model.state_dict(), 'final_icas_classifier.pth')
    print("\nModel saved as 'final_icas_classifier.pth'")

if __name__ == "__main__":
    # 检查数据集路径
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset path {DATA_PATH} not found!")
        print("Please check the path or update the DATA_PATH variable")
    else:
        main()