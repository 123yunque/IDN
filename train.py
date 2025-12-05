import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# 导入数据集和模型
from dataset.dataset import ThermalDataset
from models.net import net
from loss import loss as LossFunction

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = torch.cuda.is_available()

def compute_metrics(predicted, labels):
    # 将元组转换为列表，方便修改
    predicted_list = list(predicted)
    
    # 对每个输出进行二值化处理（>0.5 视为 1，否则为 0）
    for i in range(3):
        predicted_list[i] = (predicted_list[i] > 0.5).float()
    
    # 融合三个输出的结果
    predicted_combined = predicted_list[0] + predicted_list[1] + predicted_list[2]
    predicted_combined[predicted_combined < 2] = 0
    predicted_combined[predicted_combined >= 2] = 1
    predicted_combined = predicted_combined.view(-1).cpu().numpy()
    
    # 转换标签为 numpy 数组
    labels_np = labels.view(-1).cpu().numpy()
    
    # 计算指标，添加zero_division参数处理无正样本情况
    acc = (predicted_combined == labels_np).mean()
    prec = precision_score(labels_np, predicted_combined, zero_division=0)
    rec = recall_score(labels_np, predicted_combined, zero_division=0)
    f1 = f1_score(labels_np, predicted_combined, zero_division=0)
    
    return acc, prec, rec, f1

def train_model():
    # 超参数设置
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    STEP_SIZE = 15
    GAMMA = 0.1

    # 固定随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed(42)

    # 数据变换
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 初始化数据集
    data_root = "dataset/ICAS"
    train_set = ThermalDataset(
        data_root, split='train', 
        transform=train_transform, 
        oversample=True
    )
    val_set = ThermalDataset(
        data_root, split='val', 
        transform=val_transform, 
        oversample=False
    )
    test_set = ThermalDataset(
        data_root, split='test', 
        transform=val_transform, 
        oversample=False
    )

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # 初始化模型、损失函数、优化器
    model = net().to(device)
    criterion = LossFunction()
    if cuda:
        criterion = criterion.to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # 日志和模型保存设置
    timestamp = time.strftime("%m-%d-%H-%M", time.localtime())
    writer = SummaryWriter(log_dir=f'runs/thermal_{timestamp}')
    best_val_f1 = 0.0
    os.makedirs('saved_models', exist_ok=True)

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0  # 初始化训练损失
        train_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}

        # 训练阶段
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播（三输出）
            outputs = model(inputs)
            loss_val = criterion(*outputs, labels)  # 使用loss_val避免变量名冲突

            # 反向传播
            loss_val.backward()
            optimizer.step()

            # 计算指标
            acc, prec, rec, f1 = compute_metrics(outputs, labels)
            train_loss += loss_val.item()  # 修复：使用loss_val累加
            train_metrics['acc'].append(acc)
            train_metrics['prec'].append(prec)
            train_metrics['rec'].append(rec)
            train_metrics['f1'].append(f1)

            # 日志记录
            global_step = (epoch-1)*len(train_loader) + i
            writer.add_scalar('train/loss', loss_val.item(), global_step)  # 修复：使用loss_val
            writer.add_scalar('train/accuracy', acc, global_step)

            # 打印中间结果
            if i % 20 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}], Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss_val.item():.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")  # 修复：使用loss_val

        # 训练集 epoch 指标
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = np.mean(train_metrics['acc'])
        avg_train_f1 = np.mean(train_metrics['f1'])

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss_val = criterion(*outputs, labels)  # 避免变量名冲突
                val_loss += val_loss_val.item()

                acc, prec, rec, f1 = compute_metrics(outputs, labels)
                val_metrics['acc'].append(acc)
                val_metrics['prec'].append(prec)
                val_metrics['rec'].append(rec)
                val_metrics['f1'].append(f1)

        # 验证集 epoch 指标
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = np.mean(val_metrics['acc'])
        avg_val_prec = np.mean(val_metrics['prec'])
        avg_val_rec = np.mean(val_metrics['rec'])
        avg_val_f1 = np.mean(val_metrics['f1'])

        # 学习率调度
        scheduler.step()

        # 记录验证指标
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/accuracy', avg_val_acc, epoch)
        writer.add_scalar('val/precision', avg_val_prec, epoch)
        writer.add_scalar('val/recall', avg_val_rec, epoch)
        writer.add_scalar('val/f1', avg_val_f1, epoch)

        # 保存最佳模型
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_val_f1
            }, f'saved_models/best_model_{timestamp}.pth')
            print(f"Best model saved (F1: {best_val_f1:.4f})")

        # 打印 epoch 总结
        print(f"\nEpoch [{epoch}/{EPOCHS}] Summary:")
        print(f"Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, F1: {avg_train_f1:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}, "
              f"Prec: {avg_val_prec:.4f}, Rec: {avg_val_rec:.4f}, F1: {avg_val_f1:.4f}\n")

    # 测试集评估
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(f'saved_models/best_model_{timestamp}.pth')['model_state_dict'])
    model.eval()
    test_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            acc, prec, rec, f1 = compute_metrics(outputs, labels)
            test_metrics['acc'].append(acc)
            test_metrics['prec'].append(prec)
            test_metrics['rec'].append(rec)
            test_metrics['f1'].append(f1)

    # 打印测试结果
    print("\nTest Set Results:")
    print(f"Accuracy: {np.mean(test_metrics['acc']):.4f}")
    print(f"Precision: {np.mean(test_metrics['prec']):.4f}")
    print(f"Recall: {np.mean(test_metrics['rec']):.4f}")
    print(f"F1-Score: {np.mean(test_metrics['f1']):.4f}")

    writer.close()

if __name__ == "__main__":
    train_model()