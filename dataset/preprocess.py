import os
import random
from sklearn.model_selection import train_test_split

# ICAS数据集路径（根据实际路径调整）
DATASET_ROOT = "./dataset/ICAS"
TRAIN_FILE = "icas_train.txt"
TEST_FILE = "icas_test.txt"
VAL_FILE = "icas_val.txt"  # 新增验证集划分

# 类别映射（保持与ThermalDataset一致）
CLASS_MAPPING = {
    "icas": 1,       # 阳性样本
    "non_icas": 0    # 阴性样本（注意原问题中"CAS/noicas"应为"non_icas"，与之前代码保持一致）
}

def generate_split_files():
    """生成训练集(70%)、验证集(15%)、测试集(15%)划分文件"""
    samples = []
    
    # 收集所有样本路径和标签
    for class_name, label in CLASS_MAPPING.items():
        class_dir = os.path.join(DATASET_ROOT, class_name)
        if not os.path.exists(class_dir):
            raise ValueError(f"路径不存在: {class_dir}")
        
        # 遍历该类别下所有图像文件
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_name, img_name)  # 相对路径，便于后续读取
                samples.append((img_path, label))
    
    # 分层抽样划分（保持类别比例）
    # 先划分为训练集和临时集（85%:15%）
    train_samples, temp_samples = train_test_split(
        samples, 
        test_size=0.15, 
        random_state=42, 
        stratify=[s[1] for s in samples]  # 按标签分层
    )
    # 再将临时集划分为验证集和测试集（1:1）
    val_samples, test_samples = train_test_split(
        temp_samples, 
        test_size=0.5, 
        random_state=42, 
        stratify=[s[1] for s in temp_samples]
    )
    
    # 写入文件（格式：图像路径 标签）
    def write_file(file_path, data):
        with open(file_path, 'w') as f:
            for img_path, label in data:
                f.write(f"{img_path} {label}\n")
    
    write_file(TRAIN_FILE, train_samples)
    write_file(VAL_FILE, val_samples)
    write_file(TEST_FILE, test_samples)
    
    # 打印划分结果
    print(f"数据集划分完成：")
    print(f"训练集：{len(train_samples)} 样本（{len(train_samples)/len(samples):.1%}）")
    print(f"验证集：{len(val_samples)} 样本（{len(val_samples)/len(samples):.1%}）")
    print(f"测试集：{len(test_samples)} 样本（{len(test_samples)/len(samples):.1%}）")

if __name__ == "__main__":
    generate_split_files()