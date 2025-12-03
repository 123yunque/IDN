import torch
import torch.nn as nn
from models.stream import stream
import torchvision

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.stream = stream()
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
       # 原stream输出通道为128，拼接后为256（128+128），保持不变
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),# 输入维度匹配拼接后的通道数
            nn.BatchNorm1d(256),  # 增加批归一化
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 针对小样本的过拟合防护
            nn.Linear(256, 1),
            nn.Sigmoid()
    )

    def forward(self, inputs):
        # 移除原双通道拆分逻辑，直接使用完整3通道输入
        # 生成输入的" inverse "版本（用于对比学习）
        inputs_inverse = 255.0 - inputs  # 对RGB图像做反相处理（保持3通道）

        # 共享stream网络提取特征（输入为3通道）
        feat, feat_inv = self.stream(inputs, inputs_inverse)

        # 构建3种特征组合（适配原投票机制）
        cat_1 = torch.cat((feat, feat_inv), dim=1)
        cat_2 = torch.cat((feat_inv, feat), dim=1)
        cat_3 = torch.cat((feat, feat), dim=1)  # 基线组合

        # 分别通过分类器
        cat_1 = self.sub_forward(cat_1)
        cat_2 = self.sub_forward(cat_2)
        cat_3 = self.sub_forward(cat_3)

        return cat_1, cat_2, cat_3
    
    def sub_forward(self, inputs):
        out = self.GAP(inputs)
        out = out.view(-1, inputs.size()[1])
        out = self.classifier(out)

        return out

if __name__ == '__main__':
    net = net()
    x = torch.ones(1, 3, 32, 32)
    y = torch.ones(1, 3, 32, 32)
    x_ = torch.ones(1, 3, 32, 32)
    y_ = torch.ones(1, 3, 32, 32)
    out_1, out_2, out_3 = net(x, y, x_, y_)
    # vgg = torchvision.models.vgg13()
    # print(vgg)