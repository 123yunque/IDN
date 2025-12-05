import torch
import torch.nn as nn

class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.bce_loss = nn.BCELoss()
         # 类别权重：根据样本比例（647:303）设置，少数类权重更高
        pos_weight = torch.tensor([647/303])  # 正类（ICAS）权重
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 替换原有BCE

    
    def forward(self, x, y, z, label):
        alpha_1, alpha_2, alpha_3 = 0.3, 0.4, 0.3
        label = label.view(-1, 1)
        # print(max(x), max(label))
        loss_1 = self.bce_loss(x, label)
        loss_2 = self.bce_loss(y, label)
        loss_3 = self.bce_loss(z, label)
        return torch.mean(alpha_1*loss_1 + alpha_2*loss_2 + alpha_3*loss_3)

# def loss(x, y, z, label):
#     bce_loss = nn.BCELoss()
#     alpha_1, alpha_2, alpha_3 = 1, 1, 1
#     loss_1 = self.bce_loss(x, label)
#     loss_2 = self.bce_loss(y, label)
#     loss_3 = self.bce_loss(z, label)
#     return torch.mean(torch.add(torch.add(torch.mul(alpha_1, loss_1), torch.mul(alpha_2, loss_2)), torch.mul(alpha_3, loss_3)))
