import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
 
    
class AHRLoss(nn.Module):
    def __init__(self, 
                 temperature=0.07, 
                 label_smoothing=0.1, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.device = device

    def forward(self, logits, labels):
        batch_size = logits.size(0)
        num_neg = batch_size - 1
        
        # 1. 区分正负样本并计算负样本动态权重
        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        negative_mask = ~positive_mask
        negative_scores = logits[negative_mask].view(batch_size, -1)  # [B, num_neg]
        
        # 负样本权重计算（显式数值稳定化）
        max_neg = torch.max(negative_scores, dim=1, keepdim=True)[0]
        stabilized_neg = negative_scores - max_neg  # 避免exp溢出
        neg_weights = torch.exp(stabilized_neg / self.temperature)
        neg_weights = neg_weights / neg_weights.sum(dim=1, keepdim=True)  # 归一化

        neg_weights = neg_weights * 1.0 + 0.5 #确立区间
        
        # 2. 构建带权重的平滑标签
        smooth_label = torch.zeros((batch_size, batch_size), device=self.device)
        smooth_label[positive_mask] = 1.0 - self.label_smoothing
        smooth_label[negative_mask] = (self.label_smoothing / num_neg) * neg_weights.flatten()
        
        # 3. 计算softmax概率（显式数值稳定化）
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        stabilized_logits = logits - max_logits  # 避免exp溢出
        exp_logits = torch.exp(stabilized_logits / self.temperature)
        scores_softmax = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        
        # 4. 计算损失
        loss = -torch.sum(smooth_label * torch.log(scores_softmax + 1e-15), dim=1)

        loss = torch.mean(loss)
        
        return loss