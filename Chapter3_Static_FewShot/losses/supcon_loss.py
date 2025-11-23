import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
    也可以作为 SimCLR (Self-Supervised) 使用，如果不传 labels 即可。
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 形状 [batch_size, n_views, feature_dim]
                      例如 [32, 2, 128] -> 32个样本，每个样本2个视图，特征维度128
            labels:   形状 [batch_size]
                      真实标签，例如 [32]
            mask:     手动指定 mask，通常不需要，用 labels 自动生成即可
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # 展平维度: [batch_size, n_views, dim] -> [batch_size, n_views]
        # 用于后续维度控制
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [batch_size, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        
        # 如果没传 labels，默认退化为 SimCLR (自监督)，即只有自身的增强视图是正样本
        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # mask[i, j] = 1 表示 i 和 j 是同一类
            mask = torch.eq(labels, labels.T).float().to(device)
        elif mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        # 拆分视图并拼接
        # contrast_count = 2 (n_views)
        contrast_count = features.shape[1]
        # unbind: 把 [B, 2, D] 拆成 两个 [B, D]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) 
        
        # 如果 contrast_mode == 'one', 只拿第一个视图做 anchor
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        else:
            # 'all': 所有视图都做 anchor (标准做法)
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        # 计算相似度矩阵 (Dot Product)
        # [2B, D] * [2B, D]^T -> [2B, 2B]
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # 为了数值稳定性，减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 构建 mask
        # mask 重复扩展以匹配 [2B, 2B] 的 logits
        mask = mask.repeat(anchor_count, contrast_count)
        
        # 屏蔽掉自己和自己的对比 (对角线)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算 Log-Sum-Exp (分母部分)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算最终 Loss
        # 只计算正样本对 (Same Class) 的 log_prob
        # mask.sum(1) 是每个 anchor 对应的正样本数量
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # 取反求平均
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# --- 测试代码 ---
if __name__ == '__main__':
    # 模拟数据
    bsz = 4
    n_views = 2
    feat_dim = 128
    
    # 模拟特征向量 [Batch=4, Views=2, Dim=128] (已归一化)
    features = torch.randn(bsz, n_views, feat_dim)
    features = torch.nn.functional.normalize(features, dim=2)
    
    # 模拟标签 [0, 1, 0, 1] -> 意味着第0个和第2个是同类，第1个和第3个是同类
    labels = torch.tensor([0, 1, 0, 1])
    
    criterion = SupConLoss(temperature=0.1)
    
    if torch.cuda.is_available():
        features = features.cuda()
        labels = labels.cuda()
        criterion = criterion.cuda()
        
    loss = criterion(features, labels)
    print(f"[测试] SupCon Loss Value: {loss.item()}")
    
    # 简单的数值检查：Loss 应该是一个正数
    assert loss.item() > 0
    print("✅ 损失函数测试通过！")