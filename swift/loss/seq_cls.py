# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from .base import BaseLoss

def plcc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()

    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    cov = (pred_centered * target_centered).mean()
    pred_var = (pred_centered**2).mean()
    target_var = (target_centered**2).mean()

    corr = cov / (torch.sqrt(pred_var * target_var) + eps)
    return corr.clamp(-1.0, 1.0)

def plcc_loss_multitarget_stable(pred, target, eps=1e-6, var_floor=1e-12):
    B = pred.shape[0]
    pred = pred.reshape(B, -1).float()
    target = target.reshape(B, -1).float()
    if B <= 1:
        return pred.new_tensor(0.0)

    pred_c = pred - pred.mean(dim=0, keepdim=True)
    tgt_c  = target - target.mean(dim=0, keepdim=True)

    cov = (pred_c * tgt_c).mean(dim=0)
    pred_var = (pred_c ** 2).mean(dim=0)
    tgt_var  = (tgt_c ** 2).mean(dim=0)

    denom = torch.sqrt(pred_var * tgt_var).clamp_min(var_floor)
    corr = cov / (denom + eps)
    corr = corr.clamp(-1.0, 1.0)

    return 1.0 - corr.mean()

def fidelity_m3_loss_multitarget_stable(pred, target, eps=1e-6):
    B = pred.shape[0]
    pred = pred.reshape(B, -1).float()
    target = target.reshape(B, -1).float()
    if B <= 1:
        return pred.new_tensor(0.0)

    pred_d = pred.t().contiguous()    # [D,B]
    tgt_d  = target.t().contiguous()  # [D,B]

    preds = pred_d.unsqueeze(2) - pred_d.unsqueeze(1)  # [D,B,B]
    gts   = tgt_d.unsqueeze(2)  - tgt_d.unsqueeze(1)   # [D,B,B]

    triu = torch.triu_indices(B, B, offset=1, device=pred.device)
    preds = preds[:, triu[0], triu[1]]  # [D,N]
    gts   = gts[:, triu[0], triu[1]]    # [D,N]

    g = 0.5 * (torch.sign(gts) + 1.0)   # {0,0.5,1}

    sqrt2 = preds.new_tensor(math.sqrt(2.0))
    p = 0.5 * (1.0 + torch.erf(preds / sqrt2))
    p = p.clamp(0.0, 1.0)  # 关键

    a = (p * g).clamp_min(0.0)
    b = ((1.0 - p) * (1.0 - g)).clamp_min(0.0)

    loss = 1.0 - (torch.sqrt(a + eps) + torch.sqrt(b + eps))
    return loss.mean(dim=1).mean()

def plcc_per_dim(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred/target: [B, D]  (D=多个回归目标; 可以由 [B,K] 或 [B,...] flatten 得到)
    return: [D] 每一维的 PLCC
    """
    pred = pred.float()
    target = target.float()

    pred_c = pred - pred.mean(dim=0, keepdim=True)
    tgt_c  = target - target.mean(dim=0, keepdim=True)

    cov = (pred_c * tgt_c).mean(dim=0)
    pred_var = (pred_c ** 2).mean(dim=0)
    tgt_var  = (tgt_c ** 2).mean(dim=0)

    corr = cov / (torch.sqrt(pred_var * tgt_var) + eps)
    return corr.clamp(-1.0, 1.0)

def plcc_loss_multitarget(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred/target: [B,K] 或 [B,...]
    每个 target 维单独算 PLCC -> 取平均 -> loss = 1 - mean(plcc)
    """
    B = pred.shape[0]
    pred = pred.reshape(B, -1)
    target = target.reshape(B, -1)

    if B <= 1:
        return pred.new_tensor(0.0)

    corr = plcc_per_dim(pred, target, eps=eps)  # [D]
    return 1.0 - corr.mean()


def fidelity_m3_loss_multitarget(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred/target: [B,K] 或 [B,...]
    对每个 target 维 k：
      - 在 batch 内做两两差 preds_i - preds_j
      - 得到 p = 0.5*(1+erf(diff/sqrt2))
      - g = 0.5*(sign(gt_diff)+1)
      - loss_k = mean(1 - (sqrt(p*g)+sqrt((1-p)(1-g))))
    返回 mean_k(loss_k)
    """
    B = pred.shape[0]
    pred = pred.reshape(B, -1).float()     # [B, D]
    target = target.reshape(B, -1).float() # [B, D]
    D = pred.shape[1]

    if B <= 1:
        return pred.new_tensor(0.0)

    # 转成 [D, B]，方便对每个维度并行做 pairwise
    pred_d = pred.t().contiguous()    # [D, B]
    tgt_d  = target.t().contiguous()  # [D, B]

    # pairwise diff: [D, B, B]
    preds = pred_d.unsqueeze(2) - pred_d.unsqueeze(1)
    gts   = tgt_d.unsqueeze(2)  - tgt_d.unsqueeze(1)

    triu = torch.triu_indices(B, B, offset=1, device=pred.device)
    preds = preds[:, triu[0], triu[1]]  # [D, N]
    gts   = gts[:, triu[0], triu[1]]    # [D, N]

    g = 0.5 * (torch.sign(gts) + 1.0)   # [D, N]

    sqrt2 = preds.new_tensor(math.sqrt(2.0))
    p = 0.5 * (1.0 + torch.erf(preds / sqrt2))  # [D, N]

    # loss per dim -> mean over dims
    loss_per_dim = 1.0 - (torch.sqrt(p * g + eps) + torch.sqrt((1.0 - p) * (1.0 - g) + eps))  # [D, N]
    return loss_per_dim.mean(dim=1).mean(dim=0)  # scalar


class SmoothL1RegressionLoss(BaseLoss):

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs) -> torch.Tensor:
        logits = outputs.logits
        labels = labels.to(logits.device)
        return F.smooth_l1_loss(logits.squeeze(), labels.squeeze(), reduction='mean')
    
class PLCCLoss(BaseLoss):
    def __init__(self, args, trainer, eps: float = 1e-8):
        super().__init__(args, trainer)
        self.eps = eps

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs) -> torch.Tensor:
        logits = outputs.logits
        labels = labels.to(logits.device)
        return plcc_loss_multitarget(logits, labels, eps=self.eps)
    
    
class FidelityLoss(BaseLoss):
    """
    Prediction monotonicity related loss (based on pairwise ordering).
    输入:
      y_pred: shape [B] 或 [B, 1]
      y:      shape [B] 或 [B, 1]
    输出:
      scalar loss
    """
    def __init__(self, args, trainer, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # register_buffer 让它跟着 .to(device) 走，并且不参与训练
        self.register_buffer("sqrt2", torch.tensor(math.sqrt(2.0), dtype=torch.float32))

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 允许 [B,1] / [B]，统一拍扁成 [B]
        y_pred = y_pred.reshape(-1)
        y = y.reshape(-1)
        B = y_pred.size(0)
        assert B > 1, "batch size must be > 1 for pairwise monotonicity loss."

        # [B, 1] 方便做差
        y_pred = y_pred.unsqueeze(1)
        y = y.unsqueeze(1)

        preds = y_pred - y_pred.t()   # [B,B]
        gts = y - y.t()               # [B,B]

        # 取上三角 (i<j)
        triu = torch.triu_indices(B, B, offset=1, device=y_pred.device)
        preds = preds[triu[0], triu[1]]  # [N]
        gts = gts[triu[0], triu[1]]      # [N]

        # g = 0.5 * (sign(gts) + 1)  -> {0, 0.5, 1}
        g = 0.5 * (torch.sign(gts) + 1.0)

        # p = 0.5 * (1 + erf(preds / sqrt(2)))
        sqrt2 = self.sqrt2.to(device=preds.device, dtype=preds.dtype)
        p = 0.5 * (1.0 + torch.erf(preds / sqrt2))

        g = g.view(-1, 1)
        p = p.view(-1, 1)

        eps = self.eps
        # loss = mean( 1 - ( sqrt(p*g+eps) + sqrt((1-p)*(1-g)+eps) ) )
        loss = torch.mean(
            1.0 - (torch.sqrt(p * g + eps) + torch.sqrt((1.0 - p) * (1.0 - g) + eps))
        )
        return loss
    
# ===== Fidelity(M3) util =====
def fidelity_loss(y_pred: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    按你给的 loss_m3 逻辑实现（pairwise + erf -> p，然后用 sqrt(p*g)+sqrt((1-p)(1-g))）。
    y_pred/y: [B] or [B,1]
    """
    y_pred = y_pred.reshape(-1)
    y = y.reshape(-1)
    B = y_pred.size(0)

    # B==1 时 pairwise 不成立：返回 0（避免训练最后一个 batch size=1 崩）
    if B <= 1:
        return y_pred.new_tensor(0.0)

    y_pred = y_pred.unsqueeze(1)  # [B,1]
    y = y.unsqueeze(1)            # [B,1]

    preds = y_pred - y_pred.t()   # [B,B]
    gts = y - y.t()               # [B,B]

    triu = torch.triu_indices(B, B, offset=1, device=y_pred.device)
    preds = preds[triu[0], triu[1]]  # [N]
    gts = gts[triu[0], triu[1]]      # [N]

    g = 0.5 * (torch.sign(gts) + 1.0)  # {0,0.5,1}

    # p = 0.5 * (1 + erf(preds / sqrt(2)))
    sqrt2 = preds.new_tensor(math.sqrt(2.0))
    p = 0.5 * (1.0 + torch.erf(preds / sqrt2))

    g = g.view(-1, 1)
    p = p.view(-1, 1)

    loss = torch.mean(
        1.0 - (torch.sqrt(p * g + eps) + torch.sqrt((1.0 - p) * (1.0 - g) + eps))
    )
    return loss

l1_loss = nn.L1Loss()
# ===== Mixed loss =====
class MixedLoss(BaseLoss):
    """
    三个 loss 混合版:
      - SmoothL1
      - PLCC loss: 1 - PLCC
      - Fidelity(M3) loss: fidelity_m3_loss

    参数:
      w_smooth, w_plcc, w_fidelity: 权重
      normalize_weights: 是否把三权重归一化到和为 1
      eps: 数值稳定
    """
    def __init__(
        self,
        args,
        trainer,
        w_plcc: float = 0.5,
        w_fidelity: float = 0.5,
        normalize_weights: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__(args, trainer)
        self.w_plcc = float(w_plcc)
        self.w_fidelity = float(w_fidelity)
        self.normalize_weights = bool(normalize_weights)
        self.eps = eps

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs) -> torch.Tensor:
        logits = outputs.logits
        labels = labels.to(logits.device)

        pred = logits.squeeze()
        gt = labels.squeeze()

        # ---- PLCC loss ----
        # B==1 或方差为0会导致不稳定：plcc()里有 eps，但 B==1 相关性意义不大，这里也做一下保护
        plcc_loss = pred.new_tensor(0.0)
        if pred.numel() > 1:
            plcc_loss = plcc_loss_multitarget_stable(pred, gt, eps=self.eps)

        # ---- Fidelity(M3) ----
        fid = fidelity_m3_loss_multitarget_stable(pred, gt, eps=self.eps)

        # ---- weighted sum ----
        w2, w3 = self.w_plcc, self.w_fidelity
        if self.normalize_weights:
            s = (abs(w2) + abs(w3))
            if s > 0:
                w2, w3 = w2 / s, w3 / s

        total = w2 * plcc_loss + w3 * fid

        # 兼容一些 trainer 里传 loss_scale 的场景
        if loss_scale is not None:
            total = total * loss_scale

        return total