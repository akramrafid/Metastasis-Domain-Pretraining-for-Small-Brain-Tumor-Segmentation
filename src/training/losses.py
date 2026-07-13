"""
Loss functions (reconstruction, contrastive, DiceCE, and sensitivity-specificity tradeoff).
"""
import torch
import torch.nn as nn
from typing import Dict
from monai.losses import DiceCELoss, TverskyLoss

class SensitivitySpecificityLoss(nn.Module):
    """Sensitivity-Specificity Tradeoff Loss.
    
    Proposed by Huang et al. (2022) to address the high false-positive rate
    by controlling the trade-off between sensitivity and specificity.
    
    Formula:
        Loss = r * (1 - Sensitivity) + (1 - r) * (1 - Specificity)
    """
    def __init__(self, r: float = 0.05, eps: float = 1e-5, include_background: bool = False) -> None:
        """
        Args:
            r: Weight parameter controlling the tradeoff (default: 0.05).
            eps: Small epsilon for numerical stability.
            include_background: Whether to include background channel (index 0) in the loss calculation.
        """
        super().__init__()
        self.r = r
        self.eps = eps
        self.include_background = include_background

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates sensitivity-specificity loss.
        
        Args:
            pred: Predicted class probabilities or logits of shape [B, C, H, W, D].
                  Will apply sigmoid (if C=1 or multi-label) or softmax (if multi-class).
            target: Ground-truth binary labels of shape [B, C, H, W, D].
            
        Returns:
            Scalar loss tensor.
        """
        # Apply activation if logits are passed
        if pred.shape[1] > 1:
            # Multi-class: apply softmax
            probs = torch.softmax(pred, dim=1)
        else:
            # Binary/Multi-label: apply sigmoid
            probs = torch.sigmoid(pred)

        # Determine channels to iterate over
        start_channel = 0 if self.include_background else 1
        num_channels = probs.shape[1]

        total_loss = torch.tensor(0.0, device=pred.device)
        channels_counted = 0

        for c in range(start_channel, num_channels):
            p_c = probs[:, c, ...]
            y_c = target[:, c, ...]

            # Flatten to 1D vectors
            p_c_flat = p_c.contiguous().view(-1)
            y_c_flat = y_c.contiguous().view(-1)

            # Soft Sensitivity = TP / (TP + FN)
            tp = torch.sum(p_c_flat * y_c_flat)
            fn = torch.sum((1.0 - p_c_flat) * y_c_flat)
            sensitivity = tp / (tp + fn + self.eps)

            # Soft Specificity = TN / (TN + FP)
            tn = torch.sum((1.0 - p_c_flat) * (1.0 - y_c_flat))
            fp = torch.sum(p_c_flat * (1.0 - y_c_flat))
            specificity = tn / (tn + fp + self.eps)

            # Channel loss
            channel_loss = self.r * (1.0 - sensitivity) + (1.0 - self.r) * (1.0 - specificity)
            total_loss += channel_loss
            channels_counted += 1

        if channels_counted > 0:
            return total_loss / channels_counted
        return total_loss

def get_loss_function(loss_type: str, config: Dict = None) -> nn.Module:
    """Factory function to retrieve PyTorch loss functions.
    
    Args:
        loss_type: Type of loss ('dice_ce', 'tversky', 'sens_spec_tradeoff').
        config: Optional configuration overrides.
        
    Returns:
        nn.Module loss function.
    """
    cfg = config or {}
    if loss_type == "dice_ce":
        return DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=cfg.get("include_background", False)
        )
    elif loss_type == "tversky":
        # Focal Tversky configuration
        return TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            alpha=cfg.get("tversky_alpha", 0.3),
            beta=cfg.get("tversky_beta", 0.7),
            include_background=cfg.get("include_background", False)
        )
    elif loss_type == "sens_spec_tradeoff":
        return SensitivitySpecificityLoss(
            r=cfg.get("sens_spec_r", 0.05),
            include_background=cfg.get("include_background", False)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
