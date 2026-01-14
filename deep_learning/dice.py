import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (N, C, H, W) - C classes
        # targets: (N, H, W) - class indices (0 or 1)
        
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        # Convert targets to one-hot: (N, H, W) -> (N, C, H, W)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Flatten spatial dimensions
        probs = probs.reshape(probs.size(0), num_classes, -1)
        targets_one_hot = targets_one_hot.reshape(targets_one_hot.size(0), num_classes, -1)

        intersection = (probs * targets_one_hot).sum(dim=2)
        union = probs.sum(dim=2) + targets_one_hot.sum(dim=2)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

