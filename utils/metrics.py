import numpy as np
import torch
from pytorch_metric_learning import losses
from torch import nn
import torch.nn.functional as F


def _dice_loss(pred, target):
    smooth = 1e-5
    pred = F.softmax(pred, dim=1)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3))
    dice = 1 - ((2 * inter + smooth) / (union + smooth))

    return dice.mean()


class contrastive_loss(nn.Module):

    def __init__(self):
        super(contrastive_loss, self).__init__()
        self.loss_model = losses.NTXentLoss(temperature=0.07).cuda()

    def forward(self, feature, y_batch, prototypes, device=''):
        class_embs = None
        for cls_id in range(prototypes.shape[0]):
            mask = y_batch[:, cls_id + 1, :, :]
            mask = torch.stack([mask for _ in range(feature.shape[1])], dim=1)
            mask = F.interpolate(mask, size=64, mode="bilinear")
            class_emb = feature * mask
            class_emb = F.interpolate(class_emb, size=16, mode="bilinear")
            class_emb = class_emb.mean(1).reshape(-1, 16 * 16)

            if class_embs is None:
                class_embs = class_emb
            else:
                class_embs = torch.cat((class_embs, class_emb), dim=0)

        prototype_loss = self.loss_model(prototypes,
                                         torch.tensor(
                                             [i for i in range(1, (prototypes.size()[0] + 1))]).to(
                                             device), ref_emb=class_embs,
                                         ref_labels=torch.tensor([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]))

        return prototype_loss


def evaluation(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    dice = (2 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    iou = dice / (2 - dice)

    return iou, dice
