import math

import ipdb
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.boxlist_ops import squeeze_tensor


class FocalLoss(nn.Module):
    def __init__(
        self, alpha=1.0, gamma=2.0, logits=False, reduce=True, ignored_label_idx=None
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.ignored_label_idx = ignored_label_idx

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1.0 - pt) ** self.gamma * BCE_loss
        if self.ignored_label_idx:
            F_loss = F_loss[targets != self.ignored_label_idx]

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalLossMultiTemplate(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, logits=True):
        super(FocalLossMultiTemplate, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.focal_loss = FocalLoss(alpha, gamma, logits, reduce=False)

    def forward(self, inputs, targets):

        loss = self.focal_loss(inputs, targets).sum(-1).mean(-1)

        return loss


class FocalLossFGBGNormalization(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, logits=True):
        super(FocalLossFGBGNormalization, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_loss = FocalLoss(alpha, gamma, logits, reduce=False)



    def forward(self, inputs, targets, reduce=True):
        loss = self.focal_loss(inputs, targets)
        
        loss = loss.sum(-1)
        loss /= (len(torch.nonzero(targets)) + 1)

        return loss.mean(-1)



class WrappedBCELoss(nn.Module):
    def __init__(self):
        super(WrappedBCELoss, self).__init__()
        self.loss = F.binary_cross_entropy_with_logits

    def forward(self, inputs, targets, reduce=True):
        return self.loss(inputs, targets)

def loss_eval_bincls_single_level(pre_cls_logits, rel_labels, loss):

    bin_logits = pre_cls_logits[rel_labels != -1]

    selected_labels = rel_labels[rel_labels != -1].long()

    onehot = torch.zeros_like(bin_logits)
    onehot[selected_labels > 0] = 1
    loss_val = loss[0](inputs=bin_logits, targets=onehot, reduce=True)

    return loss_val


def loss_eval_hybrid_level(pre_cls_logits, rel_labels, loss):
    selected_cls_logits = pre_cls_logits[rel_labels != -1]

    mulitlabel_logits = selected_cls_logits[:, 1:]
    bin_logits = selected_cls_logits[:, 0]

    selected_labels = rel_labels[rel_labels != -1].long()

    onehot = torch.zeros_like(mulitlabel_logits)
    selected_fg_idx = squeeze_tensor(torch.nonzero(selected_labels > 0))
    onehot[selected_fg_idx, selected_labels[selected_fg_idx] - 1] = 1

    loss_val_mulabel = loss[0](inputs=mulitlabel_logits, targets=onehot, reduce=True)

    onehot = torch.zeros_like(bin_logits)
    onehot[selected_labels > 0] = 1
    loss_val_bin = loss[1](inputs=bin_logits, targets=onehot, reduce=True)

    # return loss_val_bin
    return loss_val_bin * 0.8 + loss_val_mulabel * 0.2


class RelAwareLoss(nn.Module):
    def __init__(self, cfg):
        super(RelAwareLoss, self).__init__()
        alpha = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.FOCAL_LOSS_ALPHA
        gamma = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.FOCAL_LOSS_GAMMA

        self.pre_clser_loss_type = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRE_CLSER_LOSS
        self.predictor_type = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.REL_AWARE_PREDICTOR_TYPE

        if "focal" in self.pre_clser_loss_type:
            self.loss_module = (
                FocalLossFGBGNormalization(alpha, gamma),
                FocalLossFGBGNormalization(alpha, gamma),
            )
        elif "bce" in self.pre_clser_loss_type:
            self.loss_module = (
                WrappedBCELoss(),
                WrappedBCELoss(),
            )

    def forward(self, pred_logit, rel_labels):
        if "focal" in self.pre_clser_loss_type:
            return loss_eval_hybrid_level(pred_logit, rel_labels, self.loss_module)

        if 'bce' in self.pre_clser_loss_type:
            return  loss_eval_bincls_single_level(pred_logit[:, 0], rel_labels, self.loss_module)


def l2_norm(feature, axis=1):
    norm = torch.norm(feature,2,axis,True)
    output = torch.div(feature, norm)
    return output


def loss_eval_feature(feature, labels, embed, weight, tau=0.1):
    feature = l2_norm(feature, axis=1)
    embed = l2_norm(embed, axis=1)
    fg_labels = squeeze_tensor(torch.nonzero(labels[labels != -1]))
    valid_feature = feature[fg_labels]
    valid_labels = labels[fg_labels].long()
    if weight is not None:
        valid_weight = weight[valid_labels]

    labels_list = []
    for i in range(valid_labels.shape[0]):
        if valid_labels[i] not in labels_list:
            labels_list.append(valid_labels[i])
    labels_list = torch.tensor(labels_list).squeeze().long().to(feature.device)

    match_inner_product = torch.exp(torch.mul(valid_feature, embed[valid_labels]).sum(-1) / tau)
    all_inner_product = torch.exp(torch.mul(valid_feature.unsqueeze(1), embed[labels_list]).sum(-1) / tau).sum(-1)
    # loss = torch.mean(-torch.log(match_inner_product / (all_inner_product - match_inner_product)))
    if weight is not None:
        loss = torch.mean(-torch.log(match_inner_product / all_inner_product) * valid_weight)
    else:
        loss = torch.mean(-torch.log(match_inner_product / all_inner_product))

    return loss


def loss_eval_relness(relness, rel_labels, loss):

    selected_relness = relness[rel_labels != -1]
    selected_labels = rel_labels[rel_labels != -1]

    onehot = torch.zeros_like(selected_relness)
    onehot[selected_labels > 0] = 1

    loss_val = loss(selected_relness, onehot)

    return loss_val


class RelnessLoss(nn.Module):
    def __init__(self, cfg):
        super(RelnessLoss, self).__init__()
        self.loss_module = nn.BCELoss()

    def forward(self, relness, rel_labels):
        return loss_eval_relness(relness, rel_labels, self.loss_module)


class FeatureLoss(nn.Module):
    def __init__(self, cfg):
        super(FeatureLoss, self).__init__()
        self.tau = 0.1

    def forward(self, feature, labels, embed, weight):
        return loss_eval_feature(feature, labels, embed, weight, self.tau)


def loss_eval_informative(informative, rel_labels, distribution, loss):
    selected_informative = informative[rel_labels > 0]
    selected_labels = rel_labels[rel_labels > 0].long()

    labels = distribution[selected_labels]

    loss_val = loss(input=selected_informative, target=labels)

    return loss_val


class InformativeLoss(nn.Module):
    def __init__(self, cfg):
        super(InformativeLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, informative, labels, distribution):
        return loss_eval_informative(informative, labels, distribution, self.loss)
