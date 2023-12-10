# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.modeling.roi_heads.relation_head.rel_proposal_network.models import (
    gt_rel_proposal_matching,
    RelationProposalModel,
    filter_rel_pairs,
)
from maskrcnn_benchmark.utils.visualize_graph import *
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator, make_roi_relation_contrast_loss_evaluator
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .sampling import make_roi_relation_samp_processor
from ..attribute_head.roi_attribute_feature_extractors import (
    make_roi_attribute_feature_extractor,
)
from ..box_head.roi_box_feature_extractors import (
    make_roi_box_feature_extractor,
    ResNet50Conv5ROIFeatureExtractor,
)
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_kern import (
    to_onehot,
)

class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()

        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True)
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels

        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)
        self.contrast_loss_evaluator = make_roi_relation_contrast_loss_evaluator(cfg)

        self.use_contrast_loss = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.USE_CONTRAST_LOSS
        self.object_cls_refine = cfg.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE
        self.pass_obj_recls_loss = cfg.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS

        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

    def forward(self, features, proposals, targets=None, logger=None, loss_weight=None, predicate_weight=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                    proposals, rel_labels, rel_pair_idxs, gt_rel_binarys_matrix = self.samp_processor.gtbox_relsample(proposals, targets)
                else:
                    proposals, rel_labels, rel_pair_idxs, gt_rel_binarys_matrix = self.samp_processor.detect_relsample(proposals, targets)
        else:
            rel_labels, gt_rel_binarys_matrix = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(
                features[0].device, proposals
            )

        if self.mode == "predcls":
            # overload the pred logits by the gt label
            device = features[0].device
            for proposal in proposals:
                obj_labels = proposal.get_field("labels")
                proposal.add_field("predict_logits", to_onehot(obj_labels, self.num_obj_cls))
                proposal.add_field("pred_scores", torch.ones(len(obj_labels)).to(device))
                proposal.add_field("pred_labels", obj_labels.to(device))

        # use box_head to extract features that will be fed to the later predictor processing
        roi_features = self.box_feature_extractor(features, proposals)
        #if isinstance(self.box_feature_extractor, ResNet50Conv5ROIFeatureExtractor):
        #    roi_features = self.box_feature_extractor.flatten_roi_features(roi_features)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)

        if self.use_union_box:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        else:
            union_features = None

        # final classifier that converts the features into predictions
        # should corresponding to all the functions and layers after the self.context class

        obj_refine_logits, relation_logits, add_losses = self.predictor(
            proposals,
            rel_pair_idxs,
            rel_labels,
            gt_rel_binarys_matrix,
            roi_features,
            union_features,
            loss_weight,
            predicate_weight,
            logger,
        )

        # proposals, rel_pair_idxs, rel_pn_labels,relness_net_input,roi_features,union_features, None
        # for test
        if not self.training:
            # re-NMS on refined object prediction logits
            if not self.object_cls_refine:
                # if don't use object classification refine, we just use the initial logits
                obj_refine_logits = [prop.get_field("predict_logits") for prop in proposals]

            result = self.post_processor(
                (relation_logits, obj_refine_logits), rel_pair_idxs, proposals
            )

            return roi_features, result, {}

        loss_relation, loss_refine = self.loss_evaluator(
            proposals, rel_labels, relation_logits, obj_refine_logits, predicate_weight
        )

        output_losses = dict()
        if self.cfg.MODEL.ATTRIBUTE_ON and isinstance(loss_refine, (list, tuple)):
            output_losses = dict(
                loss_rel=loss_relation,
                loss_refine_obj=loss_refine[0],
                loss_refine_att=loss_refine[1],
            )
        else:
            if not self.use_contrast_loss:
                if self.pass_obj_recls_loss:
                    output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine * 0.5)
                else:
                    output_losses = dict(loss_rel=loss_relation)
            else:
                if loss_weight is None:
                    loss_weight = 1
                if self.pass_obj_recls_loss:
                    output_losses = dict(loss_rel=loss_relation * loss_weight, loss_refine_obj=loss_refine) # 0.01
                else:
                    output_losses = dict(loss_rel=loss_relation * loss_weight)

        output_losses.update(add_losses)
        '''output_losses_checked = {}
        if self.training:
            for key in output_losses.keys():
                if output_losses[key] is not None:
                    if output_losses[key].grad_fn is not None:
                        output_losses_checked[key] = output_losses[key]
        output_losses = output_losses_checked'''
        return roi_features, proposals, output_losses


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
