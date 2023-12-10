import copy

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn_benchmark.modeling.make_layers import make_fc
from torch.autograd import Variable
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_msg_passing import (
    PairwiseFeatureExtractor,
)
from maskrcnn_benchmark.modeling.roi_heads.relation_head.rel_proposal_network.models import (
    make_relation_confidence_aware_module,
)
from maskrcnn_benchmark.structures.boxlist_ops import squeeze_tensor
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.data.datasets.visual_genome import get_frequency_distribution
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import (
    obj_edge_vectors,
    encode_box_info,
)
from maskrcnn_benchmark.modeling.roi_heads.relation_head.rel_proposal_network.loss import l2_norm
import os
import pickle
import numpy as np


class MessagePassingUnit(nn.Module):
    def __init__(self, input_dim, pooling_dim):
        super(MessagePassingUnit, self).__init__()
        self.ww = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            make_fc(input_dim * 2, pooling_dim),
            nn.ReLU(),
            make_fc(pooling_dim, 1)
        )
        # self.eps = 0.3
        # self.eps = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

    def forward(self, add_term, source_term, relness_gate=None):

        if add_term.size()[0] == 1 and source_term.size()[0] > 1:
            add_term = add_term.expand(source_term.size()[0], add_term.size()[1])
        if add_term.size()[0] > 1 and source_term.size()[0] == 1:
            source_term = source_term.expand(add_term.size()[0], source_term.size()[1])

        pair_feats = torch.cat([add_term, source_term], 1)
        # 学习每一条关系对应的边聚合的自适应阈值
        gate = torch.tanh(self.ww(pair_feats)).view(-1)
        if relness_gate is not None:
            gate = gate * relness_gate
        output = source_term * gate.view(-1, 1)

        '''if relness_gate is not None:
            output = source_term * relness_gate.view(-1, 1)
        else:
            output = source_term'''

        return output


class MessageFusion(nn.Module):
    def __init__(self, input_dim, dropout, eps):
        super(MessageFusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            make_fc(input_dim, input_dim),
        )
        self.whh = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            make_fc(input_dim, input_dim),
        )
        self.wrh = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            make_fc(input_dim, input_dim),
        )
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            make_fc(input_dim * 2, 1)
        )
        self.dropout = dropout
        self.eps = eps

    def forward(self, input, hidden, raw=None):
        if raw is not None:
            # output = self.whh(input + hidden) + self.wrh(raw)

            y = input + hidden
            row_norm = torch.norm(l2_norm(y) - l2_norm(raw), p=2, dim=1)
            row_norm = torch.clamp(row_norm, min=1e-7)
            score = torch.clamp(row_norm - self.eps, min=0)
            score = (score / row_norm).unsqueeze(1)
            output = (1 - score) * raw + score * y
            output = self.fusion(output)
        else:
            output = self.whh(input) + self.wrh(hidden)
        if self.dropout:
            output = F.dropout(output, training=self.training)
        return output


class LearnableRelatednessGating(nn.Module):
    def __init__(self):
        super(LearnableRelatednessGating, self).__init__()
        cfg_weight = cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.LEARNABLE_SCALING_WEIGHT
        self.alpha = nn.Parameter(torch.Tensor([cfg_weight[0]]), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([cfg_weight[1]]), requires_grad=False)

    def forward(self, relness):
        relness = torch.clamp(self.alpha * relness - self.alpha * self.beta, min=0, max=1.0)
        return relness


class FALayer(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(FALayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.drop = dropout
        self.dropout = nn.Dropout(0.5)
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.ReLU(),
            make_fc(2 * self.hidden_dim, 1)
        )

    def forward(self, inst_feature, norm_degree, aggregator_matrix, rel_pair_index):
        assert inst_feature.shape[0] == aggregator_matrix.shape[0]
        assert inst_feature.shape[0] == len(norm_degree)

        obj_index = rel_pair_index[:, 0]
        sub_index = rel_pair_index[:, 1]
        norm = norm_degree[obj_index] * norm_degree[sub_index]
        norm[norm > 10000] = 0

        new_obj_feature = torch.index_select(inst_feature, 0, obj_index)
        new_sub_feature = torch.index_select(inst_feature, 0, sub_index)
        new_inst_feature = torch.cat([new_sub_feature, new_obj_feature], dim=1)

        #g = torch.relu(self.gate(new_inst_feature)).squeeze()
        g = torch.tanh(self.gate(new_inst_feature)).squeeze()

        g = g * norm
        if self.drop:
            g = F.dropout(g, training=self.training)
        aggregator_factor = torch.zeros(
            (inst_feature.shape[0], inst_feature.shape[0]),
            dtype=inst_feature.dtype,
            device=inst_feature.device,
        )
        aggregator_factor[sub_index, obj_index] = g
        # aggregator_factor[sub_index, obj_index] = g

        aggregator_feature = torch.matmul(aggregator_factor, inst_feature)

        return aggregator_feature, g


class PISAContext(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels, # 4096
        hidden_dim=1024,
        num_iter=2,
        dropout=False,
        gate_width=128,
    ):
        super(PISAContext, self).__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.eps = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.SELF_FEATURE_PROPOTION
        self.layer_num = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.FREQUENCY_ADAPTION_LAYER_NUM
        self.message_prop_iter_num = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.MESSAGE_PROPOGATION_ITERATION_NUM
        self.message_passing_unit_num = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.MESSAGE_PASSING_UNIT_NUM
        self.valid_pair_num = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.MP_VALID_PAIRS_NUM
        self.relness_weighting_mp = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.RELNESS_MP_WEIGHTING
        self.pretrain_pre_clser_mode = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.PRETRAIN_MODE
        self.rel_aware_on = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.RELATION_CONFIDENCE_AWARE
        self.is_use_adaptive_confidence = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.USE_ADAPTIVE_CONFIDENCE

        self.pairwise_feature_extractor = PairwiseFeatureExtractor(cfg, in_channels)
        self.pooling_dim = 2048

        self.relness_score_recalibration = LearnableRelatednessGating()

        self.relation_conf_aware_models = make_relation_confidence_aware_module(
            self.hidden_dim,
        )

        self.obj_downdim_fc = nn.Sequential(
            nn.ReLU(),
            make_fc(self.pooling_dim, self.hidden_dim),
        )

        self.rel_downdim_fc = nn.Sequential(
            nn.ReLU(),
            make_fc(self.pooling_dim, self.hidden_dim),
        )

        # PSC (MPSC)
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"
        self.use_contrast_loss = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.USE_CONTRAST_LOSS

        self.num_obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_glove_prototype = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.USE_GLOVE_PROTOTYPE

        self.embed_dim = 300
        statistics = get_dataset_statistics(cfg)
        obj_classes, rel_classes = statistics["obj_classes"], statistics["rel_classes"]

        if self.use_glove_prototype:
            self.relation_embed = obj_edge_vectors(
                rel_classes, wv_dir=cfg.GLOVE_DIR, wv_dim=self.embed_dim
            )
        else:
            self.relation_embed = Variable(torch.randn((self.num_rel_classes, self.hidden_dim)).cuda())
            self.relation_embed.requires_grad = True
            self.relation_count = torch.zeros(self.num_rel_classes)

        self.obj_embed_vector = obj_edge_vectors(
            obj_classes, wv_dir=cfg.GLOVE_DIR, wv_dim=self.embed_dim
        )

        self.use_informative = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.IS_USE_INFORMATIVE
        self.use_informative_init = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.USE_INFORMATIVE_INIT
        self.predicate_frequency_dir = cfg.OUTPUT_DIR
        self.use_label_embed = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.USE_EMBED

        if self.use_informative:
            self.get_distribution = get_frequency_distribution(self.predicate_frequency_dir)
            self.frequency_distribution = self.min_max_norm(torch.tensor(self.get_distribution))
            if self.use_label_embed:
                self.pair_down_dim = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
                self.informative_gate = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim + self.embed_dim * 2, gate_width),
                    nn.ReLU(),
                    nn.Linear(gate_width, 1)
                )
            else:
                self.informative_weight = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.INFORMATIVE_WEIGHT
                self.informative_sem_embed = nn.Embedding(self.num_obj_classes * self.num_obj_classes, 1)
                self.informative_feature_gate = nn.Sequential(
                    make_fc(self.hidden_dim * 2, 1)
                )
                # self.informative_loss_eval = InformativeLoss(cfg)
                if self.use_informative_init:
                    fg_matrix = statistics['fg_matrix'][:, :, 1:].sum(-1).unsqueeze(-1)
                    self.fg_matrix_factor = statistics['fg_matrix'][:, :, :] * self.frequency_distribution
                    with open(os.path.join(cfg.OUTPUT_DIR, "fg_matrix.pkl"), "wb") as f:
                        pickle.dump(np.array(statistics['fg_matrix']), f)
                    self.fg_matrix_factor_sum = self.fg_matrix_factor.sum(-1).unsqueeze(-1) / (fg_matrix + 1e-3)
                    self.fg_matrix_final = self.fg_matrix_factor_sum.view(-1, 1)
                    with torch.no_grad():
                        self.informative_sem_embed.weight.copy_(self.fg_matrix_final, non_blocking=True)

        self.gate_obj2obj = nn.ModuleList()
        for i in range(self.layer_num):
            self.gate_obj2obj.append(FALayer(hidden_dim, dropout))

        self.gate_sub2pred = MessagePassingUnit(self.hidden_dim, gate_width)
        self.gate_obj2pred = MessagePassingUnit(self.hidden_dim, gate_width)
        self.gate_pred2sub = MessagePassingUnit(self.hidden_dim, gate_width)
        self.gate_pred2obj = MessagePassingUnit(self.hidden_dim, gate_width)

        self.proj_inst_fusion = nn.Sequential(
            make_fc(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            make_fc(self.hidden_dim, self.hidden_dim),
        )

        self.proj_rel_fusion = nn.Sequential(
            make_fc(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            make_fc(self.hidden_dim, self.hidden_dim),
        )


        #self.GRU_inst_fusion = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        #self.GRU_rel_fusion = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.inst_fusion = MessageFusion(self.hidden_dim, dropout, self.eps)

        self.inst_message_fusion = MessageFusion(self.hidden_dim, dropout, self.eps)
        self.rel_message_fusion = MessageFusion(self.hidden_dim, dropout, self.eps)

    def set_pretrain_pre_clser_mode(self, val=True):
        self.pretrain_pre_clser_mode = val

    def _prepare_adjacency_matrix(self, proposals, rel_pair_idxs, relatedness):
        rel_inds_batch_cat = []
        offset = 0
        num_proposals = [len(props) for props in proposals]
        rel_prop_pairs_relness_batch = []

        for idx, (prop, rel_ind_i) in enumerate(zip(proposals, rel_pair_idxs)):
            assert relatedness is not None
            related_matrix = relatedness[idx]
            rel_prop_pairs_relness = related_matrix[rel_ind_i[:, 0], rel_ind_i[:, 1]]
            det_score = prop.get_field("pred_scores")
            rel_prop_pairs_relness_batch.append(rel_prop_pairs_relness)
            rel_ind_i = copy.deepcopy(rel_ind_i)

            rel_ind_i += offset
            offset += len(prop)
            rel_inds_batch_cat.append(rel_ind_i)

        rel_inds_batch_cat = torch.cat(rel_inds_batch_cat, 0)

        subj_pred_map = (
            rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0]).fill_(0).float().detach()
        )
        obj_pred_map = (
            rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0]).fill_(0).float().detach()
        )

        if len(rel_prop_pairs_relness_batch) != 0:

            if self.rel_aware_on:
                offset = 0
                rel_prop_pairs_relness_sorted_idx = []
                rel_prop_pairs_relness_batch_update = []
                for idx, each_img_relness in enumerate(rel_prop_pairs_relness_batch):

                    (
                        selected_rel_prop_pairs_relness,
                        selected_rel_prop_pairs_idx,
                    ) = torch.sort(each_img_relness, descending=True)

                    selected_rel_prop_pairs_idx = selected_rel_prop_pairs_idx[: self.valid_pair_num]

                    if self.relness_weighting_mp and not self.pretrain_pre_clser_mode:
                        each_img_relness = self.relness_score_recalibration(each_img_relness)
                        selected_rel_prop_pairs_idx = squeeze_tensor(torch.nonzero(each_img_relness > 0.001))

                    rel_prop_pairs_relness_batch_update.append(each_img_relness)

                    rel_prop_pairs_relness_sorted_idx.append(
                        selected_rel_prop_pairs_idx + offset
                    )
                    offset += len(each_img_relness)

                selected_rel_prop_pairs_idx = torch.cat(rel_prop_pairs_relness_sorted_idx, 0)
                rel_prop_pairs_relness_batch_cat = torch.cat(
                    rel_prop_pairs_relness_batch_update, 0
                )

            subj_pred_map[
                rel_inds_batch_cat[selected_rel_prop_pairs_idx, 0],
                selected_rel_prop_pairs_idx,
            ] = 1
            obj_pred_map[
                rel_inds_batch_cat[selected_rel_prop_pairs_idx, 1],
                selected_rel_prop_pairs_idx,
            ] = 1
            selected_relness = rel_prop_pairs_relness_batch_cat
        else:
            # or all relationship pairs
            selected_rel_prop_pairs_idx = torch.arange(
                len(rel_inds_batch_cat[:, 0]), device=rel_inds_batch_cat.device
            )
            selected_relness = None
            subj_pred_map.scatter_(0, (rel_inds_batch_cat[:, 0].contiguous().view(1, -1)), 1)
            obj_pred_map.scatter_(0, (rel_inds_batch_cat[:, 1].contiguous().view(1, -1)), 1)
        return (
            rel_inds_batch_cat,
            subj_pred_map,
            obj_pred_map,
            selected_relness,
            selected_rel_prop_pairs_idx,
        )

    def message_pred2inst(self, inst_feats, rel_feats, subj_pred_map, gate_module, relness_scores=None):
        if subj_pred_map.sum() == 0:
            feature_data = torch.zeros(
                (inst_feats.shape[0], inst_feats.shape[1]),
                requires_grad=True,
                dtype=inst_feats.dtype,
                device=inst_feats.device,
            )
        else:
            transfer_list = (subj_pred_map > 0).nonzero()
            rel_index = transfer_list[:, 1]
            inst_index = transfer_list[:, 0]
            rel_feature = torch.index_select(rel_feats, 0, rel_index)
            inst_feature = torch.index_select(inst_feats, 0, inst_index)


            if relness_scores is not None:
                select_relness = relness_scores[rel_index]
            else:
                select_relness = None
            transferred_features = gate_module(
                inst_feature, rel_feature, select_relness
            )
            aggregator_matrix = torch.zeros(
                (inst_feats.shape[0], transferred_features.shape[0]),
                dtype=transferred_features.dtype,
                device=transferred_features.device,
            )

            for f_id in range(inst_feats.shape[0]):
                if subj_pred_map[f_id, :].data.sum() > 0:
                    feature_index = squeeze_tensor(
                        (transfer_list[:, 0] == f_id).nonzero()
                    )
                    aggregator_matrix[f_id, feature_index] = 1
            aggregate_feat = torch.matmul(aggregator_matrix, transferred_features)
            avg_factor = aggregator_matrix.sum(dim=1)
            vaild_aggregate_idx = avg_factor != 0
            avg_factor = avg_factor.unsqueeze(1).expand(
                avg_factor.shape[0], aggregate_feat.shape[1]
            )
            aggregate_feat[vaild_aggregate_idx] /= avg_factor[vaild_aggregate_idx]

            feature_data = aggregate_feat

        return feature_data

    def message_inst2inst(self, obj_feats, gate_module, rel_pair_index, inst_raw_feature, relness_scores=None):

        aggregator_matrix = torch.zeros(
            (obj_feats.shape[0], obj_feats.shape[0]),
            dtype=obj_feats.dtype,
            device=obj_feats.device,
        )

        if relness_scores is not None:
            select_rel_pair_index = rel_pair_index[relness_scores > 0.001]
        else:
            select_rel_pair_index = rel_pair_index
        obj_list = select_rel_pair_index[:, 0]
        sub_list = select_rel_pair_index[:, 1]
        # aggregator_matrix[obj_list, sub_list] = 1
        aggregator_matrix[sub_list, obj_list] = 1

        degree = aggregator_matrix.sum(dim=1)
        norm_degree = torch.pow(degree, -0.5)

        for i in range(self.layer_num):
            aggregate_feats, g = gate_module[i](obj_feats, norm_degree, aggregator_matrix, select_rel_pair_index)
            # eps = torch.cosine_similarity(raw_feature, obj_feats, dim=1)
            obj_feats = self.inst_fusion(self.eps * obj_feats, aggregate_feats, inst_raw_feature)

        # feature_data = self.inst_fusion(raw_feature, obj_feats)

        '''if relness_scores is not None and not self.pretrain_pre_clser_mode:
            pred_finetune_relatedness = torch.zeros(
                len(rel_pair_index),
                dtype=g.dtype,
                device=g.device)
            pred_finetune_relatedness[relness_scores > 0.0001] = g
        else:'''
        pred_finetune_relatedness = g

        return obj_feats, pred_finetune_relatedness

    def get_distance(self, factor, distribution):
        new_distribution = []
        for i in range(len(factor)):
            new_distribution.append(2 - (distribution - factor[i])**2)
        new_distribution = torch.stack(new_distribution, dim=0)
        return new_distribution

    def min_max_norm(self, data):
        return (data - min(data)) / max(data)

    def informative_predictor(self, inst_proposals, num_inst_list, inst_features, rel_pair_idxs):

        if self.use_label_embed:
            self.obj_embed_vector = self.obj_embed_vector.to(inst_features.device)
            inst_features = inst_features.split(num_inst_list, dim=0)

            pair_features = []
            pair_embed_vectors = []

            if self.mode == 'predcls':
                inst_pred_labels = torch.cat([each_prop.get_field("labels").detach() for each_prop in inst_proposals],
                                             dim=0).long().split(num_inst_list, dim=0)

                for pair_idx, inst_pred, inst_feature in zip(rel_pair_idxs, inst_pred_labels, inst_features):
                    pair_embed_vectors.append(torch.cat(
                        (self.obj_embed_vector[inst_pred[pair_idx[:, 0]]],
                         self.obj_embed_vector[inst_pred[pair_idx[:, 1]]]),
                        dim=1))
                    pair_features.append(torch.cat((inst_feature[pair_idx[:, 0]], inst_feature[pair_idx[:, 1]]), dim=1))
            else:
                inst_pred_logits = torch.cat(
                    [each_prop.get_field("predict_logits").detach() for each_prop in inst_proposals],
                    dim=0).split(num_inst_list, dim=0)
                for pair_idx, inst_logits, inst_feature in zip(rel_pair_idxs, inst_pred_logits, inst_features):
                    inst_embed_vectors = F.softmax(inst_logits, dim=1) @ self.obj_embed_vector
                    pair_embed_vectors.append(
                        torch.cat((inst_embed_vectors[pair_idx[:, 0]], inst_embed_vectors[pair_idx[:, 1]]), dim=1))
                    pair_features.append(torch.cat((inst_feature[pair_idx[:, 0]], inst_feature[pair_idx[:, 1]]), dim=1))

            pair_label = torch.cat(pair_embed_vectors, dim=0).long()
            pair_feature = torch.cat(pair_features, dim=0)
            small_pair_feature = self.pair_down_dim(pair_feature)
            pair_information = torch.cat((pair_label, small_pair_feature), dim=1)
            informative_factor = torch.sigmoid(self.informative_gate(pair_information))
            distance = self.get_distance(informative_factor.squeeze(),
                                         self.frequency_distribution.to(dtype=informative_factor.dtype,
                                                                        device=informative_factor.device))
        else:
            if self.mode == 'predcls':
                inst_pred_labels = torch.cat([each_prop.get_field("labels").detach() for each_prop in inst_proposals],
                                             dim=0).split(num_inst_list, dim=0)
            else:
                inst_pred_labels = torch.cat(
                    [each_prop.get_field("pred_labels").detach() for each_prop in inst_proposals],
                    dim=0).split(num_inst_list, dim=0)
            inst_features = inst_features.split(num_inst_list, dim=0)

            pair_labels = []
            pair_features = []
            for pair_idx, inst_pred, inst_feature in zip(rel_pair_idxs, inst_pred_labels, inst_features):
                pair_labels.append(torch.stack((inst_pred[pair_idx[:, 0]], inst_pred[pair_idx[:, 1]]), dim=1))
                pair_features.append(torch.cat((inst_feature[pair_idx[:, 0]], inst_feature[pair_idx[:, 1]]), dim=1))

            pair_label = torch.cat(pair_labels, dim=0).long()
            pair_feature = torch.cat(pair_features, dim=0)

            informative_feature = torch.sigmoid(self.informative_feature_gate(pair_feature))

            informative_label = torch.sigmoid(
                self.informative_sem_embed(pair_label[:, 0] * self.num_obj_classes + pair_label[:, 1]))

            informative_factor = self.informative_weight * informative_feature + (
                    1 - self.informative_weight) * informative_label

            distance = self.get_distance(informative_factor.squeeze(),
                                         self.frequency_distribution.to(dtype=informative_factor.dtype,
                                                                        device=informative_factor.device))

        return informative_factor, distance

    def forward(
        self,
        inst_features,
        rel_union_features,
        proposals,
        rel_pair_index,
        rel_labels=None,
        logger=None,
    ):
        # augment_obj_feat : through fusion predict labels and bbox embedding feature to augment instance feature
        # rel_feat : add the object-pair feature to the original relation union feature

        augment_inst_feat, rel_feats = self.pairwise_feature_extractor(inst_features, rel_union_features, proposals, rel_pair_index)

        # get the raw feature from the detection model and fusion operation
        inst_raw_feature = self.obj_downdim_fc(augment_inst_feat)
        rel_raw_feature = self.rel_downdim_fc(rel_feats)

        rel_feats_iters = [rel_raw_feature]
        inst_feats_iters = [inst_raw_feature]
        pred_relatedness_score_iters = []

        num_rel_list = [len(pair) for pair in rel_pair_index]
        num_inst_list = [len(prop) for prop in proposals]

        for iter_num in range(self.message_prop_iter_num):
            input_inst_feature = inst_feats_iters[iter_num]
            input_rel_feature = rel_feats_iters[iter_num]
            _, pred_relatedness_score = self.relation_conf_aware_models(input_rel_feature, input_inst_feature, proposals,
                                                                                      rel_pair_index)

            batchwise_rel_pair_index, subj_pred_map, obj_pred_map, relness_scores, selected_rel_prop_pairs_idx = self._prepare_adjacency_matrix(
                proposals, rel_pair_index, pred_relatedness_score)

            # update instance feature from instance
            inst_feats_update, pred_finetune_relatedness = self.message_inst2inst(inst_feats_iters[iter_num],
                                                                                  self.gate_obj2obj,
                                                                                  batchwise_rel_pair_index,
                                                                                  inst_raw_feature,
                                                                                  relness_scores=relness_scores)

            '''if self.is_use_adaptive_confidence:
                relness_scores = relness_scores + pred_finetune_relatedness'''

            # pred_relatedness_score_iters.append(pred_relatedness_score)
            pred_relatedness_score_iters.append(relness_scores)

            inst_mp_iters = [inst_feats_update]
            rel_mp_iters = [rel_feats_iters[-1]]

            valid_inst_index = []
            for p in proposals:
                valid_inst_index.append(p.get_field("pred_scores") > 0.03)

            if len(valid_inst_index) > 0:
                valid_inst_index = torch.cat(valid_inst_index, 0)
            else:
                valid_inst_index = torch.zeros(0)

            '''if self.pretrain_pre_clser_mode:
                refine_inst_feature = inst_mp_iters[-1]
                refine_rel_feature = rel_mp_iters[-1]
                inst_feats_iters.append(refine_inst_feature)
                rel_feats_iters.append(refine_rel_feature)
                continue'''
            if (len(squeeze_tensor(valid_inst_index.nonzero())) < 2 or len(
                    squeeze_tensor(batchwise_rel_pair_index.nonzero())) < 1 or len(
                squeeze_tensor(subj_pred_map.nonzero())) < 1 or len(
                squeeze_tensor(obj_pred_map.nonzero())) < 1):
                refine_inst_feature = inst_mp_iters[-1]
                refine_rel_feature = rel_mp_iters[-1]
                inst_feats_iters.append(refine_inst_feature)
                rel_feats_iters.append(refine_rel_feature)
                continue

            for mp_iter in range(self.message_passing_unit_num):
                # update relation feature from instance
                add_rel2sub_feature = self.message_pred2inst(
                    inst_mp_iters[mp_iter],
                    rel_mp_iters[mp_iter],
                    subj_pred_map,
                    self.gate_pred2sub,
                    relness_scores=relness_scores,
                )
                add_rel2obj_feature = self.message_pred2inst(
                    inst_mp_iters[mp_iter],
                    rel_mp_iters[mp_iter],
                    obj_pred_map,
                    self.gate_pred2obj,
                    relness_scores=relness_scores,
                )
                rel_fusion_feature = self.proj_rel_fusion(add_rel2sub_feature + add_rel2obj_feature)
                inst_refine_feature = self.inst_message_fusion(self.eps * inst_mp_iters[mp_iter], rel_fusion_feature, inst_raw_feature)
                # inst_refine_feature = self.inst_message_fusion(inst_mp_iters[mp_iter], rel_fusion_feature)
                inst_mp_iters.append(inst_refine_feature)

                # update instance feature from relation
                sub_all = batchwise_rel_pair_index[:, 0]
                obj_all = batchwise_rel_pair_index[:, 1]
                sub_valid = valid_inst_index[sub_all]
                obj_valid = valid_inst_index[obj_all]
                pair_valid = sub_valid & obj_valid

                sub_valid = sub_all[pair_valid]
                obj_valid = obj_all[pair_valid]

                valid_sub_feature = torch.index_select(inst_mp_iters[mp_iter], 0, sub_valid)
                valid_obj_feature = torch.index_select(inst_mp_iters[mp_iter], 0, obj_valid)
                valid_rel_feature = torch.index_select(rel_mp_iters[mp_iter], 0, squeeze_tensor(pair_valid.nonzero()))
                valid_raw_rel_feature = torch.index_select(rel_raw_feature, 0, squeeze_tensor(pair_valid.nonzero()))

                add_sub2rel_feature = self.gate_sub2pred(valid_rel_feature, valid_sub_feature)
                add_obj2rel_feature = self.gate_obj2pred(valid_rel_feature, valid_obj_feature)
                inst_fusion_feature = self.proj_inst_fusion((add_sub2rel_feature + add_obj2rel_feature) / 2.0)
                valid_rel_refine_feature = self.rel_message_fusion(self.eps * valid_rel_feature, inst_fusion_feature, valid_raw_rel_feature)
                # valid_rel_refine_feature = self.rel_message_fusion(valid_rel_feature, inst_fusion_feature)
                temp_rel_feature = rel_mp_iters[mp_iter].clone()
                temp_rel_feature[pair_valid] = valid_rel_refine_feature
                rel_mp_iters.append(temp_rel_feature)

            refine_inst_feature = inst_mp_iters[-1]
            refine_rel_feature = rel_mp_iters[-1]
            inst_feats_iters.append(refine_inst_feature)
            rel_feats_iters.append(refine_rel_feature)

        inst_feats_final = inst_feats_iters[-1]
        rel_feats_final = rel_feats_iters[-1]

        if self.use_informative:
            informative_factor, refine_distance = self.informative_predictor(proposals, num_inst_list, inst_feats_final, rel_pair_index)
        else:
            informative_factor, refine_distance = None, None

        if self.use_contrast_loss and self.training:
            if not self.use_glove_prototype:
                rel_labels = torch.cat(rel_labels, dim=0).long()
                for rel_index in range(rel_feats_final.shape[0]):
                    if rel_labels[rel_index] > 0:
                        self.relation_embed.data[rel_labels[rel_index]] = (self.relation_embed.data[
                                                                               rel_labels[rel_index]] *
                                                                           self.relation_count.data[rel_labels[
                                                                               rel_index]] + rel_feats_final.data[
                                                                                             rel_index, :].detach()) / (
                                                                                  self.relation_count.data[
                                                                                      rel_labels[rel_index]] + 1)
                        self.relation_count[rel_labels[rel_index]] += 1

        return (inst_feats_final, rel_feats_final, self.relation_embed.to(inst_feats_final.device),
                pred_relatedness_score_iters[-1], refine_distance, informative_factor)


# pairwise informative(frequency and informative) self-adaption model
def build_pisa_model(cfg, in_channels):
    return PISAContext(cfg, in_channels)