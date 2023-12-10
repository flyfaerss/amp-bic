import numpy as np
import torch
import torch.nn as nn


class PenaltyModule(nn.Module):
    def __init__(self, head_config, statistics):
        super(PenaltyModule, self).__init__()
        self.penalty_threshold = head_config.bias_module.penalty_threshold
        self.penalty_weight = head_config.bias_module.penalty_weight
        self.scale_weight = head_config.bias_module.scale_weight
        self.penalty_type = head_config.bias_module.penalty_type
        self.fusion_weight = head_config.bias_module.fusion_weight
        self.use_curriculum = head_config.bias_module.use_curriculum

        self.eval_with_penalty = head_config.bias_module.eval_with_penalty

        self.penalty_k = head_config.bias_module.penalty_k
        self.eps = head_config.bias_module.penalty_epsilon

        # default value for psb bias
        self.psb_default_value = head_config.bias_module.possible_bias_default_value
        if self.psb_default_value <= 0:
            self.psb_default_value = self.eps
        # default value for bg rel
        self.bg_default_value = head_config.bias_module.bg_default_value
        if self.bg_default_value <= 0:
            self.bg_default_value = self.eps
        self.psb_threshold = head_config.bias_module.possible_bias_threshold
        self.thing_stuff_apart = head_config.bias_module.thing_stuff_apart

        self.fg_matrix = statistics['freq_matrix'].detach().clone()
        self.fg_matrix[:, :, 0] = 0

        # penalty
        self.penalty_bias = None
        self.penalty_bias = self.cal_penalty(self.fg_matrix)

        self.num_thing_class = head_config.num_thing_class
        self.num_stuff_class = head_config.num_stuff_class

        if self.thing_stuff_apart:
            self.thing_thing_penalty = self.cal_penalty(self.fg_matrix[1:self.num_thing_class + 1, 1:self.num_thing_class + 1, :])
            self.stuff_stuff_penalty = self.cal_penalty(self.fg_matrix[self.num_thing_class + 1:, self.num_thing_class + 1:, :])
            self.thing_stuff_penalty = self.cal_penalty(self.fg_matrix[1:self.num_thing_class + 1, self.num_thing_class + 1:, :])
            self.stuff_thing_penalty = self.cal_penalty(self.fg_matrix[self.num_thing_class + 1:, 1:self.num_thing_class + 1, :])
            self.apart_penalty = torch.stack([
                self.thing_thing_penalty,
                self.thing_stuff_penalty,
                self.stuff_thing_penalty,
                self.stuff_stuff_penalty,
            ], dim=0)

    def cal_penalty(self, freq_matrix):
        counts = freq_matrix.sum((0, 1))
        dist = counts / counts.sum()
        dist[0] = 1
        dist = dist.pow(self.scale_weight)
        dist = dist / (dist.sum() - 1)
        penalty_bias = torch.log(dist + self.eps)
        penalty_bias[0] = np.log(self.bg_default_value)
        return penalty_bias

    def get_apart_penalty_bias(self, obj_pair_label, pred_dist):
        activation_onehot = torch.zeros((obj_pair_label.shape[0], 4)).to(pred_dist)
        # thing-thing: 0
        # thing-stuff: 1
        # stuff-thing: 2
        # stuff-stuff: 3
        activation_index = (obj_pair_label[:, 0] > self.num_thing_class) * 2 + (obj_pair_label[:, 1] > self.num_thing_class)
        squence_index = torch.arange(obj_pair_label.shape[0])
        activation_onehot[squence_index, activation_index] = 1.0
        return torch.mm(activation_onehot, self.apart_penalty.to(pred_dist))

    def penalty(self, pred_dist, gt=None, obj_pair_label=None, epoch=None, max_epochs=None):
        if not self.training and self.eval_with_penalty:
            self.fusion_weight = 0.5
        elif self.use_curriculum:
            self.fusion_weight = 2.0 - epoch / max_epochs

        if self.thing_stuff_apart:
            resistance_bias = None
            if self.penalty_type == 'count_bias':
                if pred_dist is not None:
                    resistance_bias = self.get_apart_penalty_bias(obj_pair_label, pred_dist)
                else:
                    pred_dist = torch.zeros((self.penalty_bias.shape[0]),
                                            dtype=torch.float32,
                                            device=obj_pair_label.device)
                    resistance_bias = self.get_apart_penalty_bias(obj_pair_label, pred_dist)
            else:
                raise Exception('unknown penalty type {}'.format(self.penalty_type))
        else:
            resistance_bias = None
            if self.penalty_type == 'count_bias':
                if pred_dist is not None:
                    resistance_bias = self.penalty_bias.to(pred_dist)
                else:
                    pred_dist = torch.zeros(self.penalty_bias)
                    resistance_bias = self.penalty_bias
            else:
                raise Exception('unknown penalty type {}'.format(self.penalty_type))

        return pred_dist + resistance_bias * self.fusion_weight, resistance_bias

    def forward(self, pred_dist, gt=None, obj_pair=None, epoch=None, max_epochs=None):
        if self.training or self.eval_with_penalty:
            return self.penalty(pred_dist, gt, obj_pair, epoch, max_epochs)
        return pred_dist


class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, cfg, statistics):
        super(FrequencyBias, self).__init__()

        pred_dist = statistics['pred_dist'].float()
        assert pred_dist.size(0) == pred_dist.size(1)
        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)
        pred_dist = pred_dist.view(-1, self.num_rels)

        self.obj_baseline = nn.Embedding(self.num_objs * self.num_objs, self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, obj_pair_labels):
        """
        :param obj_pair_labels: [batch_size, 2]
        :return:
        """
        pair_idx = obj_pair_labels[:, 0] * self.num_objs + obj_pair_labels[:, 1]
        pred_dist = self.obj_baseline(pair_idx.long())
        return pred_dist

    def index_with_probability(self, pair_prob):
        """
        :param labels: [batch_size, num_obj, 2]
        :return:
        """
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:, :, 0].contiguous().view(batch_size, num_obj, 1) * pair_prob[:, :,
                                                                                    1].contiguous().view(batch_size, 1,
                                                                                                         num_obj)

        pred_dist = joint_prob.view(batch_size, num_obj * num_obj) @ self.obj_baseline.weight

        return pred_dist

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)


class BiasModule(nn.Module):
    def __init__(self, head_config, statistics):
        super(BiasModule, self).__init__()
        self.use_penalty = head_config.bias_module.use_penalty
        # #### init post operation
        if self.use_penalty:
            self.penalty_module = PenaltyModule(head_config, statistics)

    def before(self, x):
        return x

    def post(self, bias, gt=None, obj_pair=None, epoch=None, max_epochs=None):
        if self.use_penalty:
            bias, resistance_bias = self.penalty_module(bias, gt=gt, obj_pair=obj_pair, epoch=epoch, max_epochs=max_epochs)
        return bias, resistance_bias

    def forward(self, gt=None, *args, **kwargs):
        bias = None
        bias = self.post(bias, gt=gt)
        return bias


class FreqBiasModule(BiasModule):
    def __init__(self, cfg, statistics):
        super(FreqBiasModule, self).__init__(cfg, statistics)
        self.bias_module = FrequencyBias(cfg, statistics)

    def index_with_labels(self, obj_pair_labels, gt=None, epoch=None, max_epochs=None):
        bias = self.bias_module.index_with_labels(obj_pair_labels)
        # bias = None
        new_bias, resistance_bias = self.post(bias, gt=gt, obj_pair=obj_pair_labels, epoch=epoch, max_epochs=max_epochs)
        return new_bias, bias, resistance_bias

    def index_with_probability(self, pair_prob, gt=None):
        bias = self.bias_module.index_with_probability(pair_prob)
        bias = self.post(bias, gt=gt)
        return bias

    def forward(self, obj_pair_labels=None, gt=None, obj_pair=None, *args, **kwargs):
        return self.index_with_labels(obj_pair_labels=obj_pair_labels, gt=gt)

