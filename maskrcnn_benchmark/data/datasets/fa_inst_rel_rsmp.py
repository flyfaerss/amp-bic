import json
import os
from collections import OrderedDict, Counter
from typing import Dict
import torch
import numpy as np
import pickle
import math

from maskrcnn_benchmark.config import cfg

'''def resampling_dict_generation(dataset, predicate_list, instance_list,  logger):

    logger.info("using resampling method:" + dataset.resampling_method)
    repeat_dict_dir = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_DICT_DIR
    curr_dir_repeat_dict = os.path.join(cfg.OUTPUT_DIR, "repeat_dict.pkl")
    predicate_dir = os.path.join(cfg.OUTPUT_DIR, f'rel_informative.txt')
    if repeat_dict_dir is not None and repeat_dict_dir != "" or os.path.exists(curr_dir_repeat_dict):
        if os.path.exists(curr_dir_repeat_dict):
            repeat_dict_dir = curr_dir_repeat_dict

        logger.info("load repeat_dict from " + repeat_dict_dir)
        with open(repeat_dict_dir, 'rb') as f:
            repeat_dict = pickle.load(f)

        return repeat_dict

    else:
        logger.info(
            "generate the repeat dict by balancing instance frequency and predicate frequency")

        if dataset.resampling_method == "faip":
            # when we use the lvis sampling method,
            #global_rf = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR
            #logger.info(f"global repeat factor: {global_rf};  ")
            pass
        else:
            raise NotImplementedError(dataset.resampling_method)

        predicate_times = np.zeros(len(predicate_list))
        object_times = np.zeros(len(instance_list))
        subject_times = np.zeros(len(instance_list))
        for i in range(len(dataset)):
            anno = dataset.get_groundtruth(i)
            tgt_rel_matrix = anno.get_field('relation')
            tgt_inst_labels = anno.get_field('labels')
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labels = tgt_rel_matrix[tgt_head_idxs,
                                          tgt_tail_idxs].contiguous().view(-1)
            tgt_sub_labels = tgt_inst_labels[tgt_head_idxs].contiguous().view(-1)
            tgt_obj_labels = tgt_inst_labels[tgt_tail_idxs].contiguous().view(-1)

            for each_rel in tgt_rel_labels:
                predicate_times[each_rel] += 1
            for each_sub, each_obj in zip(tgt_sub_labels, tgt_obj_labels):
                subject_times[each_sub] += 1
                object_times[each_obj] += 1

        instance_times = subject_times + object_times
        predicate_total = sum(predicate_times)
        subject_total = sum(subject_times)
        object_total = sum(object_times)
        instance_total = subject_total + object_total
        predicate_frequency = predicate_times[1:] / (predicate_total + 1e-11)
        instance_frequency = instance_times[1:] / (instance_total + 1e-11)
        padding = np.ones((len(instance_list) - 1))
        subject_frequency = (subject_times[1:] + padding) / (subject_total + 1e-11)
        object_frequency = (object_times[1:] + padding) / (object_total + 1e-11)

        def min_max_norm(data):
            return (data - np.min(data)) / np.max(data)

        predicate_informative = -1 * np.log(predicate_frequency)

        mean_predicate = np.mean(predicate_informative)
        predicate_sum = sum([1 / n for n in predicate_informative]) / 3
        temp = np.sort(predicate_informative)
        count = 0
        for e in temp:
            count += 1 / e
            if count > predicate_sum:
                mean_predicate = e
                break

        instance_distribution = -1 * np.log(instance_frequency)
        subject_distribution = -1 * np.log(subject_frequency)
        object_distribution = -1 * np.log(object_frequency)
        mean_object = np.mean(object_distribution)
        mean_subject = np.mean(subject_distribution)
        mean_instance = np.mean(instance_distribution)

        sub_factor = []
        for sub_label in range(len(instance_distribution)):
            if subject_distribution[sub_label] > mean_subject:
                sub_factor.append((subject_distribution[sub_label] / min(subject_distribution)) * (
                            subject_distribution[sub_label] / mean_subject))
            else:
                sub_factor.append(pow(subject_distribution[sub_label] / mean_subject, 1))

        obj_factor = []
        for obj_label in range(len(instance_distribution)):
            if object_distribution[obj_label] > mean_object:
                obj_factor.append((object_distribution[obj_label] / min(object_distribution)) * (
                        object_distribution[obj_label] / mean_object))
            else:
                obj_factor.append(pow(object_distribution[obj_label] / mean_object, 1))

        pred_factor = []
        for pred_label in range(len(predicate_informative)):
            if predicate_informative[pred_label] > mean_predicate:
                pred_factor.append((predicate_informative[pred_label] / min(predicate_informative)) * pow(
                    (predicate_informative[pred_label] / mean_object), 1))
            else:
                pred_factor.append(pow(predicate_informative[pred_label] / mean_predicate, 2))

        sub_factor = np.array(sub_factor)
        obj_factor = np.array(obj_factor)
        pred_factor = np.array(pred_factor)
        # repeat_max = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.REPEAT_THRESHOLD

        repeat_dict = {}
        for i in range(len(dataset)):
            anno = dataset.get_groundtruth(i)
            tgt_rel_matrix = anno.get_field('relation')
            tgt_inst_labels = anno.get_field('labels')
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0).numpy()
            tgt_head_idxs = tgt_pair_idxs[:, 0].reshape(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].reshape(-1)
            tgt_rel_labels = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].numpy(
            ).reshape(-1)
            tgt_sub_labels = tgt_inst_labels[tgt_head_idxs].numpy().reshape(-1)
            tgt_obj_labels = tgt_inst_labels[tgt_tail_idxs].numpy().reshape(-1)

            tgt_sub_factor = np.array(sub_factor[tgt_sub_labels - 1]).reshape(-1)
            tgt_obj_factor = np.array(obj_factor[tgt_obj_labels - 1]).reshape(-1)
            tgt_rel_factor = np.array(pred_factor[tgt_rel_labels - 1]).reshape(-1)
            assert len(tgt_sub_factor) == len(tgt_obj_factor)
            assert len(tgt_rel_factor) == len(tgt_sub_factor)
            instance_factor = pow(tgt_sub_factor * tgt_obj_factor, 0.5)

            relation_factor = instance_factor * tgt_rel_factor

            repeat_dict[i] = math.ceil(np.max(relation_factor)) if math.ceil(np.max(relation_factor)) > 1 else 1

        repeat_dict['obj_factor'] = obj_factor
        repeat_dict['sub_factor'] = sub_factor
        repeat_dict['rel_factor'] = pred_factor

        return repeat_dict'''


def resampling_dict_generation(dataset, predicate_list, instance_list,  logger):

    logger.info("using resampling method:" + dataset.resampling_method)
    repeat_dict_dir = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_DICT_DIR
    curr_dir_repeat_dict = os.path.join(cfg.OUTPUT_DIR, "repeat_dict.pkl")
    curr_dir_times_list = os.path.join(cfg.OUTPUT_DIR, "times_list.pkl")
    if repeat_dict_dir is not None and repeat_dict_dir != "" or os.path.exists(curr_dir_repeat_dict):
        if os.path.exists(curr_dir_repeat_dict):
            repeat_dict_dir = curr_dir_repeat_dict

        logger.info("load repeat_dict from " + repeat_dict_dir)
        with open(repeat_dict_dir, 'rb') as f:
            repeat_dict = pickle.load(f)

        logger.info("load repeat_dict from " + curr_dir_times_list)
        with open(curr_dir_times_list, 'rb') as f:
            times_list = pickle.load(f)

        return repeat_dict, times_list

    else:
        logger.info(
            "generate the balance sample by recurrent the predicate")

        if dataset.resampling_method == "faip":
            # when we use the lvis sampling method,
            #global_rf = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR
            #logger.info(f"global repeat factor: {global_rf};  ")
            pass
        else:
            raise NotImplementedError(dataset.resampling_method)

        times_list = [0 for i in range(len(predicate_list) - 1)]

        predicate_times = np.zeros(len(predicate_list))
        for i in range(len(dataset)):
            anno = dataset.get_groundtruth(i)
            tgt_rel_matrix = anno.get_field('relation')
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labels = tgt_rel_matrix[tgt_head_idxs,
                                          tgt_tail_idxs].contiguous().view(-1)

            for each_rel in tgt_rel_labels:
                predicate_times[each_rel] += 1
                times_list[each_rel - 1] += 1
                # predicate_dict[each_rel].append(i)

        predicate_total = sum(predicate_times)
        predicate_frequency = predicate_times[1:] / (predicate_total + 1e-11)

        predicate_dict = [[] for _ in range(len(predicate_list))]
        image2predicate = np.zeros(len(dataset))
        for i in range(len(dataset)):
            anno = dataset.get_groundtruth(i)
            tgt_rel_matrix = anno.get_field('relation')
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labels = tgt_rel_matrix[tgt_head_idxs,
                                            tgt_tail_idxs].contiguous().view(-1)
            min_predicate = tgt_rel_labels[0]
            for each_rel in tgt_rel_labels:
                if predicate_frequency[min_predicate - 1] > predicate_frequency[each_rel - 1]:
                    min_predicate = each_rel
            image2predicate[i] = min_predicate
            predicate_dict[min_predicate].append(i)
        predicate_length = [len(predicate_dict[i + 1]) for i in range(len(predicate_list) - 1)]
        repeat_dict = []
        predicate_end_count = np.zeros(len(predicate_list) - 1)
        predicate_loc = np.zeros(len(predicate_list) - 1)

        end_count = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.END_COUNT

        while int(np.sum(predicate_end_count)) < end_count:
            new_turn = [predicate_dict[i + 1][int(predicate_loc[i])] for i in range(len(predicate_list) - 1)]
            for i in range(len(predicate_list) - 1):
                if predicate_loc[i] + 1 < predicate_length[i]:
                    predicate_loc[i] = predicate_loc[i] + 1
                else:
                    predicate_loc[i] = 0
                    predicate_end_count[i] = 1
            repeat_dict.extend(new_turn)

        rest_start_index = predicate_loc[predicate_end_count != 1]
        rest_index = [i for i, x in enumerate(predicate_end_count) if x != 1]
        for i in range(len(rest_index)):
            repeat_dict.extend(predicate_dict[rest_index[i] + 1][int(rest_start_index[i]):])

        return repeat_dict, times_list


def apply_resampling(index: int, relation: np.ndarray, instance_labels: np.ndarray,
                     repeat_dict: Dict, drop_rate):
    """

    Args:
        index:
        relation: N x 3 array
        repeat_dict: r_c, rc_cls image repeat number and repeat number of each category
        drop_rate:

    Returns:

    """
    relation_non_masked = relation.copy()

    # randomly drop the head and body categories for more balance distribution
    # reduce duplicate head and body for more balances
    '''rel_factor = repeat_dict['rel_factor']
    sub_factor = repeat_dict['sub_factor']
    obj_factor = repeat_dict['obj_factor']
    mean_rel_factor = np.median(rel_factor) / max(rel_factor)
    mean_sub_factor = np.median(sub_factor) / max(sub_factor)
    mean_obj_factor = np.median(obj_factor) / max(obj_factor)

    repeat_factor = repeat_dict[index]

    if repeat_factor > 1:
        selected_rel_index = []
        for i, each_rel in enumerate(relation):
            rel_label = each_rel[-1]
            if rel_factor[rel_label - 1] is not None:
                selected_rel_index.append(i)
        if len(selected_rel_index) > 0:
            selected_head_rel_index = np.array(selected_rel_index, dtype=int)
            total_repeat_times = repeat_factor

            ignored_rel = np.random.normal(mean_rel_factor, 0.05, size=len(selected_head_rel_index)).reshape(-1)
            ignored_sub =  np.random.normal(mean_sub_factor, 0.1, size=len(selected_head_rel_index)).reshape(-1)
            ignored_obj =  np.random.normal(mean_obj_factor, 0.1, size=len(selected_head_rel_index)).reshape(-1)

            # ignored_rel = np.random.uniform(0, 1, size=len(selected_head_rel_index)).reshape(-1)
            # ignored_sub = np.random.uniform(0, 1, size=len(selected_head_rel_index)).reshape(-1)
            # ignored_obj = np.random.uniform(0, 1, size=len(selected_head_rel_index)).reshape(-1)

            rel_repeat_factor = np.array([rel_factor[rel - 1] for rel in relation[:, -1]])
            sub_repeat_factor = np.array([sub_factor[instance_labels[sub] - 1] for sub in relation[:, 0]])
            obj_repeat_factor = np.array([obj_factor[instance_labels[obj] - 1] for obj in relation[:, 1]])

            rel_drop_rate = (1 - (rel_repeat_factor / (total_repeat_times + 1e-11)))
            sub_drop_rate = (1 - (sub_repeat_factor / (np.mean(sub_factor) + 1e-11)))
            obj_drop_rate = (1 - (obj_repeat_factor / (np.mean(obj_factor) + 1e-11)))

            # rel_drop_rate = (1 - (rel_repeat_factor / total_repeat_times + 1e-11))
            # sub_drop_rate = (1 - (sub_repeat_factor / total_repeat_times + 1e-11))
            # obj_drop_rate = (1 - (obj_repeat_factor / total_repeat_times + 1e-11))

            drop_rel = ignored_rel < rel_drop_rate
            drop_sub = ignored_sub < sub_drop_rate
            drop_obj = ignored_obj < obj_drop_rate

            #weight = np.random.randint(low=1, high=3, size=len(selected_head_rel_index)).reshape(-1)
            weight = 2
            drop_triplet = weight * drop_rel + drop_obj + drop_sub
            drop_index = np.zeros(len(selected_head_rel_index)).reshape(-1)

            compare = 3 # if repeat_factor > 1 else 2
            for i in range(len(selected_head_rel_index)):
                if drop_triplet[i] >= compare:
                    drop_index[i] = 1
            drop_index = np.array(drop_index, dtype=bool)
            selected_head_rel_index = np.array(selected_head_rel_index, dtype=int)
            relation[selected_head_rel_index[drop_index], -1] = -1'''

    return relation, relation_non_masked
