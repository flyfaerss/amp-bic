# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import argparse
import datetime
import os
import random
import time
import threading
import json
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

#import gpustat
import numpy as np
import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils import visualize_graph as vis_graph
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print, TFBoardHandler_LEVEL
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.global_buffer import save_buffer
from maskrcnn_benchmark.data.datasets.visual_genome import get_frequency_distribution

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError("Use APEX for multi-precision via apex.amp")

SEED = 3407 # 3407

torch.cuda.manual_seed(SEED)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(SEED)  # 为所有GPU设置随机种子
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.enabled = True  # 默认值
torch.backends.cudnn.benchmark = True  # 默认为False
torch.backends.cudnn.deterministic = True  # 默认为False;benchmark为True时,y要排除随机性必须为True


# torch.autograd.set_detect_anomaly(True)

SHOW_COMP_GRAPH = False


def show_params_status(model):
    """
    Prints parameters of a model
    """
    st = {}
    strings = []
    total_params = 0
    trainable_params = 0
    for p_name, p in model.named_parameters():

        if not ("bias" in p_name.split(".")[-1] or "bn" in p_name.split(".")[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
        if p.requires_grad:
            trainable_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in st.items():
        strings.append(
            "{:<80s}: {:<16s}({:8d}) ({})".format(
                p_name, "[{}]".format(",".join(size)), prod, "grad" if p_req_grad else "    "
            )
        )
    strings = "\n".join(strings)
    return (
        f"\n{strings}\n ----- \n \n"
        f"      trainable parameters:  {trainable_params/ 1e6:.3f}/{total_params / 1e6:.3f} M \n "
    )


def train(
    cfg,
    local_rank,
    distributed,
    logger,
):
    debug_print(logger, "Start initializing dataset & dataloader")

    arguments = {}
    arguments["iteration"] = 0
    output_dir = cfg.OUTPUT_DIR

    train_data_loader = make_data_loader(
        cfg,
        mode="train",
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode="val",
        is_distributed=distributed,
    )

    debug_print(logger, "end dataloader")

    debug_print(logger, "prepare training")
    model = build_detection_model(cfg)
    model.train()
    debug_print(logger, "end model construction")
    logger.info(str(model))
    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (
        model.rpn,
        model.backbone,
        model.roi_heads.box,
    )
    train_modules = ()
    rel_pn_module_ref = []
    if cfg.MODEL.ROI_RELATION_HEAD.FIX_FEATURE:
        if cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON:
            if cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD == "rel_pn":
                rel_pn_module_ref.append(model.roi_heads.relation.rel_pn)
            elif cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD == "pre_clser":
                rel_pn_module_ref.append(
                    model.roi_heads.relation.predictor.context_layer.pre_rel_classifier
                )

    fix_eval_modules(eval_modules)
    set_train_modules(train_modules)

    logger.info("trainable models:")
    logger.info(show_params_status(model))

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    slow_heads = []
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "PISA_MODULE":
        if cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.RELATION_CONFIDENCE_AWARE:
            slow_heads = [
                "roi_heads.relation.predictor.context_layer.relation_conf_aware_models",
            ]

    except_weight_decay = []

    # load pretrain layers to new layers
    load_mapping = {
        "roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
        "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor",
    }

    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping[
            "roi_heads.relation.att_feature_extractor"
        ] = "roi_heads.attribute.feature_extractor"
        load_mapping[
            "roi_heads.relation.union_feature_extractor.att_feature_extractor"
        ] = "roi_heads.attribute.feature_extractor"

    print("load model to GPU")
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(
        cfg,
        model,
        logger,
        slow_heads=slow_heads,
        slow_ratio=10,
        rl_factor=float(num_batch),
        except_weight_decay=except_weight_decay,
    )
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, "end optimizer and shcedule")
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = "O1" if use_mixed_precision else "O0"
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # todo, unless mark as resume, otherwise load from the pretrained checkpoint
    if cfg.MODEL.PRETRAINED_DETECTOR_CKPT != "":
        checkpointer.load(
            cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping
        )
    else:
        checkpointer.load(
            cfg.MODEL.WEIGHT,
            with_optim=False,
        )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    debug_print(logger, "end load checkpointer")

    if cfg.MODEL.ROI_RELATION_HEAD.RE_INITIALIZE_CLASSIFIER:
        model.roi_heads.relation.predictor.init_classifier_weight()

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, "end distributed")

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed, logger)

    '''pre_clser_pretrain_on = False
    if (
        cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRETRAIN_RELNESS_MODULE
        and cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
    ):
        if distributed:
            m2opt = model.module
        else:
            m2opt = model
        # m2opt.roi_heads.relation.predictor.start_preclser_relpn_pretrain()
        # m2opt.roi_heads.relation.predictor.end_preclser_relpn_pretrain()
        logger.info("Start preclser_relpn_pretrain")
        pre_clser_pretrain_on = True

        STOP_ITER = (
            cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRETRAIN_ITER_RELNESS_MODULE
        )'''

    curriculmn_learning = cfg.MODEL.ROI_RELATION_HEAD.PISA_MODULE.USE_CURRICULUM

    head_num, body_num, tail_num = 7, 21, 22 # 7, 21, 22 # 5, 21, 24 # 7, 21, 22
    head_idx = torch.tensor([8, 20, 29, 30, 31]).to(device)
    body_idx = torch.tensor([6, 11, 12, 14, 19, 21, 22, 23, 24, 25, 33, 35, 38, 40, 41, 43, 46, 47, 48, 49, 50]).to(device)
    all_body_idx = torch.tensor([6, 8, 11, 12, 14, 19, 20, 21, 22, 23, 24, 25, 29, 30, 31, 33, 35, 38, 40, 41, 43, 46, 47, 48, 49, 50]).to(device)
    tail_idx = torch.tensor([1, 2, 3, 4, 5, 7, 9, 10, 13, 15, 16, 17, 18, 26, 27, 28, 32, 34, 36, 37, 39, 42, 44, 45]).to(device)
    predicate_frequency_dir = cfg.OUTPUT_DIR
    predicate_distribution = torch.tensor(get_frequency_distribution(predicate_frequency_dir)).to(device)
    frequency_median = torch.median(predicate_distribution)
    a, idx = torch.sort(predicate_distribution, descending=False)
    # body_first = torch.min(predicate_distribution[body_idx])
    # tail_first = torch.min(predicate_distribution[tail_idx])
    body_first = a[head_num + 1]
    tail_first = a[head_num + body_num + 1]
    tail_end = a[-1]
    tail_mean = torch.mean(a[head_num + body_num + 1:])
    body_start = torch.mean(a[head_num + 1: head_num + body_num + 1]) / torch.mean(a[head_num + body_num + 1:])
    weight_ceiling = torch.ones(head_num + body_num + tail_num + 1).to(device)
    weight_ceiling[predicate_distribution > tail_first] = torch.sqrt(torch.exp(predicate_distribution[
                                                        predicate_distribution > tail_first]) / torch.exp(tail_first))
    weight_ceiling[predicate_distribution < tail_first] = torch.sqrt(torch.exp(predicate_distribution[
                                                        predicate_distribution < tail_first]) / torch.exp(tail_first))
    logger.info(f"base weight: {weight_ceiling}")
    #weight_ceiling[predicate_distribution < body_first] *= torch.sqrt(torch.exp(predicate_distribution[
    #                                                    predicate_distribution < body_first]) / torch.exp(body_first))
    '''weight_ceiling[tail_idx] = torch.sqrt(torch.exp(predicate_distribution[
                                                                                   tail_idx]) / torch.exp(tail_first))
    weight_ceiling[all_body_idx] = torch.sqrt(torch.exp(predicate_distribution[
                                                                                   all_body_idx]) / torch.exp(tail_first))
    weight_ceiling[head_idx] *= torch.sqrt(torch.exp(predicate_distribution[
                                                                                    head_idx]) / torch.exp(body_first))'''
    predicate_weight = torch.ones(head_num + body_num + tail_num + 1).to(device)

    max_epoch = cfg.SOLVER.MAX_EPOCH

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    model.train()

    multi_factor = cfg.SOLVER.MULTIPLY_FACTOR
    init_head_weight = 0.0
    init_body_weight = 0.2
    init_tail_weight = 1.0

    print_first_grad = True
    for epoch in range(max_epoch):
        for iteration, (images, targets, _) in tqdm(enumerate(train_data_loader, start_iter)):
            if any(len(target) < 1 for target in targets):
                logger.error(
                    f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}"
                )
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration



            model.train()
            fix_eval_modules(eval_modules)

            images = images.to(device)
            targets = [target.to(device) for target in targets]

            # loss_weight = torch.sigmoid(torch.tensor((iteration - float(max_iter) / 2) / 2500.0)).to(device)
            # temp_iter = int(iteration / 2000) * 2000
            loss_weight = torch.tensor((epoch + 1) / float(multi_factor * max_epoch + 1)).to(device)

            if epoch < 0.3 * max_epoch:
                ratio = epoch
            elif 0.3 * max_epoch <= epoch < 0.6 * max_epoch:
                ratio = 0.6 * max_epoch - epoch
            elif 0.6 * max_epoch <= epoch < 0.8 * max_epoch:
                ratio = epoch - (0.6 * max_epoch - 1)
            else:
                ratio = epoch - (0.8 * max_epoch - 1)

            if curriculmn_learning:
                # 0.007, 0.103, 1 ;
                predicate_weight[predicate_distribution < tail_first] = weight_ceiling[predicate_distribution < tail_first] * (init_body_weight + ratio / float(multi_factor * max_epoch))
                predicate_weight[predicate_distribution >= tail_first] = weight_ceiling[predicate_distribution >= tail_first] * (init_tail_weight + ratio / float(multi_factor * max_epoch))
                predicate_weight[predicate_distribution < body_first] = weight_ceiling[predicate_distribution < body_first] * (init_head_weight + ratio / float(multi_factor * max_epoch))
                predicate_weight[0] = 1.0
                loss_dict = model(images, targets, logger=logger, loss_weight=loss_weight,
                                  predicate_weight=predicate_weight)
            else:
                loss_dict = model(images, targets, logger=logger) #, loss_weight=loss_weight)
            losses = sum(loss for loss in loss_dict.values())
            flag = 1
            if epoch >= 0.6 * max_epoch and epoch < 0.8 * max_epoch:
                losses = 0.1 * losses
                flag = 0.1
            elif epoch >= 0.8 * max_epoch:
                losses = 0.01 * losses
                flag = 0.01

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            meters.update(loss=losses_reduced, **loss_dict_reduced)
            optimizer.zero_grad()
            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            # try:
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()

            # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
            verbose = iteration == 0 or print_first_grad  # print grad or not
            print_first_grad = False
            clip_grad_norm(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad],
                max_norm=cfg.SOLVER.GRAD_NORM_CLIP,
                logger=logger,
                verbose=verbose,
                clip=True,
            )

            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            elapsed_time = str(datetime.timedelta(seconds=int(end - start_training_time)))

            if iteration % 200 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "\ninstance name: {instance_name}\t",
                            "epoch: {epoch}\t",
                            "eta: {eta}\t",
                            "iter: {iter}\t",
                            "lr: {lr:.6f}\t",
                            "max mem: {memory:.0f}\n",
                            "rel_weight: {rel_weight:.3f} --- feature_weight: {feature_weight:.3f}\n",
                            "{meters}",
                        ]
                    ).format(
                        instance_name=cfg.OUTPUT_DIR[len("checkpoints/"):],
                        epoch=epoch + 1,
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"] * flag,
                        rel_weight=loss_weight,
                        feature_weight=1.0 - loss_weight,
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

            #if iteration % checkpoint_period == 0:
            #    checkpointer.save("model_{}_{:06d}".format(epoch + 1, iteration), **arguments)
            if iteration == max_iter or iteration == len(train_data_loader):
                checkpointer.save("model_{}".format(epoch + 1), **arguments)

            val_result_value = None  # used for scheduler updating
            if cfg.SOLVER.TO_VAL and (iteration == max_iter): # or iteration % 3000 == 0):
                logger.info("Start validating")
                val_result = run_val(cfg, model, val_data_loaders, distributed, logger)
                val_result_value = val_result[1]
                if get_rank() == 0:
                    for each_ds_eval in val_result[0]:
                        for each_evalator_res in each_ds_eval[1]:
                            logger.log(TFBoardHandler_LEVEL, (each_evalator_res, iteration))
            # scheduler should be called after optimizer.step() in pytorch>=1.1.0
            # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
                scheduler.step(val_result_value, epoch=iteration)
                if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                    logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                    break
            else:
                scheduler.step()

            #if iteration == max_iter:
            #    break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    return model


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        if module is None:
            continue

        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False


def set_train_modules(modules):
    for module in modules:
        for _, param in module.named_parameters():
            param.requires_grad = True


def run_val(cfg, model, val_data_loaders, distributed, logger):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
            cfg,
            model,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=None,
            logger=logger,
        )
        synchronize()
        val_result.append(dataset_result)

    val_values = []
    for each in val_result:
        if isinstance(each, tuple):
            val_values.append(each[0])
    # support for multi gpu distributed testing
    # send evaluation results to each process
    gathered_result = all_gather(torch.tensor(val_values).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result >= 0]
    val_result_val = float(valid_result.mean())

    del gathered_result, valid_result
    return val_result, val_result_val


def run_test(cfg, model, distributed, logger):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode="test", is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(
        output_folders, dataset_names, data_loaders_val
    ):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="/home/sylvia/yjr/sgg/benchmark38/PySGG/configs/e2e_afe_oiv6.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        new_timeout = datetime.timedelta(minutes=60)
        torch.distributed.init_process_group(backend="nccl", timeout=new_timeout, init_method="env://")
        synchronize()

    try:
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
    except Exception as e:
        print(e)

    # mode
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = "predcls"
        else:
            mode = "sgcls"
    else:
        mode = "sgdet"

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M") #

    cfg.OUTPUT_DIR = os.path.join(
        cfg.OUTPUT_DIR,
        f"{mode}-{cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR}",
        f"({time_str}){cfg.EXPERIMENT_NAME}"
    )

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    # if cfg.DEBUG:
    #     logger.info("Collecting env info (might take some time)")
    #     logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, "config.yml")
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed, logger)

    if not args.skip_test:
        run_test(cfg, model, args.distributed, logger)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "8"
    main()
