# modified from https://github.com/facebookresearch/CutLER/blob/main/cutler/train_net.py

"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.data import MetadataCatalog
from engine import DefaultTrainer, MeanTeacherTrainer, default_argument_parser, default_setup
from detectron2.engine import hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    # COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from evaluation import COCOEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA
import data
import modeling.roi_heads
import modeling.proposal_generator

def add_hassod_config(cfg):
    cfg.DATALOADER.COPY_PASTE = False
    cfg.DATALOADER.COPY_PASTE_RATE = 0.0
    cfg.DATALOADER.COPY_PASTE_MIN_RATIO = 0.5
    cfg.DATALOADER.COPY_PASTE_MAX_RATIO = 1.0
    cfg.DATALOADER.COPY_PASTE_RANDOM_NUM = True
    cfg.DATALOADER.VISUALIZE_COPY_PASTE = False

    cfg.MODEL.ROI_HEADS.USE_DROPLOSS = False
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = 0.0

    cfg.SOLVER.BASE_LR_MULTIPLIER = 1
    cfg.SOLVER.BASE_LR_MULTIPLIER_NAMES = []

    cfg.TEST.NO_SEGM = False

    # configs for hierarchical level prediction
    cfg.MODEL.ROI_BOX_HEAD.PRED_LEVEL = False
    cfg.MODEL.ROI_BOX_HEAD.PRED_LEVEL_LOSS_TYPE = 'focal'
    cfg.MODEL.ROI_BOX_HEAD.PRED_LEVEL_NUMLEVELS = 3
    cfg.MODEL.ROI_BOX_HEAD.WEIGHT_BY_CONFID = False

    # configs for Mean Teacher
    cfg.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    cfg.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    cfg.DATASETS.CROSS_DATASET = False

    cfg.SOLVER.IMS_PER_BATCH_LABEL = 8
    cfg.SOLVER.IMS_PER_BATCH_UNLABEL = 8
    cfg.SOLVER.BASE_LR_MULTIPLIER = 1
    cfg.SOLVER.BASE_LR_MULTIPLIER_NAMES = []

    cfg.SEMISUPNET = CN()
    cfg.SEMISUPNET.TRAINER = "default"
    cfg.SEMISUPNET.BBOX_THRESHOLD = 0.7
    cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    cfg.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    cfg.SEMISUPNET.BURN_UP_STEP = 0
    cfg.SEMISUPNET.EMA_KEEP_RATE = 0.9996
    cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT = 2.0
    cfg.SEMISUPNET.SUP_LOSS_WEIGHT = 1.0
    cfg.SEMISUPNET.LOSS_WEIGHT_SCHEDULE = "cosine"
    cfg.SEMISUPNET.REMOVE_ERASED_OBJECTS = False

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, no_segm=cfg.TEST.NO_SEGM))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_hassod_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.TRAINER == "default":
        Trainer = DefaultTrainer
    elif cfg.SEMISUPNET.TRAINER == "mean_teacher":
        Trainer = MeanTeacherTrainer
    else:
        raise ValueError("Trainer Name is not found.")
    Trainer.build_evaluator = build_evaluator

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
