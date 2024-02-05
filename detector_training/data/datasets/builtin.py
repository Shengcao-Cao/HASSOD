# modified from https://github.com/facebookresearch/CutLER/blob/main/cutler/data/datasets/builtin.py

"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from .builtin_meta import _get_builtin_metadata
from .coco import register_coco_instances

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO_SEMI = {}
_PREDEFINED_SPLITS_COCO_SEMI["coco_semi"] = {
    # we use seed 42 to be consistent with previous works on SSL detection and segmentation
    "coco_semi_1perc": ("coco/train2017", "coco/annotations/1perc_instances_train2017.json"),
    "coco_semi_2perc": ("coco/train2017", "coco/annotations/2perc_instances_train2017.json"),
    "coco_semi_5perc": ("coco/train2017", "coco/annotations/5perc_instances_train2017.json"),
    "coco_semi_10perc": ("coco/train2017", "coco/annotations/10perc_instances_train2017.json"),
    "coco_semi_20perc": ("coco/train2017", "coco/annotations/20perc_instances_train2017.json"),
    "coco_semi_30perc": ("coco/train2017", "coco/annotations/30perc_instances_train2017.json"),
    "coco_semi_40perc": ("coco/train2017", "coco/annotations/40perc_instances_train2017.json"),
    "coco_semi_50perc": ("coco/train2017", "coco/annotations/50perc_instances_train2017.json"),
    "coco_semi_60perc": ("coco/train2017", "coco/annotations/60perc_instances_train2017.json"),
    "coco_semi_80perc": ("coco/train2017", "coco/annotations/80perc_instances_train2017.json"),
}

_PREDEFINED_SPLITS_COCO_CA = {}
_PREDEFINED_SPLITS_COCO_CA["coco_cls_agnostic"] = {
    "cls_agnostic_coco_val": ("coco/val2017", "coco/annotations/hassod/coco_cls_agnostic_instances_val2017.json"),
    "cls_agnostic_coco+lvis_val": ("coco/val2017", "coco/annotations/hassod/coco+lvis_cls_agnostic_instances_val2017.json"),
    "cls_agnostic_coco_train+unlabeled_pseudo": ("coco/train+unlabeled2017", "coco/annotations/hassod/coco_fixsize480_thresh0.1_0.2_0.4_hier.json"),
}

_PREDEFINED_SPLITS_IMAGENET = {}
_PREDEFINED_SPLITS_IMAGENET["imagenet"] = {
    # maskcut annotations
    "imagenet_train": ("imagenet/train", "imagenet/annotations/imagenet_train_fixsize480_tau0.15_N3.json"),
    # self-training round 1
    "imagenet_train_r1": ("imagenet/train", "imagenet/annotations/cutler_imagenet1k_train_r1.json"),
    # self-training round 2
    "imagenet_train_r2": ("imagenet/train", "imagenet/annotations/cutler_imagenet1k_train_r2.json"),
    # self-training round 3
    "imagenet_train_r3": ("imagenet/train", "imagenet/annotations/cutler_imagenet1k_train_r3.json"),
}

_PREDEFINED_SPLITS_VOC = {}
_PREDEFINED_SPLITS_VOC["voc"] = {
    'cls_agnostic_voc': ("voc/", "voc/annotations/trainvaltest_2007_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_CROSSDOMAIN = {}
_PREDEFINED_SPLITS_CROSSDOMAIN["cross_domain"] = {
    'cls_agnostic_clipart': ("clipart/", "clipart/annotations/traintest_cls_agnostic.json"),
    'cls_agnostic_watercolor': ("watercolor/", "watercolor/annotations/traintest_cls_agnostic.json"),
    'cls_agnostic_comic': ("comic/", "comic/annotations/traintest_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_KITTI = {}
_PREDEFINED_SPLITS_KITTI["kitti"] = {
    'cls_agnostic_kitti': ("kitti/", "kitti/annotations/trainval_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_LVIS = {}
_PREDEFINED_SPLITS_LVIS["lvis"] = {
    "cls_agnostic_lvis": ("coco/", "coco/annotations/hassod/lvis1.0_cocofied_val_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_OBJECTS365 = {}
_PREDEFINED_SPLITS_OBJECTS365["objects365"] = {
    'cls_agnostic_objects365': ("objects365/val", "objects365/annotations/zhiyuan_objv2_val_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_OpenImages = {}
_PREDEFINED_SPLITS_OpenImages["openimages"] = {
    'cls_agnostic_openimages': ("openImages/validation", "openImages/annotations/openimages_val_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_UVO = {}
_PREDEFINED_SPLITS_UVO["uvo"] = {
    "cls_agnostic_uvo": ("uvo/all_UVO_frames", "uvo/annotations/val_sparse_cleaned_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_OTHERS = {}
_PREDEFINED_SPLITS_OTHERS["ade20k"] = {
    "cls_agnostic_ade20k": ("ADEChallengeData2016/images/validation", "ADEChallengeData2016/ade20k_instance_val_cls_agnostic.json"),
}
_PREDEFINED_SPLITS_OTHERS["entityseg"] = {
    "cls_agnostic_entityseg": ("EntitySeg", "EntitySeg/entityseg_val_lr_cls_agnostic.json"),
}
_PREDEFINED_SPLITS_OTHERS["paco"] = {
    "cls_agnostic_paco": ("coco", "coco/annotations/cutler/paco_lvis_v1_val_part_cls_agnostic.json"),
}
_PREDEFINED_SPLITS_OTHERS["partimagenet"] = {
    "cls_agnostic_partimagenet": ("PartImageNet/images/val", "PartImageNet/annotations/val/val_cls_agnostic.json"),
}
_PREDEFINED_SPLITS_OTHERS["sa1b"] = {
    "cls_agnostic_sa1b": ("SA1B/images", "SA1B/annotations/sa1b_p0_cls_agnostic.json"),
}

def register_all_imagenet(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_IMAGENET.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_voc(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_VOC.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_cross_domain(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_CROSSDOMAIN.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_kitti(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_KITTI.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_objects365(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_OBJECTS365.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_openimages(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_OpenImages.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_uvo(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_UVO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_others(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_OTHERS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_coco_semi(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO_SEMI.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_coco_ca(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO_CA.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                extra_annotation_keys=["hier_level"],
            )

_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
register_all_coco_semi(_root)
register_all_coco_ca(_root)
register_all_imagenet(_root)
register_all_uvo(_root)
register_all_voc(_root)
register_all_cross_domain(_root)
register_all_kitti(_root)
register_all_openimages(_root)
register_all_objects365(_root)
register_all_lvis(_root)
register_all_others(_root)