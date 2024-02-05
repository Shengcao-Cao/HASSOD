conda activate hassod
cd detector_training

# Convert single model to teacher-student model
python convert_model.py --s-to-ts \
    ../save/detector_training/model_final.pth \
    ../save/detector_training/model_ts.pth

# Initiate Mean Teacher training
python train_net.py --num-gpus 4 \
    --config-file configs/cascade_mask_rcnn_R_50_FPN_MT.yaml \
    SEMISUPNET.TRAINER "mean_teacher" \
    MODEL.WEIGHTS ../save/detector_training/model_ts.pth \
    MODEL.ROI_BOX_HEAD.PRED_LEVEL True \
    MODEL.ROI_BOX_HEAD.PRED_LEVEL_LOSS_TYPE "focal" \
    DATASETS.TRAIN_LABEL "('cls_agnostic_coco_train+unlabeled_pseudo',)" \
    DATASETS.TRAIN_UNLABEL "('cls_agnostic_coco_train+unlabeled_pseudo',)" \
    DATASETS.TEST "('cls_agnostic_coco+lvis_val',)" \
    SOLVER.MAX_ITER 2000 \
    SOLVER.LR_SCHEDULER_NAME "WarmupCosineLR" \
    TEST.EVAL_PERIOD 500 \
    TEST.DETECTIONS_PER_IMAGE 1000 \
    OUTPUT_DIR ../save/mean_teacher

# Convert teacher-student model back to single model
python convert_model.py --ts-to-s \
    ../save/mean_teacher/model_final.pth \
    ../save/mean_teacher/model_student.pth
