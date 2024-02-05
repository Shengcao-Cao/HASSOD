conda activate hassod
cd detector_training

python train_net.py --num-gpus 4 \
    --eval-only \
    --config-file configs/cascade_mask_rcnn_R_50_FPN.yaml \
    SEMISUPNET.TRAINER "default" \
    MODEL.WEIGHTS ../save/mean_teacher/model_student.pth \
    MODEL.ROI_BOX_HEAD.PRED_LEVEL True \
    MODEL.ROI_BOX_HEAD.PRED_LEVEL_LOSS_TYPE "focal" \
    DATASETS.TRAIN "('cls_agnostic_coco_train+unlabeled_pseudo',)" \
    DATASETS.TEST "('cls_agnostic_lvis',)" \
    TEST.DETECTIONS_PER_IMAGE 1000 \
    OUTPUT_DIR ../save/mean_teacher/eval_student_lvis
