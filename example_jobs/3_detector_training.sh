conda activate hassod
cd detector_training

# Use the training routine in Detectron2 to initial detector training
# Please note the configurations and make changes if necessary
python train_net.py --num-gpus 4 \
    --config-file configs/cascade_mask_rcnn_R_50_FPN.yaml \
    SEMISUPNET.TRAINER "default" \
    MODEL.WEIGHTS ../checkpoints/dino_RN50_pretrain_d2_format.pkl \
    MODEL.ROI_BOX_HEAD.PRED_LEVEL True \
    MODEL.ROI_BOX_HEAD.PRED_LEVEL_LOSS_TYPE "focal" \
    DATASETS.TRAIN "('cls_agnostic_coco_train+unlabeled_pseudo',)" \
    DATASETS.TEST "('cls_agnostic_coco+lvis_val',)" \
    SOLVER.MAX_ITER 40000 \
    TEST.EVAL_PERIOD 5000 \
    TEST.DETECTIONS_PER_IMAGE 1000 \
    OUTPUT_DIR ../save/detector_training
