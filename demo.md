## Demo
Once you have finished the preparation for the [environment setup](https://github.com/Shengcao-Cao/HASSOD/blob/main/preparation.md) and downloaded [our model](https://drive.google.com/file/d/1pZ1yP3cs-Ezrw3OVJxi4arUFkxQ8W5fo/view?usp=sharing), you can try out the demo and see how HASSOD works on your image.

Check the following example:
```
conda activate hassod
cd demo

python demo.py \
    --config-file ../detector_training/configs/cascade_mask_rcnn_R_50_FPN.yaml \
    --input ../datasets/coco/val2017/000000000139.jpg \
    --output ../save/demo/000000000139.jpg \
    --confidence-threshold 0.5 \
    --hier-level all \
    --opts MODEL.WEIGHTS ../save/model.pth \
    MODEL.ROI_BOX_HEAD.PRED_LEVEL True \
    MODEL.ROI_BOX_HEAD.PRED_LEVEL_LOSS_TYPE "focal"

```
The usage is almost the same as the original Detectron2 [demo](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html). Additionally, to see the model outputs for a set of levels ("whole", "part", and/or "subpart"), you can specify the `--hier-level`. For example, `--hier-level whole part` will generate predictions including whole objects and parts, but excluding subparts.
