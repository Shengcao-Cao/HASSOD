# Please note that this example is just for one data batch
# You need to submit multiple jobs with different `JOB_I` and `GPU_I` to process all the data

# Specify the job index for the current batch
# When `num-image-per-job` is 5000, the total number of jobs is ceil(241690/5000)=49
# So the job index should be 0, 1, 2, ..., 48
JOB_I=0

# Specify which GPU to use
# For example, if you have 4 GPUs, you can set GPU_I=$((JOB_I % 4))
GPU_I=0

conda activate hassod
cd hac

CUDA_VISIBLE_DEVICES=$GPU_I python hac.py \
    --vit-arch base --patch-size 8 \
    --thresh 0.4 0.2 0.1 --fixed_size 480 \
    --pretrain_path ../checkpoints/dino_vitbase8_pretrain.pth \
    --num-image-per-job 5000 --job-index $JOB_I \
    --dataset-path ../datasets/coco/train+unlabeled2017 \
    --out-dir ../save/hac
