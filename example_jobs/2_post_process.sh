conda activate hassod
cd hac

# First merge the JSON files generated by each HAC job
# Need to merge for each threshold individually
python merge_jsons.py \
    --base-dir ../save/hac \
    --save-path ../save/hac/coco_fixsize480_thresh0.1.json \
    --num-image 5000 \
    --thresh 0.1 \
    --end_idx 241690

python merge_jsons.py \
    --base-dir ../save/hac \
    --save-path ../save/hac/coco_fixsize480_thresh0.2.json \
    --num-image 5000 \
    --thresh 0.2 \
    --end_idx 241690

python merge_jsons.py \
    --base-dir ../save/hac \
    --save-path ../save/hac/coco_fixsize480_thresh0.4.json \
    --num-image 5000 \
    --thresh 0.4 \
    --end_idx 241690

# Then run the post-processing script to ensemble the results from multiple thresholds
# which also analyzes the hierarchical levels of annotations
python post_process.py \
    --input ../save/hac/coco_fixsize480_thresh0.1.json ../save/hac/coco_fixsize480_thresh0.2.json ../save/hac/coco_fixsize480_thresh0.4.json \
    --output ../save/hac/coco_fixsize480_thresh0.1_0.2_0.4_hier.json