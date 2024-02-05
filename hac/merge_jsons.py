# modified from https://github.com/facebookresearch/CutLER/blob/main/maskcut/merge_jsons.py

import os
import json
import argparse

if __name__ == '__main__':
    # load model arguments
    parser = argparse.ArgumentParser(description='Merge JSON files')
    parser.add_argument('--base-dir', type=str, help='directory of HAC-generated annotation files')
    parser.add_argument('--save-path', type=str, help='path to save the merged annotation file')
    parser.add_argument('--num-image', type=int, default=1, help='number of images per json file')
    parser.add_argument('--fixed-size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--thresh', type=float, default=0.1, help='merging threshold')
    parser.add_argument('--start_idx', type=int, default=0, help='start of image index')
    parser.add_argument('--end_idx', type=int, default=241690, help='end of image index')

    args = parser.parse_args()

    base_name = 'coco_fixsize{}_thresh{}'.format(args.fixed_size, args.thresh)

    start_idx = args.start_idx
    every_k = args.num_image
    tobe_merged_ann_dicts = []

    while start_idx < args.end_idx:
        end_idx = min(start_idx + every_k, args.end_idx)
        filename = '{}_{}_{}.json'.format(base_name, start_idx, end_idx)
        tobe_merged = os.path.join(args.base_dir, filename)
        if not os.path.isfile(tobe_merged):
            print('File not found:', tobe_merged)
        else:
            start_idx += every_k
        tobe_merged_ann_dict = json.load(open(tobe_merged))
        tobe_merged_ann_dicts.append(tobe_merged_ann_dict)

    # re-generate image_id and segment_id, and combine annotation info and image info from all annotation files
    base_ann_dict = tobe_merged_ann_dicts[0]
    for tobe_merged_ann_dict in tobe_merged_ann_dicts[1:]:
        base_ann_dict['images'].extend(tobe_merged_ann_dict['images'])
        base_ann_dict['annotations'].extend(tobe_merged_ann_dict['annotations'])

    segment_id = 1
    for ann in base_ann_dict['annotations']:
        ann['id'] = segment_id
        segment_id += 1

    # save the final json file
    anns = [ann['id'] for ann in base_ann_dict['annotations']]
    anns_image_id = [ann['image_id'] for ann in base_ann_dict['annotations']]
    json.dump(base_ann_dict, open(args.save_path, 'w'), indent=2)
    print('Done: {} images; {} anns.'.format(len(base_ann_dict['images']), len(base_ann_dict['annotations'])))
