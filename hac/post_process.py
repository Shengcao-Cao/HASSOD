# written by Shengcao Cao

import argparse
import json
import numpy as np
import pycocotools.mask as mask_util
import multiprocessing as mp
from functools import partial

def compute_iou(mask1, masks):
    intersection = np.logical_and(mask1, masks).sum(axis=(1, 2))
    union = np.logical_or(mask1, masks).sum(axis=(1, 2))
    return intersection / union

def compute_coverage(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    mask1_sum = mask1.sum()
    return intersection / mask1_sum

def assign_levels(list_of_objs, tree, current_level, current_node):
    for child in tree.get(current_node, []):
        hier_level = min(current_level, 2)
        list_of_objs[child]['hier_level'] = hier_level
        assign_levels(list_of_objs, tree, current_level + 1, child)

def build_tree_and_assign_levels(list_of_objs, coverage_threshold=0.9):
    tree = {}
    root_nodes = set()
    decoded_masks = [mask_util.decode(obj['segmentation']) for obj in list_of_objs]

    for idx_a, mask_a in enumerate(decoded_masks):
        smallest_parent = None
        smallest_parent_area = float('inf')

        for idx_b, mask_b in enumerate(decoded_masks):
            if idx_a == idx_b:
                continue

            coverage_a_in_b = compute_coverage(mask_a, mask_b)
            coverage_b_in_a = compute_coverage(mask_b, mask_a)

            if coverage_a_in_b >= coverage_threshold and coverage_b_in_a < coverage_threshold:
                mask_b_area = mask_b.sum()

                if mask_b_area < smallest_parent_area:
                    smallest_parent = idx_b
                    smallest_parent_area = mask_b_area

        if smallest_parent is not None:
            if smallest_parent not in tree:
                tree[smallest_parent] = []
            tree[smallest_parent].append(idx_a)
        else:
            root_nodes.add(idx_a)

    for root in root_nodes:
        list_of_objs[root]['hier_level'] = 0
        assign_levels(list_of_objs, tree, 1, root)

    return list_of_objs

def process_image_objs(list_of_objs, progress_dict, lock, coverage_threshold=0.9):
    list_of_objs = build_tree_and_assign_levels(list_of_objs, coverage_threshold)

    with lock:
        progress_dict['count'] += 1
        if progress_dict['count'] % 100 == 0:
            print(f"Processed images: {progress_dict['count']}", flush=True)

    return list_of_objs

def process_images(image_id_to_objs, coverage_threshold=0.9):
    with mp.Pool(processes=8) as pool:
        manager = mp.Manager()
        progress_dict = manager.dict()
        progress_dict['count'] = 0
        lock = manager.Lock()
        process_image_objs_partial = partial(process_image_objs,
            progress_dict=progress_dict,
            lock=lock,
            coverage_threshold=coverage_threshold
        )
        results = pool.map(process_image_objs_partial, list(image_id_to_objs.values()), chunksize=1250)

    flatten_objs = []
    for objs in results:
        flatten_objs.extend(objs)

    return flatten_objs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-process annotations')
    parser.add_argument('--input', type=str, nargs='+', help='path to the input annotation file(s)')
    parser.add_argument('--output', type=str, help='path to the output annotation file')
    parser.add_argument('--coverage', type=float, default=0.90, help='coverage threshold for hierarchical relations')
    args = parser.parse_args()

    if not isinstance(args.input, list):
        args.input = [args.input]

    image_id_to_objs = {}
    for anno_file in args.input:
        anno = json.load(open(anno_file))
        print(f"Annotations from {anno_file}: {len(anno['annotations'])}")
        for idx, obj in enumerate(anno['annotations']):
            image_id = obj['image_id']
            if image_id in image_id_to_objs:
                image_id_to_objs[image_id].append(obj)
            else:
                image_id_to_objs[image_id] = [obj]

    flatten_objs = process_images(image_id_to_objs, args.coverage)

    obj_id = 1
    for obj in flatten_objs:
        obj['id'] = obj_id
        obj_id += 1

    print('After processing, annotations:', len(flatten_objs))

    # # can use the following to check the distribution of hier_level
    # level_counters = {}
    # for obj in flatten_objs:
    #     level = obj['hier_level']
    #     if level not in level_counters:
    #         level_counters[level] = 0
    #     level_counters[level] += 1
    # print('Level counters:', level_counters)

    anno = json.load(open(args.input[0]))
    anno['annotations'] = flatten_objs

    with open(args.output, 'w') as f:
        json.dump(anno, f, indent=2)
