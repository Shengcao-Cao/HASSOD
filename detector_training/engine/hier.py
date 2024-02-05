# written by Shengcao Cao

import torch

def compute_coverage(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2).sum()
    mask1_sum = mask1.sum()
    return intersection / mask1_sum

def assign_levels(levels, tree, current_level, current_node):
    for child in tree.get(current_node, []):
        hier_level = min(current_level, 2)
        levels[child] = hier_level
        assign_levels(levels, tree, current_level + 1, child)

def build_tree_and_assign_levels(masks, coverage_threshold=0.9):
    assert len(masks.shape) == 3
    max_area = masks.shape[1] * masks.shape[2] + 1
    tree = {}
    root_nodes = set()

    for idx_a in range(masks.shape[0]):
        mask_a = masks[idx_a]
        smallest_parent = None
        smallest_parent_area = max_area

        for idx_b in range(masks.shape[0]):
            mask_b = masks[idx_b]
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

    levels = [-1] * masks.shape[0]
    for root in root_nodes:
        levels[root] = 0
        assign_levels(levels, tree, 1, root)

    return levels
