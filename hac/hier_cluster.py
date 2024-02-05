# written by Shengcao Cao

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def cos_similarity(r1, r2):
    s = (r1['feature_avg'] * r2['feature_avg']).sum().item()
    return s

def hier_cluster(feature, thresh):
    # allow multiple thresholds
    if not isinstance(thresh, (tuple, list)):
        thresh = [thresh]
    thresh = sorted(list(thresh))[::-1]

    C, H, W = feature.shape

    # build regions
    R = []
    R_n = 0
    for y in range(H):
        for x in range(W):
            r = {
                'label': R_n,
                'pixels': [(x, y)],
                'n_pixels': 1,
                'feature_sum': feature[:, y, x],
                'feature_avg': F.normalize(feature[:, y, x], dim=0, p=2),
            }
            R.append(r)
            R_n += 1

    # build neighbors
    S = {}
    for i, r in enumerate(R):
        # right
        j = i + 1
        if j < R_n and j % W != 0:
            s = cos_similarity(R[i], R[j])
            S[(i, j)] = s
        # down
        j = i + W
        if j < R_n:
            s = cos_similarity(R[i], R[j])
            S[(i, j)] = s

    masks = []

    # start merging
    for thresh_i in thresh:
        while len(S):
            # find maximum similarity
            i, j = max(S, key=S.get)
            s = S[(i, j)]
            if s < thresh_i:
                break

            # merge two regions
            r = {
                'label': R_n,
                'pixels': R[i]['pixels'] + R[j]['pixels'],
                'n_pixels': R[i]['n_pixels'] + R[j]['n_pixels'],
                'feature_sum': R[i]['feature_sum'] + R[j]['feature_sum'],
                'feature_avg': F.normalize((R[i]['feature_sum'] + R[j]['feature_sum']) / (R[i]['n_pixels'] + R[j]['n_pixels']), dim=0, p=2),
            }
            R.append(r)
            R_n += 1

            # record neighbors to remove
            S_remove = []
            for key in S:
                if i == key[0] or i == key[1] or j == key[0] or j == key[1]:
                    S_remove.append(key)
            for key in S_remove:
                del S[key]

            # compute new neighbors
            for key in S_remove:
                if key != (i, j):
                    k = key[1] if key[0] in (i, j) else key[0]
                    S[(k, R_n - 1)] = cos_similarity(R[k], R[R_n - 1])

        # collect remaining regions
        R_reindex = {}
        R_m = 0
        if len(S):
            for (i, j) in S:
                if i not in R_reindex:
                    R_reindex[i] = R_m
                    R_m += 1
                if j not in R_reindex:
                    R_reindex[j] = R_m
                    R_m += 1
        else:
            R_reindex[R_n - 1] = 0
            R_m = 1

        # generate one-hot mask
        mask = np.zeros((R_m, H, W), dtype=np.float32)
        for i, j in R_reindex.items():
            r = R[i]
            for x, y in r['pixels']:
                mask[j, y, x] = 1.0
        masks.append(mask)

    return masks
