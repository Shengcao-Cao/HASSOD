# modified from https://github.com/facebookresearch/CutLER/blob/main/maskcut/crf.py

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch

MAX_ITER = 10
POS_W = 7 
POS_XY_STD = 3
Bi_W = 10
Bi_XY_STD = 50 
Bi_RGB_STD = 5

def densecrf_hac(image, mask):
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    mask_interp = torch.nn.functional.interpolate(torch.tensor(mask).unsqueeze(0),
        size=(H, W), mode="bilinear").squeeze(0).numpy()

    c = mask_interp.shape[0]
    h = mask_interp.shape[1]
    w = mask_interp.shape[2]

    U = utils.unary_from_softmax(mask_interp)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q
