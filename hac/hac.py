# modified from https://github.com/facebookresearch/CutLER/blob/main/maskcut/maskcut.py

import os
import argparse
import numpy as np
import re
import PIL
import PIL.Image as Image
import torch
from torchvision import transforms
from pycocotools import mask
import pycocotools.mask as mask_util
from scipy import ndimage
import json
import time
import copy

import dino
from crf import densecrf_hac
from hier_cluster import hier_cluster

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])

def resize_pil(I, patch_size=16) : 
    w, h = I.size

    new_w, new_h = int(round(w / patch_size)) * patch_size, int(round(h / patch_size)) * patch_size
    feat_w, feat_h = new_w // patch_size, new_h // patch_size

    return I.resize((new_w, new_h), resample=Image.LANCZOS), w, h, feat_w, feat_h

def hac(img_path, backbone, patch_size, thresh, fixed_size=480, cpu=False) :
    I = Image.open(img_path).convert('RGB')

    I_new = I.resize((int(fixed_size), int(fixed_size)), PIL.Image.LANCZOS)
    I_resize, w, h, feat_w, feat_h = resize_pil(I_new, patch_size)

    tensor = ToTensor(I_resize).unsqueeze(0)
    if not cpu: tensor = tensor.cuda()
    feat = backbone(tensor)[0]
    feat = feat.view(-1, feat_h, feat_w)

    masks = hier_cluster(feat, thresh)

    return masks, I_new

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def create_image_info(image_id, file_name, image_size,
                      license_id=1):
    """Return image_info in COCO style
    Args:
        image_id: the image ID
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        license: license of this image
    """
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[1],
            "height": image_size[0],
            "license": license_id,
    }
    return image_info

def create_annotation_info(annotation_id, image_id, category_info, binary_mask, 
                           image_size=None, bounding_box=None):
    """Return annotation info in COCO style
    Args:
        annotation_id: the annotation ID
        image_id: the image ID
        category_info: the information on categories
        binary_mask: a 2D binary numpy array where '1's represent the object
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        bounding_box: the bounding box for detection task. If bounding_box is not provided, 
        we will generate one according to the binary mask.
    """
    upper = np.max(binary_mask)
    lower = np.min(binary_mask)
    thresh = upper / 2.0
    binary_mask[binary_mask > thresh] = upper
    binary_mask[binary_mask <= thresh] = lower
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask.astype(np.uint8), image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    segmentation = rle

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    } 

    return annotation_info

# necessay info used for coco style annotations
INFO = {
    "description": "Pseudo-labels of object masks by Hierarchical Adaptive Clustering",
    "url": "https://github.com/Shengcao-Cao/HASSOD",
    "version": "1.0",
    "year": 2023,
    "contributor": "Shengcao Cao",
}

LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
        "url": "https://github.com/Shengcao-Cao/HASSOD/blob/main/LICENSE"
    }
]

# only one class, i.e. foreground
CATEGORIES = [
    {
        'id': 1,
        'name': 'fg',
        'supercategory': 'fg',
    },
]

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []}

category_info = {
    "is_crowd": 0,
    "id": 1
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Hierarchical Adaptive Clustering')
    # default arguments
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')
    parser.add_argument('--nb-vis', type=int, default=20, choices=[1, 200], help='nb of visualization')
    parser.add_argument('--img-path', type=str, default=None, help='single image visualization')

    # additional arguments
    parser.add_argument('--dataset-path', type=str, default="coco/train2017/", help='path to the dataset')
    parser.add_argument('--thresh', type=float, default=[0.1], nargs='+', help='merging threshold(s)')
    parser.add_argument('--num-image-per-job', type=int, default=1, help='the number of images each job processes')
    parser.add_argument('--job-index', type=int, default=0, help='the index of the job')
    parser.add_argument('--fixed_size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--cpu', action='store_true', help='use cpu')

    args = parser.parse_args()

    if args.pretrain_path is not None:
        url = args.pretrain_path
    if args.vit_arch == 'base' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == 'small' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        feat_dim = 384
    elif args.vit_arch == 'base' and args.patch_size == 16:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == 'small' and args.patch_size == 16:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        feat_dim = 384

    backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)

    backbone.eval()
    if not args.cpu:
        backbone.cuda()

    img_names = sorted(os.listdir(args.dataset_path))

    if args.out_dir is not None and not os.path.exists(args.out_dir) :
        os.mkdir(args.out_dir)

    start_idx = max(args.job_index*args.num_image_per_job, 0)
    end_idx = min((args.job_index+1)*args.num_image_per_job, len(img_names))

    n_thresh = len(args.thresh)
    args.thresh = sorted(args.thresh)[::-1]
    outputs = [copy.deepcopy(output) for _ in range(n_thresh)]
    segmentation_ids = [1] * n_thresh

    print(f'Beginning Job {args.job_index}...', flush=True)
    t0 = time.time()
    for img_idx, img_name in enumerate(img_names[start_idx:end_idx]):
        if (img_idx + 1) % 100 == 0:
            print(f'Job {args.job_index} processing image {img_idx} / {end_idx - start_idx}, time {time.time() - t0}', flush=True)
        # get image path
        img_path = os.path.join(args.dataset_path, img_name)
        # get pseudo-masks for each image using MaskCut
        try:
            pseudo_masks, I_new = hac(img_path, backbone, args.patch_size, args.thresh, \
                fixed_size=args.fixed_size, cpu=args.cpu)
        except:
            print(f'Skipping {img_name}')
            continue

        I = Image.open(img_path).convert('RGB')
        width, height = I.size

        # create coco-style image info
        img_id = int(os.path.splitext(img_name)[0])
        image_info = create_image_info(
            img_id, img_name, (height, width, 3))

        for thresh_idx in range(n_thresh):
            output_i = outputs[thresh_idx]
            segmentation_id = segmentation_ids[thresh_idx]
            pseudo_mask = pseudo_masks[thresh_idx]

            output_i["images"].append(image_info)

            # filter out masks with little overlap before and after CRF
            pseudo_mask_precrf = torch.nn.functional.interpolate(torch.tensor(pseudo_mask).unsqueeze(0),
                size=(args.fixed_size, args.fixed_size), mode="bilinear").squeeze(0).numpy()
            pseudo_mask_postcrf = densecrf_hac(np.array(I_new), pseudo_mask)
            mask1 = (pseudo_mask_precrf >= 0.5).astype(np.uint8)
            mask2 = (pseudo_mask_postcrf >= 0.5).astype(np.uint8)
            inter_mask = np.logical_and(mask1, mask2)
            union_mask = np.logical_or(mask1, mask2)
            keep_masks = ((inter_mask.sum(axis=1).sum(axis=1) /
                union_mask.sum(axis=1).sum(axis=1)) >= 0.5)

            if keep_masks.sum() == 0:
                continue
            pseudo_mask = pseudo_mask_postcrf[keep_masks]

            pseudo_mask = torch.nn.functional.interpolate(torch.tensor(pseudo_mask).unsqueeze(0),
                size=(height, width), mode="bilinear").squeeze(0).numpy()

            for idx in range(pseudo_mask.shape[0]):
                binary_mask_i = (pseudo_mask[idx] >= 0.5).astype(np.uint8)
                # filter out too small masks
                if binary_mask_i.sum() < 100:
                    continue
                # filter out three-corner masks
                if binary_mask_i[0, 0] + binary_mask_i[0, -1] + binary_mask_i[-1, 0] + binary_mask_i[-1, -1] >= 3:
                    continue
                # fill holes
                binary_mask_i = ndimage.binary_fill_holes(binary_mask_i)
                # create coco-style annotation info
                annotation_info = create_annotation_info(
                    segmentation_id, img_id, category_info, binary_mask_i, None)
                if annotation_info is not None:
                    output_i["annotations"].append(annotation_info)
                    segmentation_id += 1

            segmentation_ids[thresh_idx] = segmentation_id

            if (img_idx + 1) % 100 == 0:
                # save annotations
                if len(img_names) == args.num_image_per_job and args.job_index == 0:
                    json_name = '{}/coco_fixsize{}_thresh{}.json'.format(args.out_dir, args.fixed_size, args.thresh[thresh_idx])
                else:
                    json_name = '{}/coco_fixsize{}_thresh{}_{}_{}.json'.format(args.out_dir, args.fixed_size, args.thresh[thresh_idx], start_idx, end_idx)
                with open(json_name, 'w') as output_json_file:
                    json.dump(output_i, output_json_file, indent=2)
                print(f'dumping {json_name}', flush=True)

    for thresh_idx in range(n_thresh):
        output_i = outputs[thresh_idx]
        # save annotations
        if len(img_names) == args.num_image_per_job and args.job_index == 0:
            json_name = '{}/coco_fixsize{}_thresh{}.json'.format(args.out_dir, args.fixed_size, args.thresh[thresh_idx])
        else:
            json_name = '{}/coco_fixsize{}_thresh{}_{}_{}.json'.format(args.out_dir, args.fixed_size, args.thresh[thresh_idx], start_idx, end_idx)
        with open(json_name, 'w') as output_json_file:
            json.dump(output_i, output_json_file, indent=2)
        print(f'dumping {json_name}', flush=True)

        print("Job {} thresh {} done: {} images; {} anns.".format(args.job_index, args.thresh[thresh_idx], len(output_i['images']), len(output_i['annotations'])))
