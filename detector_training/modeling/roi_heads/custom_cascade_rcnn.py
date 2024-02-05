# modified from https://github.com/facebookresearch/CutLER/blob/main/cutler/modeling/roi_heads/custom_cascade_rcnn.py

from typing import List
import torch
from torch import nn
from torch.autograd.function import Function

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, pairwise_iou
from structures import pairwise_iou_max_scores
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference
from .roi_heads import ROI_HEADS_REGISTRY, CustomStandardROIHeads

import torch.nn.functional as F

class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


@ROI_HEADS_REGISTRY.register()
class CustomCascadeROIHeads(CustomStandardROIHeads):
    """
    The ROI heads that implement :paper:`Cascade R-CNN`.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        pred_level: bool = False,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_pooler (ROIPooler): pooler that extracts region features from given boxes
            box_heads (list[nn.Module]): box head for each cascade stage
            box_predictors (list[nn.Module]): box predictor for each cascade stage
            proposal_matchers (list[Matcher]): matcher with different IoU thresholds to
                match boxes with ground truth for each stage. The first matcher matches
                RPN proposals with ground truth, the other matchers use boxes predicted
                by the previous stage as proposals and match them with ground truth.
        """
        assert "proposal_matcher" not in kwargs, (
            "CustomCascadeROIHeads takes 'proposal_matchers=' for each stage instead "
            "of one 'proposal_matcher='."
        )
        # The first matcher matches RPN proposals with ground truth, done in the base class
        kwargs["proposal_matcher"] = proposal_matchers[0]
        num_stages = self.num_cascade_stages = len(box_heads)
        box_heads = nn.ModuleList(box_heads)
        box_predictors = nn.ModuleList(box_predictors)
        assert len(box_predictors) == num_stages, f"{len(box_predictors)} != {num_stages}!"
        assert len(proposal_matchers) == num_stages, f"{len(proposal_matchers)} != {num_stages}!"
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_heads,
            box_predictor=box_predictors,
            **kwargs,
        )
        self.proposal_matchers = proposal_matchers
        self.pred_level = pred_level

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.pop("proposal_matcher")
        ret["pred_level"] = cfg.MODEL.ROI_BOX_HEAD.PRED_LEVEL
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        assert len(cascade_bbox_reg_weights) == len(cascade_ious)
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CustomCascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )

        box_heads, box_predictors, proposal_matchers = [], [], []
        for match_iou, bbox_reg_weights in zip(cascade_ious, cascade_bbox_reg_weights):
            box_head = build_box_head(cfg, pooled_shape)
            box_heads.append(box_head)
            box_predictors.append(
                FastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights),
                )
            )
            proposal_matchers.append(Matcher([match_iou], [0, 1], allow_low_quality_matches=False))
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_heads": box_heads,
            "box_predictors": box_predictors,
            "proposal_matchers": proposal_matchers,
        }

    def forward(
        self,
        images,
        features,
        proposals,
        targets=None,
        branch: str = "",
        compute_loss: bool = True,
        compute_val_loss: bool = False,
    ):
        del images
        if self.training and compute_loss:
            assert targets, "'targets' argument is required during training"
            if targets[0].has("scores"):  # has confidence; then weight loss
                proposals = self.label_and_sample_proposals_pseudo(proposals, targets, branch)
            else:
                proposals = self.label_and_sample_proposals(proposals, targets, branch)
        elif compute_val_loss:
            assert targets, "'targets' argument is required during training"
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt

        if (self.training and compute_loss) or compute_val_loss:
            # Need targets to box head
            losses, _ = self._forward_box(features, proposals, targets,
                branch=branch, compute_loss=compute_loss, compute_val_loss=compute_val_loss)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(features, proposals,
                branch=branch, compute_loss=compute_loss, compute_val_loss=compute_val_loss)
            pred_instances = self.forward_with_given_boxes(features, pred_instances, compute_loss, compute_val_loss)
            return pred_instances, predictions

    def _forward_box(
        self,
        features,
        proposals,
        targets=None,
        branch: str = "",
        compute_loss: bool = True,
        compute_val_loss: bool = False,
    ):
        """
        Args:
            features, targets: the same as in
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        """
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are used to create the input
                # proposals of the next stage.
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes, compute_loss, compute_val_loss)
                if (self.training and compute_loss) or compute_val_loss:
                    if targets[0].has("scores"):
                        proposals = self._match_and_label_boxes_pseudo(proposals, k, targets, branch)
                    else:
                        proposals = self._match_and_label_boxes(proposals, k, targets, branch)
            predictions = self._run_stage(features, proposals, k, compute_loss, compute_val_loss)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        no_gt_found = False
        if (self.training and compute_loss) or compute_val_loss:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                no_gt_found = False
                with storage.name_scope("stage{}".format(stage)):
                    if self.use_droploss:
                        try:
                            box_num_list = [len(x.gt_boxes) for x in proposals]
                            gt_num_list = [torch.unique(x.gt_boxes.tensor[:100], dim=0).size()[0] for x in proposals]
                        except:
                            box_num_list = [0 for x in proposals]
                            gt_num_list = [0 for x in proposals]
                            no_gt_found = True

                        if not no_gt_found:
                            # NOTE: confidence score
                            prediction_score, predictions_delta = predictions[0], predictions[1]
                            prediction_score = F.softmax(prediction_score, dim=1)[:,0]

                            # NOTE: maximum overlapping with GT (IoU)
                            proposal_boxes = Boxes.cat([x.proposal_boxes for x in proposals])
                            predictions_bbox = predictor.box2box_transform.apply_deltas(predictions_delta, proposal_boxes.tensor)
                            idx_start = 0
                            iou_max_list = []
                            for idx, x in enumerate(proposals):
                                idx_end = idx_start + box_num_list[idx]
                                iou_max_list.append(pairwise_iou_max_scores(predictions_bbox[idx_start:idx_end], x.gt_boxes[:gt_num_list[idx]].tensor))
                                idx_start = idx_end
                            iou_max = torch.cat(iou_max_list, dim=0)

                            # NOTE: get the weight of each proposal
                            weights = iou_max.le(self.droploss_iou_thresh).float()
                            weights = 1 - weights.ge(1.0).float()
                            stage_losses = predictor.losses(predictions, proposals, branch=branch, weights=weights.detach())
                        else:
                            stage_losses = predictor.losses(predictions, proposals, branch=branch)
                    else:
                        stage_losses = predictor.losses(predictions, proposals, branch=branch)

                    if self.pred_level:
                        # record distribution and accuracy of levels
                        fg_mask = torch.cat([x.gt_classes for x in proposals]) < self.num_classes
                        gt_levels = torch.cat([x.gt_levels if x.has('gt_levels') else torch.zeros_like(x.gt_classes) for x in proposals])
                        gt_levels = gt_levels[fg_mask]
                        len_gt_levels = len(gt_levels)
                        if len_gt_levels == 0:
                            len_gt_levels += 1
                        gt_whole = (gt_levels == 0).sum().item() / len_gt_levels
                        gt_part = (gt_levels == 1).sum().item() / len_gt_levels
                        gt_subpart = (gt_levels == 2).sum().item() / len_gt_levels
                        pred_levels = predictions[2].argmax(dim=1)[fg_mask]
                        level_accuracy = (pred_levels == gt_levels).sum().item() / len_gt_levels
                        storage.put_scalar("roi_head/ratio_whole_{}".format(branch), gt_whole)
                        storage.put_scalar("roi_head/ratio_part_{}".format(branch), gt_part)
                        storage.put_scalar("roi_head/ratio_subpart_{}".format(branch), gt_subpart)
                        storage.put_scalar("roi_head/level_total_{}".format(branch), len_gt_levels)
                        storage.put_scalar("roi_head/level_accuracy_{}".format(branch), level_accuracy)

                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses, predictions
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]

            # Average the scores across heads
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]

            if self.pred_level:
                levels_per_stage = [h[0].predict_levels(h[1], h[2]) for h in head_outputs]
                levels = [
                    sum(list(levels_per_image)) * (1.0 / self.num_cascade_stages)
                    for levels_per_image in zip(*levels_per_stage)
                ]
            else:
                levels = None

            # Use the boxes of the last head
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
                levels=levels,
            )
            return pred_instances, predictions

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets, branch: str = ""):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.

        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances

        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes

            if self.pred_level:
                if len(targets_per_image) > 0:
                    gt_levels = targets_per_image.gt_levels[matched_idxs]
                    gt_levels[proposal_labels == 0] = -1
                else:
                    gt_levels = torch.zeros_like(matched_idxs) - 1
                proposals_per_image.gt_levels = gt_levels

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples_{}".format(stage, branch),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples_{}".format(stage, branch),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    @torch.no_grad()
    def _match_and_label_boxes_pseudo(self, proposals, stage, targets, branch: str = ""):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.

        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances

        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        gt_confids = [x.scores for x in targets]

        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image, confids_per_image in zip(proposals, targets, gt_confids):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
                gt_confid = confids_per_image[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
                gt_confid = torch.ones_like(matched_idxs)

            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes
            proposals_per_image.gt_confid = gt_confid

            if self.pred_level:
                if len(targets_per_image) > 0:
                    gt_levels = targets_per_image.gt_levels[matched_idxs]
                    gt_levels[proposal_labels == 0] = 0
                else:
                    gt_levels = torch.zeros_like(matched_idxs)
                proposals_per_image.gt_levels = gt_levels

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_target_fg_samples_{}".format(stage, branch),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_target_bg_samples_{}".format(stage, branch),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    def _run_stage(self, features, proposals, stage, compute_loss=True, compute_val_loss=False):
        """
        Args:
            features (list[Tensor]): #lvl input features to ROIHeads
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage

        Returns:
            Same output as `FastRCNNOutputLayers.forward()`.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # The original implementation averages the losses among heads,
        # but scale up the parameter gradients of the heads.
        # This is equivalent to adding the losses among heads,
        # but scale down the gradients on features.
        if (self.training and compute_loss) or compute_val_loss:
            box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        return self.box_predictor[stage](box_features)

    def _create_proposals_from_boxes(self, boxes, image_sizes, compute_loss=True, compute_val_loss=False):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if (self.training and compute_loss) or compute_val_loss:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals
