# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.bbox import bbox_overlaps


import torch
from torch.autograd import Variable


def points2d_transform(ex_rois, points2d):
  ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
  ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

  numi = [0, 2, 4, 6, 8, 10, 12, 14, 16]
  numj = [1, 3, 5, 7, 9, 11, 13, 15, 17]
  for i in numi:
    points2d[:, i] = (points2d[:, i] - ex_ctr_x) / (ex_widths * 2.0)
  for j in numj:
    points2d[:, j] = (points2d[:, j] - ex_ctr_y) / (ex_heights * 2.0)
  # print(points2d)
  # fang[-1]

  return points2d

def center_transform(ex_rois, center):
  ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
  ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

  gt_center_x = center[:, 0]
  gt_center_y = center[:, 1]

  targets_dx = (gt_center_x - ex_ctr_x) / (ex_widths)
  targets_dy = (gt_center_y - ex_ctr_y) / (ex_heights)

  # print(torch.max(targets_dx), torch.max(targets_dy))

  targets = torch.stack((targets_dx, targets_dy), 1)
  
  return targets

def points_transform(ex_rois, points):
  # print(front_4points)
  ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
  ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights


  gt_ctr_x = points[:, 0]
  gt_ctr_y = points[:, 1]
  gt_widths = torch.abs(points[:, 0] - points[:, 2])
  gt_heights = torch.abs(points[:, 1] - points[:, 3])

  targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
  targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights

  targets_dw = torch.log(gt_widths / ex_widths)
  targets_dh = torch.log(gt_heights / ex_heights)
  # print(front_4points)
  # fang[-1]
  targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), 1)

  return targets

def back_4points_transform(ex_rois, back_4points):
  ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
  ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

  # print(front_4points)
  # fang[-1]
  back_4points[:, 0] = (back_4points[:, 0] - ex_ctr_x) / (ex_widths * 1.0 * 0.5)
  back_4points[:, 2] = (back_4points[:, 2] - ex_ctr_x) / (ex_widths * 1.0 * 0.5)
  back_4points[:, 4] = (back_4points[:, 4] - ex_ctr_x) / (ex_widths * 1.0 * 0.5)
  back_4points[:, 6] = (back_4points[:, 6] - ex_ctr_x) / (ex_widths * 1.0 * 0.5)

  back_4points[:, 1] = (back_4points[:, 1] - ex_ctr_y) / (ex_heights * 1.0 * 0.5)
  back_4points[:, 3] = (back_4points[:, 3] - ex_ctr_y) / (ex_heights * 1.0 * 0.5)
  back_4points[:, 5] = (back_4points[:, 5] - ex_ctr_y) / (ex_heights * 1.0 * 0.5)
  back_4points[:, 7] = (back_4points[:, 7] - ex_ctr_y) / (ex_heights * 1.0 * 0.5)

  return back_4points


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # print(all_rois.shape)
  # print(gt_boxes.shape)
  # fang[-1]

  # Include ground-truth boxes in the set of candidate rois
  if cfg.TRAIN.USE_GT:
    zeros = rpn_rois.data.new(gt_boxes.shape[0], 1)
    all_rois = torch.cat(
      # (all_rois, torch.cat((zeros, gt_boxes[:, :-1]), 1))
      (all_rois, torch.cat((zeros, gt_boxes[:, :5]), 1))
    , 0)
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = torch.cat((all_scores, zeros), 0)

  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  fg_rois_per_image = int(round(cfg.TRAIN.FG_FRACTION * rois_per_image))

  # Sample rois with classification labels and bounding box regression
  # targets

  labels, rois, roi_scores, bbox_targets, bbox_inside_weights, front_2_1_points_targets, front_2_2_points_targets, \
      front_center_targets, back_2_1_points_targets, back_2_2_points_targets, back_center_targets, center_targets, \
        front_center_inside_weights = _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, \
            rois_per_image, _num_classes)

  rois = rois.view(-1, 5)
  roi_scores = roi_scores.view(-1)
  labels = labels.view(-1, 1)
  bbox_targets = bbox_targets.view(-1, _num_classes * 4)


  bbox_inside_weights = bbox_inside_weights.view(-1, _num_classes * 4)
  bbox_outside_weights = (bbox_inside_weights > 0).float()
  
  front_2_1_points_targets = front_2_1_points_targets.view(-1, _num_classes * 4)
  front_2_2_points_targets = front_2_2_points_targets.view(-1, _num_classes * 4)
  front_center_targets = front_center_targets.view(-1, _num_classes * 2)

  back_2_1_points_targets = back_2_1_points_targets.view(-1, _num_classes * 4)
  back_2_2_points_targets = back_2_2_points_targets.view(-1, _num_classes * 4)
  back_center_targets = back_center_targets.view(-1, _num_classes * 2)

  center_targets = center_targets.view(-1, _num_classes * 2)

  front_center_inside_weights = front_center_inside_weights.view(-1, _num_classes * 2)
  front_center_outside_weights = (front_center_inside_weights > 0).float()
  
  return rois, roi_scores, labels, Variable(bbox_targets), Variable(bbox_inside_weights), \
          Variable(bbox_outside_weights), Variable(front_2_1_points_targets), Variable(front_2_2_points_targets), Variable(front_center_targets),\
          Variable(back_2_1_points_targets), Variable(back_2_2_points_targets), Variable(back_center_targets), Variable(center_targets), \
          Variable(front_center_inside_weights), Variable(front_center_outside_weights)

def _get_bbox_regression_labels(bbox_target_data, num_classes, front_2_1_points_targets_data, front_2_2_points_targets_data, front_center_targets_data, back_2_1_points_targets_data, back_2_2_points_targets_data, back_center_targets_data, center_targets_data):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """
  # Inputs are tensor

  clss = bbox_target_data[:, 0]
  bbox_targets = clss.new(clss.numel(), 4 * num_classes).zero_()
  
  front_2_1_points_targets = clss.new(clss.numel(), 4 * num_classes).zero_()
  front_2_2_points_targets = clss.new(clss.numel(), 4 * num_classes).zero_()
  front_center_targets = clss.new(clss.numel(), 2 * num_classes).zero_()

  back_2_1_points_targets = clss.new(clss.numel(), 4 * num_classes).zero_()
  back_2_2_points_targets = clss.new(clss.numel(), 4 * num_classes).zero_()
  back_center_targets = clss.new(clss.numel(), 2 * num_classes).zero_()

  center_targets = clss.new(clss.numel(), 2 * num_classes).zero_()

  front_center_inside_weights = clss.new(front_center_targets.shape).zero_()

  bbox_inside_weights = clss.new(bbox_targets.shape).zero_()
  
  inds = (clss > 0).nonzero().view(-1)
  if inds.numel() > 0:
    clss = clss[inds].contiguous().view(-1,1)

    dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
    dim2_inds = torch.cat([4*clss, 4*clss+1, 4*clss+2, 4*clss+3], 1).long()
    # print(dim2_inds) # e.g. dim 16 * 4

    dim3_inds = inds.unsqueeze(1).expand(inds.size(0), 2)
    dim4_inds = torch.cat([2*clss, 2*clss+1], 1).long()

    # fang[-1]
    bbox_targets[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 1:]

    front_2_1_points_targets[dim1_inds, dim2_inds] = front_2_1_points_targets_data[inds][:, 0:]
    front_2_2_points_targets[dim1_inds, dim2_inds] = front_2_2_points_targets_data[inds][:, 0:]
    front_center_targets[dim3_inds, dim4_inds] = front_center_targets_data[inds][:, 0:]

    back_2_1_points_targets[dim1_inds, dim2_inds] = back_2_1_points_targets_data[inds][:, 0:]
    back_2_2_points_targets[dim1_inds, dim2_inds] = back_2_2_points_targets_data[inds][:, 0:]
    back_center_targets[dim3_inds, dim4_inds] = back_center_targets_data[inds][:, 0:]

    center_targets[dim3_inds, dim4_inds] = center_targets_data[inds][:, 0:]

    bbox_inside_weights[dim1_inds, dim2_inds] = bbox_targets.new(cfg.TRAIN.BBOX_INSIDE_WEIGHTS).view(-1, 4).expand_as(dim1_inds)

    front_center_inside_weights[dim3_inds, dim4_inds] = front_center_targets.new(cfg.TRAIN.CENTER_INSIDE_WEIGHTS).view(-1, 2).expand_as(dim3_inds)

  return bbox_targets, bbox_inside_weights, front_2_1_points_targets, front_2_2_points_targets, front_center_targets, back_2_1_points_targets, \
                  back_2_2_points_targets, back_center_targets, center_targets, front_center_inside_weights


def _compute_targets(ex_rois, gt_rois, labels, front_2_1_points, front_2_2_points, front_center_points, back_2_1_points, back_2_2_points, back_center_points, center_points):
  """Compute bounding-box regression targets for an image."""
  # Inputs are tensor

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4
  # print(gt_rois)
  # fang[-1]

  targets = bbox_transform(ex_rois, gt_rois)

  front_2_1_points_targets = points_transform(ex_rois, front_2_1_points)
  front_2_2_points_targets = points_transform(ex_rois, front_2_2_points)
  front_center_targets = center_transform(ex_rois, front_center_points)

  back_2_1_points_targets = points_transform(ex_rois, back_2_1_points)
  back_2_2_points_targets = points_transform(ex_rois, back_2_2_points)
  back_center_targets = center_transform(ex_rois, back_center_points)

  center_targets = center_transform(ex_rois, center_points)

  # print(targets)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))

    front_2_1_points_targets = ((front_2_1_points_targets - front_2_1_points_targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / front_2_1_points_targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    front_2_2_points_targets = ((front_2_2_points_targets - front_2_2_points_targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / front_2_2_points_targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    front_center_targets = ((front_center_targets - front_center_targets.new(cfg.TRAIN.CENTER_NORMALIZE_MEANS))
               / front_center_targets.new(cfg.TRAIN.CENTER_NORMALIZE_STDS))
    
    back_2_1_points_targets = ((back_2_1_points_targets - back_2_1_points_targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / back_2_1_points_targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    back_2_2_points_targets = ((back_2_2_points_targets - back_2_2_points_targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / back_2_2_points_targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    back_center_targets = ((back_center_targets - back_center_targets.new(cfg.TRAIN.CENTER_NORMALIZE_MEANS))
               / back_center_targets.new(cfg.TRAIN.CENTER_NORMALIZE_STDS))

    center_targets = ((center_targets - center_targets.new(cfg.TRAIN.CENTER_NORMALIZE_MEANS))
               / center_targets.new(cfg.TRAIN.CENTER_NORMALIZE_STDS))
  
  return torch.cat(
    [labels.unsqueeze(1), targets], 1), front_2_1_points_targets, front_2_2_points_targets, front_center_targets, \
        back_2_1_points_targets, back_2_2_points_targets, back_center_targets, center_targets

def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """

  # print(gt_boxes)
  # fang[-1] ok

  # overlaps: (rois x gt_boxes)
  overlaps = bbox_overlaps(
    all_rois[:, 1:5].data,
    gt_boxes[:, :4].data)
  max_overlaps, gt_assignment = overlaps.max(1)
  labels = gt_boxes[gt_assignment, [4]]

  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = (max_overlaps >= cfg.TRAIN.FG_THRESH).nonzero().view(-1)
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  bg_inds = ((max_overlaps < cfg.TRAIN.BG_THRESH_HI) + (max_overlaps >= cfg.TRAIN.BG_THRESH_LO) == 2).nonzero().view(-1)

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.numel() > 0 and bg_inds.numel() > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.numel())
    fg_inds = fg_inds[torch.from_numpy(npr.choice(np.arange(0, fg_inds.numel()), size=int(fg_rois_per_image), replace=False)).long().cuda()]
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    to_replace = bg_inds.numel() < bg_rois_per_image
    bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(bg_rois_per_image), replace=to_replace)).long().cuda()]
  elif fg_inds.numel() > 0:
    to_replace = fg_inds.numel() < rois_per_image
    fg_inds = fg_inds[torch.from_numpy(npr.choice(np.arange(0, fg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long().cuda()]
    fg_rois_per_image = rois_per_image
  elif bg_inds.numel() > 0:
    to_replace = bg_inds.numel() < rois_per_image
    bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long().cuda()]
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()


  # The indices that we're selecting (both fg and bg)
  keep_inds = torch.cat([fg_inds, bg_inds], 0)
  
  # Select sampled values from various arrays:
  labels = labels[keep_inds].contiguous()
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image):] = 0
  # print(int(fg_rois_per_image))  -> 16

  rois = all_rois[keep_inds].contiguous()
  roi_scores = all_scores[keep_inds].contiguous()



  bbox_target_data, front_2_1_points_targets_data, front_2_2_points_targets_data, front_center_targets_data, \
      back_2_1_points_targets_data, back_2_2_points_targets_data, back_center_targets_data, center_targets_data\
        = _compute_targets(rois[:, 1:5].data, gt_boxes[gt_assignment[keep_inds]][:, :4].data, labels.data,\
          gt_boxes[gt_assignment[keep_inds]][:, 5:9].data, gt_boxes[gt_assignment[keep_inds]][:, 9:13].data, \
          gt_boxes[gt_assignment[keep_inds]][:, 13:15].data, gt_boxes[gt_assignment[keep_inds]][:, 15:19].data, \
          gt_boxes[gt_assignment[keep_inds]][:, 19:23].data, gt_boxes[gt_assignment[keep_inds]][:, 23:25].data, \
          gt_boxes[gt_assignment[keep_inds]][:, 25:27].data)

  bbox_targets, bbox_inside_weights, front_2_1_points_targets, front_2_2_points_targets, front_center_targets, \
      back_2_1_points_targets, back_2_2_points_targets, back_center_targets, center_targets, front_center_inside_weights \
          = _get_bbox_regression_labels(bbox_target_data, num_classes, front_2_1_points_targets_data, front_2_2_points_targets_data, \
              front_center_targets_data, back_2_1_points_targets_data, back_2_2_points_targets_data, back_center_targets_data, center_targets_data)
  
  

  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights, front_2_1_points_targets, front_2_2_points_targets, front_center_targets, \
            back_2_1_points_targets, back_2_2_points_targets, back_center_targets, center_targets, front_center_inside_weights