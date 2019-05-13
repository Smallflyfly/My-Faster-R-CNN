# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from model.nms_wrapper import nms
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

import torch

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def points2d_tansform_inv(boxes, points2d):
  # if len(boxes) == 0:
  #   return deltas.detach() * 0

  widths = boxes[:, 2] - boxes[:, 0] + 1.0
  heights = boxes[:, 3] - boxes[:, 1] + 1.0
  # print(widths)
  # print(heights)
  # fang[-1]
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights
  # points2d_a = points2d[:, 0::2]
  # points2d_b = points2d[:, 1::2]

  # points2d_a = points2d_a * widths + ctr_x
  # points2d_b = points2d_b * heights + ctr_y
  points2d_tmp = points2d.copy()
  numi = [0, 2, 4, 6, 8, 10, 12, 14, 16]
  numj = [1, 3, 5, 7, 9, 11, 13, 15, 17]
  for i in numi:
    points2d[:, i] = points2d_tmp[:, i] * widths * 2.0 + ctr_x
  for j in numj:
    points2d[:, j] = points2d_tmp[:, j] * heights * 2.0 + ctr_y

  # print(widths.shape)
  # print(ctr_x.shape)
  # print(points2d[:, 0::2].shape)
  # fang[-1] 

  # points2d_pred = np.zeros(points2d.shape).astype(float)

  # points2d_pred[:, 0::2] = points2d[:, 0::2] * widths + ctr_x
  # points2d_pred[:, 1::2] = points2d[:, 1::2] * heights + ctr_y

  return points2d

def center_tansform_inv(boxes, front_center):
  # if len(boxes) == 0:
  #   return deltas.detach() * 0

  widths = boxes[:, 2] - boxes[:, 0] + 1.0
  heights = boxes[:, 3] - boxes[:, 1] + 1.0
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  dx = front_center[:, 0::2]
  dy = front_center[:, 1::2]

  pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
  pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)

  front_center_pred = torch.cat(\
    [_.unsqueeze(2) for _ in [pred_ctr_x,\
                              pred_ctr_y]], 2).view(len(boxes), -1)

  return front_center_pred

def points_transform_inv(boxes, front_2points):
  widths = boxes[:, 2] - boxes[:, 0] + 1.0
  heights = boxes[:, 3] - boxes[:, 1] + 1.0
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  dx = front_2points[:, 0::4]
  dy = front_2points[:, 1::4]
  dw = front_2points[:, 2::4]
  dh = front_2points[:, 3::4]

  pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
  pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
  pred_w = torch.exp(dw) * widths.unsqueeze(1)
  pred_h = torch.exp(dh) * heights.unsqueeze(1)

  front_2points_pred = torch.cat(\
    [_.unsqueeze(2) for _ in [pred_ctr_x,\
                              pred_ctr_y,\
                              pred_w,\
                              pred_h]], 2).view(len(boxes), -1)

  return front_2points_pred

def back_4points_transform_inv(boxes, back_4points):
  widths = boxes[:, 2] - boxes[:, 0] + 1.0
  heights = boxes[:, 3] - boxes[:, 1] + 1.0
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  back_4points[:, 0] = back_4points[:, 0] * widths * 0.5 + ctr_x
  back_4points[:, 2] = back_4points[:, 2] * widths * 0.5 + ctr_x
  back_4points[:, 4] = back_4points[:, 4] * widths * 0.5 + ctr_x
  back_4points[:, 6] = back_4points[:, 6] * widths * 0.5 + ctr_x

  back_4points[:, 1] = back_4points[:, 1] * heights * 0.5 + ctr_y
  back_4points[:, 3] = back_4points[:, 3] * heights * 0.5 + ctr_y
  back_4points[:, 5] = back_4points[:, 5] * heights * 0.5 + ctr_y
  back_4points[:, 7] = back_4points[:, 7] * heights * 0.5 + ctr_y

  return back_4points


def im_detect(net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  # _, scores, bbox_pred, rois, points2d_pred = net.test_image(blobs['data'], blobs['im_info'])
  _, scores, bbox_pred, rois, front_2_1_points_pred, front_2_2_points_pred, front_center_pred, back_2_1_points_pred, \
                              back_2_2_points_pred, back_center_pred, center_pred = net.test_image(blobs['data'], blobs['im_info'])
  # _, scores, bbox_pred, rois, center_pred = net.test_image(blobs['data'], blobs['im_info'])``
  
  # print(bbox_pred.shape)
  # print(points2d_pred.shape)
  # print('*************************************')
  # print(scores.shape[0])
  # fang[-1]
  # print(im_scales[0])
  # fang[-1]

  boxes = rois[:, 1:5] / im_scales[0]
  # print(im_scales)
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])

  front_2_1_points_pred = np.reshape(front_2_1_points_pred, [front_2_1_points_pred.shape[0], -1])
  front_2_2_points_pred = np.reshape(front_2_2_points_pred, [front_2_2_points_pred.shape[0], -1])
  front_center_pred = np.reshape(front_center_pred, [front_center_pred.shape[0], -1])

  back_2_1_points_pred = np.reshape(back_2_1_points_pred, [back_2_1_points_pred.shape[0], -1])
  back_2_2_points_pred = np.reshape(back_2_2_points_pred, [back_2_2_points_pred.shape[0], -1])
  back_center_pred = np.reshape(back_center_pred, [back_center_pred.shape[0], -1])

  center_pred = np.reshape(center_pred, [center_pred.shape[0], -1])

  if cfg.TEST.BBOX_REG:

    front_2_1_points_pred = points_transform_inv(torch.from_numpy(boxes), torch.from_numpy(front_2_1_points_pred)).numpy()
    front_2_2_points_pred = points_transform_inv(torch.from_numpy(boxes), torch.from_numpy(front_2_2_points_pred)).numpy()
    front_center_pred = center_tansform_inv(torch.from_numpy(boxes), torch.from_numpy(front_center_pred)).numpy()

    back_2_1_points_pred = points_transform_inv(torch.from_numpy(boxes), torch.from_numpy(back_2_1_points_pred)).numpy()
    back_2_2_points_pred = points_transform_inv(torch.from_numpy(boxes), torch.from_numpy(back_2_2_points_pred)).numpy()
    back_center_pred = center_tansform_inv(torch.from_numpy(boxes), torch.from_numpy(back_center_pred)).numpy()

    center_pred = center_tansform_inv(torch.from_numpy(boxes), torch.from_numpy(center_pred)).numpy()

    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
    
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  # return scores, pred_boxes, points2d_pred
  return scores, pred_boxes, front_2_1_points_pred, front_2_2_points_pred, front_center_pred, back_2_1_points_pred, \
            back_2_2_points_pred, back_center_pred, center_pred
  # return scores, pred_boxes, center_pred

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(torch.from_numpy(dets), thresh).numpy()
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_net(net, imdb, weights_filename, max_per_image=100, thresh=0.):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    scores, boxes = im_detect(net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(torch.from_numpy(cls_dets), cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time(),
            _t['misc'].average_time()))

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

