# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import random
import matplotlib.pyplot as plt

def show_3D_BBOX(im, gt_box):
  img = im.astype(np.uint8)
  for i in range(11):
    ii = 2*i + 5
    jj = 2*i + 1 + 5
    cv2.circle(img, (int(gt_box[0,ii]), int(gt_box[0,jj])), 2, (255, 0, 255), 2)
  cv2.imshow('im', img)
  cv2.waitKey()
  cv2.destroyAllWindows()
  # fang[-1]

def get_minibatch(roidb, num_classes, bg_file_names):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # print(roidb)
  # fang[-1]
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales, pleft, ptop = _get_image_blob(roidb, random_scale_inds, bg_file_names)

  blobs = {'data': im_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)      #fang fangpengfei  fang gt_boxes:(x1, y1, x2, y2, points2d1, points2d2 ... points2d19)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  # gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes = np.empty((len(gt_inds), 5 + 10 + 10 + 2), dtype=np.float32)    # add 19  for points2d #fang
  ################## box ########################################################
  # gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 0] = (roidb[0]['boxes'][gt_inds, 0] - pleft) * im_scales[0]
  gt_boxes[:, 1] = (roidb[0]['boxes'][gt_inds, 1] - ptop) * im_scales[0]
  gt_boxes[:, 2] = (roidb[0]['boxes'][gt_inds, 2] - pleft) * im_scales[0]
  gt_boxes[:, 3] = (roidb[0]['boxes'][gt_inds, 3] - ptop) * im_scales[0]

  ################## gt_classes ########################################################
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]

  ################## gt_front_2_1_points ########################################################
  # gt_boxes[:, 5:9] = roidb[0]['gt_front_2_1_points'][gt_inds, 0:] * im_scales[0]
  gt_boxes[:, 5] = (roidb[0]['gt_front_2_1_points'][gt_inds, 0] - pleft) * im_scales[0]
  gt_boxes[:, 6] = (roidb[0]['gt_front_2_1_points'][gt_inds, 1] - ptop) * im_scales[0]
  gt_boxes[:, 7] = (roidb[0]['gt_front_2_1_points'][gt_inds, 2] - pleft) * im_scales[0]
  gt_boxes[:, 8] = (roidb[0]['gt_front_2_1_points'][gt_inds, 3] - ptop) * im_scales[0]

  ################## gt_front_2_2_points ########################################################
  # gt_boxes[:, 9:13] = roidb[0]['gt_front_2_2_points'][gt_inds, 0:] * im_scales[0]
  gt_boxes[:,  9] = (roidb[0]['gt_front_2_2_points'][gt_inds, 0] - pleft) * im_scales[0]
  gt_boxes[:, 10] = (roidb[0]['gt_front_2_2_points'][gt_inds, 1] - ptop) * im_scales[0]
  gt_boxes[:, 11] = (roidb[0]['gt_front_2_2_points'][gt_inds, 2] - pleft) * im_scales[0]
  gt_boxes[:, 12] = (roidb[0]['gt_front_2_2_points'][gt_inds, 3] - ptop) * im_scales[0]
  
  ################## gt_front_center ########################################################
  # gt_boxes[:, 13:15] = roidb[0]['gt_front_center'][gt_inds, 0:] * im_scales[0]
  gt_boxes[:, 13] = (roidb[0]['gt_front_center'][gt_inds, 0] - pleft) * im_scales[0]
  gt_boxes[:, 14] = (roidb[0]['gt_front_center'][gt_inds, 1] - ptop) * im_scales[0]

  ################## gt_back_2_1_points ########################################################
  # gt_boxes[:, 15:19] = roidb[0]['gt_back_2_1_points'][gt_inds, 0:] * im_scales[0]
  gt_boxes[:, 15] = (roidb[0]['gt_back_2_1_points'][gt_inds, 0] - pleft) * im_scales[0]
  gt_boxes[:, 16] = (roidb[0]['gt_back_2_1_points'][gt_inds, 1] - ptop) * im_scales[0]
  gt_boxes[:, 17] = (roidb[0]['gt_back_2_1_points'][gt_inds, 2] - pleft) * im_scales[0]
  gt_boxes[:, 18] = (roidb[0]['gt_back_2_1_points'][gt_inds, 3] - ptop) * im_scales[0]

  ################## gt_back_2_2_points ########################################################
  # gt_boxes[:, 19:23] = roidb[0]['gt_back_2_2_points'][gt_inds, 0:] * im_scales[0]
  gt_boxes[:, 19] = (roidb[0]['gt_back_2_2_points'][gt_inds, 0] - pleft) * im_scales[0]
  gt_boxes[:, 20] = (roidb[0]['gt_back_2_2_points'][gt_inds, 1] - ptop) * im_scales[0]
  gt_boxes[:, 21] = (roidb[0]['gt_back_2_2_points'][gt_inds, 2] - pleft) * im_scales[0]
  gt_boxes[:, 22] = (roidb[0]['gt_back_2_2_points'][gt_inds, 3] - ptop) * im_scales[0]

  ################## gt_back_center ########################################################
  # gt_boxes[:, 23:25] = roidb[0]['gt_back_center'][gt_inds, 0:] * im_scales[0]
  gt_boxes[:, 23] = (roidb[0]['gt_back_center'][gt_inds, 0] - pleft) * im_scales[0]
  gt_boxes[:, 24] = (roidb[0]['gt_back_center'][gt_inds, 1] - ptop) * im_scales[0]

  ################## gt_back_center ########################################################
  gt_boxes[:, 25] = (roidb[0]['gt_center'][gt_inds, 0] - pleft) * im_scales[0]
  gt_boxes[:, 26] = (roidb[0]['gt_center'][gt_inds, 1] - ptop) * im_scales[0]

  # show_3D_BBOX(im_blob[0], gt_boxes)

  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
    dtype=np.float32)

  return blobs

def change_background(im, mask, bg_path):

  bg = cv2.imread(bg_path)
  bg = cv2.resize(bg, (im.shape[1], im.shape[0]))
  mask1 = mask
  mask[mask>0] = 1
  new_im = mask * im
  # cv2.imshow('im', new_im)
  # cv2.waitKey()
  # cv2.destroyAllWindows()
  # fang[-1]
  
  mask1[mask1>0] = 255
  tmp = np.ones((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8) * 255
  mask1 = tmp - mask1
  mask1 =  mask1 / 255
  new_bg = mask1.astype(np.uint8) * bg
  # cv2.imshow('bg', new_bg)
  # cv2.waitKey()
  # cv2.destroyAllWindows()
  # fang[-1]
  new_img = new_im + new_bg
  # cv2.imshow('new_img', new_img)
  # cv2.waitKey()
  # cv2.destroyAllWindows()
  # fang[-1]
  return new_img

def _get_image_blob(roidb, scale_inds, bg_file_names):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  mask_path = '/media/smallflyfly/Others/faster-rcnn_9points_ape/data/VOCdevkit2007/VOC2007/Mask/'
  # print(num_images)
  # fang[-1]
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    print(roidb[i]['image'])
    filename = roidb[i]['image'][-10:-4]
    # print(filename)
    # print(roidb[i]['image']) # /media/smallflyfly/Others/faster-rcnn_9points_anchor/data/VOCdevkit2007/VOC2007/JPEGImages/000653.jpg
    mask_name = mask_path + filename[2:] + '.png'
    mask = cv2.imread(mask_name)
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
      mask = mask[:, ::-1, :]
    # change_flag = random.randint(1, 1000000) % 2
    change_flag = 1
    # print(len(bg_file_names))  #17125
    ######### change background ###############################
    if change_flag == 1:
      random_bg_index = random.randint(0, len(bg_file_names)-1)
      bg_path = bg_file_names[random_bg_index]
      im = change_background(im, mask, bg_path)
      cv2.imshow('im',im)
      cv2.waitKey()
      cv2.destroyAllWindows()
      # fang[-1]

    jitter = 0.1
    ow, oh = im.shape[1], im.shape[0]
    dw = int(ow*jitter)
    dh = int(oh*jitter)

    pleft = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    pbot = random.randint(-dh, dh)

    if pleft < 0:
      pleft = 0
    if pright < 0:
      pright = 0
    if ptop < 0:
      ptop = 0
    if pbot < 0:
      pbot = 0
    
    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot
    
    im = im[ptop:ptop+sheight, pleft:pleft+swidth, :]
    # cv2.imshow('crop', im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # fang[-1]

    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales, pleft, ptop
