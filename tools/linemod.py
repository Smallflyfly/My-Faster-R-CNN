#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# Edited by Matthew Seals
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import argparse
from matplotlib import cm

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from MeshPly import Meshply

import torch

CLASSES = ('__background__',
           'ape')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',), 'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

COLORS = [cm.tab10(i) for i in np.linspace(0., 1., 10)]

def show_result_3points(frontp1, frontp2, fcenterp, backp1, backp2, bcenterp, centerp, img):
    # img = cv2.imread(filename)
    x1 = frontp1[0]
    y1 = frontp1[1]
    dw1 = frontp1[2]
    dh1 = frontp1[3]

    cx1 = fcenterp[0]
    cy1 = fcenterp[1]

    x3 = frontp2[0]
    y3 = frontp2[1]
    dw2 = frontp2[2]
    dh2 = frontp2[3]
    # point1, point4
    if cx1 > x1:
        x4 = x1 + dw1
        if cy1 > y1:
            y4 = y1 + dh1
        else:
            y4 = y1 - dh1
    else:
        x4 = x1 - dw1
        if cy1 > y1:
            y4 = y1 + dh1
        else:
            y4 = y1 - dh1
    
    # point3, points2
    if cx1 > x3:
        x2 = x3 + dw2
        if cy1 > y3:
            y2 = y3 + dh2
        else:
            y2 = y3 - dh2
    else:
        x2 = x3 - dw2
        if cy1 > y3:
            y2 = y3 + dh2
        else:
            y2 = y3 - dh2

    # cv2.line(img, (x1,y1), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x2, y2), 255, 2)
    # cv2.line(img, (x1,y1), (x2, y2), 255, 2)
    # cv2.line(img, (x1, y1), (x3, y3), 255, 2)
    # cv2.line(img, (x2,y2), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x4, y4), 255, 2)

    x5 = backp1[0]
    y5 = backp1[1]
    dw5 = backp1[2]
    dh5 = backp1[3]

    cx2 = bcenterp[0]
    cy2 = bcenterp[1]

    x7 = backp2[0]
    y7 = backp2[1]
    dw6 = backp2[2]
    dh6 = backp2[3]
    # point1, point4
    if cx2 > x5:
        x8 = x5 + dw5
        if cy2 > y5:
            y8 = y5 + dh5
        else:
            y8 = y5 - dh5
    else:
        x8 = x5 - dw5
        if cy2 > y5:
            y8 = y5 + dh5
        else:
            y8 = y5 - dh5
    
    # point3, points2
    if cx2 > x7:
        x6 = x7 + dw6
        if cy2 > y7:
            y6 = y7 + dh6
        else:
            y6 = y7 - dh6
    else:
        x6 = x7 - dw6
        if cy2 > y7:
            y6 = y7 + dh6
        else:
            y6 = y7 - dh6
    cx = centerp[0]
    cy = centerp[1]
    
    # cv2.line(img, (x1,y1), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x2, y2), 255, 2)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
    cv2.line(img, (x1, y1), (x3, y3), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x4, y4), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x4, y4), (255, 255, 0), 1)

    # cv2.line(img, (x5, y5), (x8, y8), 255, 2)
    # cv2.line(img, (x7, y7), (x6, y6), 255, 2)
    cv2.line(img, (x5, y5), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x5, y5), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x6, y6), (x8, y8), (255, 255, 0), 1)
    cv2.line(img, (x7, y7), (x8, y8), (255, 255, 0), 1)

    cv2.line(img, (x1, y1), (x5, y5), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (255, 255, 0), 1)
    
    # cv2.circle(img, (cx, cy), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx1, cy1), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx2, cy2), 2, (255, 0, 255), 1)
    
    pr_points = np.array([cx, cy, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8], float)
    # print(pr_points)
    # fang[-1]
    return pr_points

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # image_name = os.path.join(cfg.DATA_DIR, 'linemod_demo', image_name)
    # print(im_file)
    # fang[-1]
    im = cv2.imread(image_name)
    # print(im)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    # scores, boxes, points2d = im_detect(net, im)
    scores, boxes, front_2_1_points, front_2_2_points, front_center, back_2_1_points, back_2_2_points, back_center, center = im_detect(net, im)
    # print(np.max(front_4points))
    # print(np.max(back_4points))
    # print(np.max(center))
    # fang[-1]
    # scores, boxes, center = im_detect(net, im)
    timer.toc()
    # print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    thresh = 0.75  # CONF_THRESH
    NMS_THRESH = 0.3

    im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    cntr = -1

    img = cv2.imread(image_name)

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background

        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]

        cls_front_2_1_points = front_2_1_points[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_front_2_2_points = front_2_2_points[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_front_center = front_center[:, 2*cls_ind:2*(cls_ind + 1)]

        cls_back_2_1_points = back_2_1_points[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_back_2_2_points = back_2_2_points[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_back_center = back_center[:, 2*cls_ind:2*(cls_ind + 1)]

        cls_center = center[:, 2*cls_ind:2*(cls_ind + 1)]

        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]

        front_2_1_points_det = cls_front_2_1_points[keep.numpy(), :]
        front_2_2_points_det = cls_front_2_2_points[keep.numpy(), :]
        front_center_det = cls_front_center[keep.numpy(), :]

        back_2_1_points_det = cls_back_2_1_points[keep.numpy(), :]
        back_2_2_points_det = cls_back_2_2_points[keep.numpy(), :]
        back_center_det = cls_back_center[keep.numpy(), :]

        center_det = cls_center[keep.numpy(), :]

        inds = np.where(dets[:, -1] >= thresh)[0]
        # print('inds', inds)
        # fang[-1]
        inds = [0]
        if len(inds) == 0:
            continue
        else:
            cntr += 1
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            frontp1 = front_2_1_points_det[i, :]
            frontp2 = front_2_2_points_det[i, :]
            fcenterp = front_center_det[i, :]

            brontp1 = back_2_1_points_det[i, :]
            brontp2 = back_2_2_points_det[i, :]
            bcenterp = back_center_det[i, :]

            centerp = center_det[i, :]

            pr_points = show_result_3points(frontp1, frontp2, fcenterp, brontp1, brontp2, bcenterp, centerp, img)
            # show_result_3points(brontp1, brontp2, bcenterp, img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # fang[-1]
    return pr_points
            # ax.plot((center_x, center_y), s=1, c='b')

    #         ax.add_patch(
    #             plt.Rectangle((bbox[0], bbox[1]),
    #                           bbox[2] - bbox[0],
    #                           bbox[3] - bbox[1], fill=False,
    #                           edgecolor=COLORS[cntr % len(COLORS)], linewidth=3.5)
    #         )
    #         ax.text(bbox[0], bbox[1] - 2,
    #                 '{:s} {:.3f}'.format(cls, score),
    #                 bbox=dict(facecolor='blue', alpha=0.5),
    #                 fontsize=14, color='white')

    #     ax.set_title('All detections with threshold >= {:.1f}'.format(thresh), fontsize=14)

    #     plt.axis('off')
    #     plt.tight_layout()
    # plt.savefig('demo_' + image_name)
    # print('Saved to `{}`'.format(os.path.join(os.getcwd(), 'demo_' + image_name)))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args

def get_camera_intrinsic():
    K = np.ones((3, 3), dtype='float64')
    K[0, 0], K[0, 2] = 572.4114, 325.2611
    K[1, 1], K[1, 2] = 573.5704, 242.0489
    K[2, 2] = 1.

    return K

def get_3D_corners(vertices):
    
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])
    
    return corners

def compute_projection(points3d, R, K):
    projections_2d = np.zeros((2, points3d.shape[1]), float)
    camera_projection = (K.dot(R)).dot(points3d)
    projections_2d[0,:] = camera_projection[0,:] / camera_projection[2,:]
    projections_2d[1,:] = camera_projection[1,:] / camera_projection[2,:]

    return projections_2d

def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)

def test():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    
    imgs_path = '/media/smallflyfly/Others/faster-rcnn_9points_ape/data/VOCdevkit2007/VOC2007/JPEGImages/'
    labels_path = '/media/smallflyfly/Others/faster-rcnn_9points_ape/data/VOCdevkit2007/VOC2007/Annotations/'
    test_file_path = '/media/smallflyfly/Others/faster-rcnn_9points_ape/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
    fid = open(test_file_path, 'r')
    lines = fid.readlines()
    fid.close()
     # model path
    demonet = args.demo_net
    dataset = args.dataset

    saved_model = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                            NETS[demonet][0] % (70000 if dataset == 'pascal_voc' else 110000))

    print(saved_model)                           

    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(2, tag='default', anchor_scales=[8, 12, 16])  # class 7 #fang

    net.load_state_dict(torch.load(saved_model))

    net.eval()
    net.cuda()
    errs_2d = []
    errs_3d = []
    px_threshold = 5.
    eps = 1e-5
    count = 0
    diam = 0.103
    for line in lines:
        count += 1
        print('%d / %d' % (count, len(lines)))
        filename = line.strip()
        im_name = imgs_path + filename + '.jpg'
        pr_points = demo(net, im_name)
        label_fid = open(labels_path+filename+'.txt')
        line = label_fid.readlines()
        label_fid.close()
        label = line[0].strip().split()[1:-2]
        gt_points = np.array(label, float)
        gt_points[0::2] = gt_points[0::2] * 640.0
        gt_points[1::2] = gt_points[1::2] * 480.0

        gt_points = gt_points.reshape(9,2)
        pr_points = pr_points.reshape(9,2)
        K = get_camera_intrinsic()
        mesh = Meshply('./data/ape.ply')
        vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
        corners3D = get_3D_corners(vertices)
        corners3D = np.concatenate((np.zeros((1, 3), float), corners3D), axis=0)
        _, R_exp, t_exp = cv2.solvePnP(corners3D, gt_points, K, None)
        _, R_pre, t_pre = cv2.solvePnP(corners3D, pr_points, K, None)
        R_mat_exp, _ = cv2.Rodrigues(R_exp)
        R_mat_pre, _ = cv2.Rodrigues(R_pre)
        Rt_gt = np.concatenate((R_mat_exp, t_exp), axis=1)
        Rt_pr = np.concatenate((R_mat_pre, t_pre), axis=1)
        proj_2d_gt = compute_projection(vertices, Rt_gt, K)
        proj_2d_pr = compute_projection(vertices, Rt_pr, K)
        norm = np.linalg.norm(proj_2d_gt - proj_2d_pr, axis=0)
        pixel_dist = np.mean(norm)
        print(pixel_dist)
        if(pixel_dist > 5.0):
            print('!!!!!', filename)
        errs_2d.append(pixel_dist)

        # error 3d distance
        transform_3d_gt = compute_transformation(vertices, Rt_gt)
        transform_3d_pr = compute_transformation(vertices, Rt_pr)
        norm3d          = np.linalg.norm(transform_3d_gt - transform_3d_pr, axis=0)
        vertex_dist     = np.mean(norm3d)
        print(vertex_dist)
        errs_3d.append(vertex_dist)

    error_count = len(np.where(np.array(errs_2d) > px_threshold)[0])
    acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100.0 / (len(errs_2d)+eps)
    acc3d10     = len(np.where(np.array(errs_3d) <= diam * 0.1)[0]) * 100. / (len(errs_3d)+eps)
    print('Test finish!')
    print('Acc using {} px 2D projection = {:.2f}%'.format(px_threshold, acc))
    print('Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam * 0.1, acc3d10))

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset

    saved_model = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                            NETS[demonet][0] % (40000 if dataset == 'pascal_voc' else 110000))

    print(saved_model)                           

    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(2, tag='default', anchor_scales=[4, 8, 12])  # class 7 #fang

    net.load_state_dict(torch.load(saved_model))

    net.eval()
    net.cuda()

    print('Loaded network {:s}'.format(saved_model))

    im_names = [i for i in os.listdir('data/linemod_demo/')  # Pull in all jpgs
                if i.lower().endswith(".jpg")]
    print(im_names)                
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/linemod_demo/{}'.format(im_name))
        demo(net, im_name)

    # plt.show()
