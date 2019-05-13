# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg
import PIL
import matplotlib.pyplot as plt
import cv2


class pascal_voc(imdb):
  def __init__(self, image_set, year, use_diff=False):
    name = 'voc_' + year + '_' + image_set
    if use_diff:
      name += '_diff'
    imdb.__init__(self, name)
    self._year = year
    self._image_set = image_set
    self._devkit_path = self._get_default_path()
    self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
    # self._classes = ('__background__',  # always index 0
    #                 'chair', 'table', 'sofa', 'bed', 'shelf', 'cabinet')
    self._classes = ('__background__',  # always index 0
                    'ape')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.jpg'
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': use_diff,
                   'matlab_eval': False,
                   'rpn_file': None}

    assert os.path.exists(self._devkit_path), \
      'VOCdevkit path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, 'JPEGImages',
                              index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_pascal_annotation(index)
                for index in self.image_index]
    # print('*******************************')
    # print(gt_roidb)
    # fang[-1]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def get_pic_size(self, filename):
    w = PIL.Image.open(filename).size[0]
    h = PIL.Image.open(filename).size[1]
    return w, h
  
  def Show_3D_box(self, img, points, gt_front_center, gt_back_center, center):
    im = cv2.imread(img)

    for i in range(9):
      ii = 2 * i
      jj = 2 * i + 1
      cv2.circle(im, (int(points[ii]), int(points[jj])), 2, (255, 0, 255), 2)
    cv2.circle(im, (int(gt_front_center[0,0]), int(gt_front_center[0,1])), 2, (255, 0, 255), 2)
    cv2.circle(im, (int(gt_back_center[0,0]), int(gt_back_center[0,1])), 2, (255, 0, 255), 2)
    cv2.circle(im, (int(center[0,0]), int(center[0,1])), 2, (255, 0, 255), 2)
    cv2.imshow('im', im)
    cv2.waitKey()
    cv2.destroyAllWindows()

  def _load_pascal_annotation(self, index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    # filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
    filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
    picname = os.path.join(self._data_path, 'JPEGImages', index + '.jpg') # pci with  height
    w , h = self.get_pic_size(picname)
    # tree = ET.parse(filename)
    # objs = tree.findall('object')
    # if not self.config['use_diff']:
    #   # Exclude the samples labeled as difficult
    #   non_diff_objs = [
    #     obj for obj in objs if int(obj.find('difficult').text) == 0]
    #   # if len(non_diff_objs) != len(objs):
    #   #     print 'Removed {} difficult objects'.format(
    #   #         len(objs) - len(non_diff_objs))
    #   objs = non_diff_objs
    # num_objs = len(objs)
    num_objs = 1

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # gt_points2d = np.zeros((num_objs, 18), dtype=np.float32) #8 points + 1 center
    # gt_points2d_tmp = np.zeros((num_objs, 16), dtype=np.float32)
    # gt_center = np.zeros((num_objs, 2), dtype=np.float32)
    gt_front_4points = np.zeros((num_objs, 8), dtype = np.float32)
    gt_back_4points = np.zeros((num_objs, 8), dtype = np.float32)



    gt_front_2_1_points = np.zeros((num_objs, 4)).astype(float)
    gt_front_2_2_points = np.zeros((num_objs, 4)).astype(float)
    gt_front_center = np.zeros((num_objs, 2)).astype(float)
    gt_back_2_1_points = np.zeros((num_objs, 4)).astype(float)
    gt_back_2_2_points = np.zeros((num_objs, 4)).astype(float)
    gt_back_center = np.zeros((num_objs, 2)).astype(float)
    gt_center = np.zeros((num_objs, 2)).astype(float)

    # Load object bounding boxes into a data frame.
    fid = open(filename, 'r')
    label = fid.readlines()[0].strip().split()
    label = label[1:-2]
    points = np.array(label, float)
    points[0::2] = points[0::2] * w
    points[1::2] = points[1::2] * h

    x1 = np.min(points[0::2])
    y1 = np.min(points[1::2])
    x2 = np.max(points[0::2])
    y2 = np.max(points[1::2])

    cx = points[0]
    cy = points[1]
    px1 = points[2]
    py1 = points[3]
    px2 = points[4]
    py2 = points[5]
    px3 = points[6]
    py3 = points[7]
    px4 = points[8]
    py4 = points[9]
    px5 = points[10]
    py5 = points[11]
    px6 = points[12]
    py6 = points[13]
    px7 = points[14]
    py7 = points[15]
    px8 = points[16]
    py8 = points[17]
    ix = 0
    gt_front_4points[ix, :] = [px1, py1, px2, py2, px3, py3, px4, py4]
    gt_back_4points[ix, :] = [px5, py5, px6, py6, px7, py7, px8, py8]
    gt_front_center[ix, :] = [np.mean(gt_front_4points[ix, 0::2]), np.mean(gt_front_4points[ix, 1::2])]
    gt_back_center[ix, :] = [np.mean(gt_back_4points[ix, 0::2]), np.mean(gt_back_4points[ix, 1::2])]
    
    if px1 == px4:
        px4 += 1
    if py1 == py4:
      py4 += 1
    gt_front_2_1_points[ix, :] = [px1, py1, px4, py4]

    if px2 == px3:
      px2 += 1
    if py2 == py3:
      py2 += 1
    gt_front_2_2_points[ix, :] = [px3, py3, px2, py2]

    if px5 == px8:
      px8 += 1
    if py5 == py8:
      py8 += 1
    gt_back_2_1_points[ix, :] = [px5, py5, px8, py8]

    if px6 == px7:
      px6 += 1
    if py6 == py7:
      py6 += 1
    gt_back_2_2_points[ix, :] = [px7, py7, px6, py6]
    
    gt_center[ix, :] = [cx, cy]
    
    # self.Show_3D_box(picname, points, gt_front_center, gt_back_center, center)
    # fang[-1]
      # print(index, px, py)
      # fang[-1]x1



 
    claname = 'ape'
    cls = self._class_to_ind[claname]
    boxes[ix, :] = [x1, y1, x2, y2]
    gt_classes[ix] = cls
    overlaps[ix, cls] = 1.0
    seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

      



    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_front_center': gt_front_center, #fang
            'gt_back_center': gt_back_center,
            'gt_front_2_1_points': gt_front_2_1_points,
            'gt_front_2_2_points': gt_front_2_2_points,
            'gt_back_2_1_points': gt_back_2_1_points,
            'gt_back_2_2_points': gt_back_2_2_points,
            'gt_center': gt_center,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    path = os.path.join(
      self._devkit_path,
      'results',
      'VOC' + self._year,
      'Main',
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} VOC results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def _do_python_eval(self, output_dir='output'):
    annopath = os.path.join(
      self._devkit_path,
      'VOC' + self._year,
      'Annotations',
      '{:s}.xml')
    imagesetfile = os.path.join(
      self._devkit_path,
      'VOC' + self._year,
      'ImageSets',
      'Main',
      self._image_set + '.txt')
    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(self._year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      filename = self._get_voc_results_file_template().format(cls)
      rec, prec, ap = voc_eval(
        filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
        use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
      aps += [ap]
      print(('AP for {} = {:.4f}'.format(cls, ap)))
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
      print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


if __name__ == '__main__':
  from datasets.pascal_voc import pascal_voc

  d = pascal_voc('trainval', '2007')
  res = d.roidb
  from IPython import embed;

  embed()
