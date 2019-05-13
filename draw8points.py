import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import torch
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

# test_result =np.array([0.61146486, 0.35519063, 0.64516056, 0.24063933, 0.61253345, 0.3924165,
#   0.6619433,  0.26853168, 0.59920365, 0.35358822, 0.6299748,  0.25024676,
#   0.61231947, 0.39302433, 0.65594035, 0.27575362]).astype(float)

# test_result =np.array([0.49141535, 0.47574186, 0.49630564, 0.32518727, 0.41260856, 0.5957668,
#   0.4066204,  0.40914568, 0.594179,   0.51386803, 0.60663754, 0.3644182,
#   0.56371784, 0.65976834, 0.5829381,  0.4813519]).astype(float)

# test_result =np.array([0.2783297,  0.32261193, 0.26356098, 0.24497353, 0.1256431,  0.38216102,
#   0.08425483, 0.2846499,  0.41955787, 0.35012925, 0.4142725,  0.27319178,
#   0.30742413, 0.4338936,  0.28698516, 0.34129813]).astype(float)

# test_result =np.array([0.54148185, 0.5989439,  0.5360199,  0.36216533, 0.47865498, 0.74634564,
#   0.44820774, 0.4474908,  0.6495563,  0.65977645, 0.6580533,  0.3742265,
#   0.62183696, 0.85716605, 0.6291058,  0.48528886]).astype(float)

# test_result =np.array([254.54929,     269.52136,     257.1027,      -35.071198,    245.9971,
#     422.66135,     250.33188,      34.43669,     273.67792,     331.52164,
#     280.62592,       4.660797,    267.6921,      558.5957,      277.20822,
#     100.11922]).astype(float)


name = '000545'
gt_xml = './data/VOCdevkit2007/VOC2007/Annotations/'

test_result = np.array([164.47864,  129.67883,  157.35123,   14.565598, 126.717735, 141.5711,
 111.32858,   10.293983, 204.75978,  156.70523,  205.63129,   22.919502,
 186.15489,  184.99905,  186.48483,   26.87223,  168.26607,   88.78332]).astype(float)



test_result = np.array([600.4378,  207.9633,  611.5942,   83.98434, 614.6177,  246.6752,  628.5866,
 110.32362, 606.3272,  215.91039, 619.8371,   91.36256, 627.44574, 252.19025,
 648.9563,  116.79321, 619.8242,  165.53494]).astype(float)

points = test_result
# l = points[-3]
# w = points[-2]
# h = points[-1]

img = cv2.imread('000545.jpg')
# maxdata = 899.993907
# mindata = -199.90121 
l = img.shape[1]
w = img.shape[0]
print(l,w)

# points = points * (maxdata - mindata) + mindata
j = 0
# print(points)

for i in range(9):

    x = int(points[j])
    y = int(points[j+1])
    
    cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
    j += 2

# center = np.array([340,  263]).astype(float)
# 621.52686  164.70004
# 428.7258   395.5853
# 197.54993  205.94223
# 500 387
# cv2.circle(img, (int(center[0]), int(center[1])), 2, (255, 0, 255), 3)

# fang[-1]

xml = gt_xml + name + '.xml'
tree = ET.parse(xml)
objs = tree.findall('object')
num_objs = len(objs)
gt_points2d = np.zeros((num_objs, 16)).astype(float)
gt_center = np.zeros((num_objs, 2)).astype(float)
for ix, obj in enumerate(objs):
  points2d = obj.find('points2d')  #fang
  px1 = float(points2d.find('x1').text)
  py1 = float(points2d.find('y1').text)
  px2 = float(points2d.find('x2').text)
  py2 = float(points2d.find('y2').text)
  px3 = float(points2d.find('x3').text)
  py3 = float(points2d.find('y3').text)
  px4 = float(points2d.find('x4').text)
  py4 = float(points2d.find('y4').text)
  px5 = float(points2d.find('x5').text)
  py5 = float(points2d.find('y5').text)
  px6 = float(points2d.find('x6').text)
  py6 = float(points2d.find('y6').text)
  px7 = float(points2d.find('x7').text)
  py7 = float(points2d.find('y7').text)
  px8 = float(points2d.find('x8').text)
  py8 = float(points2d.find('y8').text)
  gt_points2d[ix, :] = [px1, py1, px2, py2, px3, py3, px4, py4, px5, py5, px6, py6, px7, py7, px8, py8]
  px = np.mean(gt_points2d[ix, 0::2])
  py = np.mean(gt_points2d[ix, 1::2])
#   print(center[0], center[1])
  print(px, py)
  gt_center[ix, 0] = px
  gt_center[ix, 1] = py

  cv2.circle(img, (int(px), int(py)), 2, (255, 0, 0), 3)


# #   mindata = -199.901211
# #   maxdata = 899.993907
# j = 0
# gt_points2d = np.double(gt_points2d[0])
# for i in range(8):

#   x = int(gt_points2d[j])
#   y = int(gt_points2d[j+1])
  
#   cv2.circle(img, (x, y), 2, (255, 0, 0), 3)
#   j += 2
#   j = 0
#   for i in range(9):
#     x = gt_points2d[ix, j:j+1]
#     y = gt_points2d[ix, j+1:j+2]
# # #     # print(x,y)
# # #     x = int(x * (maxdata - mindata) + mindata)
# # #     y = int(y * (maxdata - mindata) + mindata)
# # #     # print(x,y)
#     j = j + 2
#     if ix == 0:
#       cv2.circle(img, (x,y), 3, (255, 0, 255), 2)
#     elif ix == 1:
#       cv2.circle(img, (x,y), 3, (0, 0, 255), 2)
#     elif ix == 2:
#       cv2.circle(img, (x,y), 3, (255, 0, 0), 2)
#     elif ix == 3:
#       cv2.circle(img, (x,y), 3, (255, 255, 0), 2)

# pre_cneter = Variable(torch.from_numpy(center))
# gt_center = Variable(torch.from_numpy(gt_center))

# loss = F.smooth_l1_loss(pre_cneter, gt_center)
# print(loss)
# fang[-1]

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()