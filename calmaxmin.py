import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

xmlpath = './data/VOCdevkit2007/VOC2007/Annotations/'

xmlfiles = os.listdir(xmlpath)

points_x = []
points_y = []
xmin = 9999
ymin = 9999
xmax = -9999
ymax = -9999
gt_points2d_x = []
gt_points2d_y = []
center_x = []
center_y = []
for xmlfile in xmlfiles:
    xml = xmlpath + xmlfile
    tree = ET.parse(xml)
    objs = tree.findall('object')
    num_objs = len(objs)
    gt_points2d = np.zeros((num_objs, 16), dtype=np.float32)
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
        cp_points2d = gt_points2d.copy()
        px = np.mean(gt_points2d[ix, 0::2])
        py = np.mean(gt_points2d[ix, 1::2])
        # gt_points2d[ix, 0::2] = gt_points2d[ix, 0::2] - (px - 730.0 / 2)
        # gt_points2d[ix, 1::2] = gt_points2d[ix, 1::2] - (py - 530.0 / 2)
        # xmin = min(xmin, min(gt_points2d[ix, 0::2]))
        # ymin = min(ymin, min(gt_points2d[ix, 1::2]))
        # xmax = max(xmax, max(gt_points2d[ix, 0::2]))
        # ymax = max(ymax, max(gt_points2d[ix, 1::2]))
        # points_x.append(gt_points2d[ix, 0::2])
        # points_y.append(gt_points2d[ix, 1::2])
        # gt_points2d_x.append(cp_points2d[ix, 0::2])
        # gt_points2d_y.append(cp_points2d[ix, 1::2])
        center_x.append(px)
        center_y.append(py)
        xmin = min(xmin, px)
        ymin = min(ymin, py)
        xmax = max(xmax, px)
        ymax = max(ymax, py)
        
# print(xmin, ymin)
# print(xmax, ymax)
# print(len(points_x), len(points_y))
# plt.figure()
# plt.title('Transformed')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.scatter(points_x, points_y, s=5, c='b', marker='.')
# plt.show()

# plt.figure()
# plt.title('No transformed')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.scatter(gt_points2d_x, gt_points2d_y, s=5, c='r', marker='.')
# plt.show()

print(xmin, ymin)
print(xmax, ymax)

plt.figure()
plt.scatter(center_x,center_y, s=5, c='b', marker='.')
plt.show()