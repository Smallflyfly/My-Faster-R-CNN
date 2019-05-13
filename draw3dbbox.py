import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

xmlpath = './data/VOCdevkit2007/VOC2007/Annotations/'
imgpath = './data/VOCdevkit2007/VOC2007/JPEGImages/'

test_name = '010057'
# test_name = '000002'


img = imgpath + test_name + '.jpg'
xml = xmlpath + test_name + '.xml'
img = cv2.imread(img)
# img2 = img1[:, ::-1, :]
# cv2.imshow('img1',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('img2',img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



tree = ET.parse(xml)
objs = tree.findall('object')
num_objs = len(objs)
boxes = np.zeros((num_objs, 4)).astype(float)

gt_points2d = np.zeros((num_objs, 16), dtype=np.float32)
gt_front_4points = np.zeros((num_objs, 8), dtype=np.float32)
gt_back_4points = np.zeros((num_objs, 8), dtype=np.float32)
for ix, obj in enumerate(objs):

  bbox = obj.find('bndbox')
  x1 = float(bbox.find('xmin').text)
  y1 = float(bbox.find('ymin').text)
  x2 = float(bbox.find('xmax').text)
  y2 = float(bbox.find('ymax').text)

  boxes[ix, :] = [x1, y1, x2, y2]

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
  gt_front_4points[ix, :] = [px1, py1, px2, py2, px3, py3, px4, py4]
  gt_back_4points[ix, :] = [px5, py5, px6, py6, px7, py7, px8, py8]
  center_x = np.mean(gt_points2d[ix, 0::2]).astype(int)
  center_y = np.mean(gt_points2d[ix, 1::2]).astype(int)
  front_center_x = np.mean(gt_front_4points[ix, 0::2]).astype(int)
  front_center_y = np.mean(gt_front_4points[ix, 1::2]).astype(int)
  back_center_x = np.mean(gt_back_4points[ix, 0::2]).astype(int)
  back_center_y = np.mean(gt_back_4points[ix, 1::2]).astype(int)

  if px1 > px5:
    front_x1 = int(px1)
    front_y1 = int(py4)
    front_x2 = int(px4)
    front_y2 = int(py1)
  else:
    front_x1 = int(px1)
    front_y1 = int(py4)
    front_x2 = int(px4)
    front_y2 = int(py1)

  # cv2.rectangle(img, (front_x1, front_y1), (front_x2, front_y2), (0, 123, 255), 2)
  # cv2.circle(img, (center_x,center_y), 2, (255, 0, 255), 2)
  # cv2.circle(img, (front_center_x,front_center_y), 2, (255, 0, 255), 2)
  # cv2.circle(img, (back_center_x,back_center_y), 2, (255, 0, 255), 2)
  
  cv2.circle(img, (int(px1), int(py1)), 2, (0, 255, 0), 2)
  cv2.circle(img, (int(px5), int(py5)), 2, (0, 255, 0), 2)
  # cv2.circle(img, (int(px4), int(py4)), 2, (255, 0, 255), 2)

  cv2.line(img, (int(px1),  int(py1)), (int(px2),  int(py2)), 255, 2)
  cv2.line(img, (int(px1),  int(py1)), (int(px3),  int(py3)), 255, 2)
  cv2.line(img, (int(px4),  int(py4)), (int(px2),  int(py2)), 255, 2)
  cv2.line(img, (int(px4),  int(py4)), (int(px3),  int(py3)), 255, 2)

  cv2.line(img, (int(px5),  int(py5)), (int(px6),  int(py6)), 255, 2)
  cv2.line(img, (int(px5),  int(py5)), (int(px7),  int(py7)), 255, 2)
  cv2.line(img, (int(px8),  int(py8)), (int(px6),  int(py6)), 255, 2)
  cv2.line(img, (int(px8),  int(py8)), (int(px7),  int(py7)), 255, 2)

  cv2.line(img, (int(px1),  int(py1)), (int(px5),  int(py5)), 255, 2)
  cv2.line(img, (int(px2),  int(py2)), (int(px6),  int(py6)), 255, 2)
  cv2.line(img, (int(px3),  int(py3)), (int(px7),  int(py7)), 255, 2)
  cv2.line(img, (int(px4),  int(py4)), (int(px8),  int(py8)), 255, 2)

  # cv2.rectangle(img, (int(x1),int(y1)), (int(x2), int(y2)), (255, 255, 0), 2, )

  # cv2.line(img, (int(px2),  int(py2)), (int(px3),  int(py3)), 255, 2)
  # cv2.line(img, (int(px1),  int(py1)), (int(px4),  int(py4)), 255, 2)

  # cv2.line(img, (int(px5),  int(py5)), (int(px8),  int(py8)), 255, 2)
  # cv2.line(img, (int(px6),  int(py6)), (int(px7),  int(py7)), 255, 2)

  # cv2.line(img, (int(front_center_x),  int(front_center_y)), (int(back_center_x),  int(back_center_y)), 255, 2)


  # j = 0
  # for i in range(4):
  #   x = gt_points2d[ix, j:j+1]
  #   y = gt_points2d[ix, j+1:j+2]

  #   j = j + 2
  #   if ix == 0:
  #     cv2.circle(img, (x,y), 3, (255, 0, 255), 2)
  #   elif ix == 1:
  #     cv2.circle(img, (x,y), 3, (0, 0, 255), 2)
  #   elif ix == 2:
  #     cv2.circle(img, (x,y), 3, (255, 0, 0), 2)
  #   elif ix == 3:
  #     cv2.circle(img, (x,y), 3, (255, 255, 0), 2)


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# plt.scatter(x, y)
# plt.show()