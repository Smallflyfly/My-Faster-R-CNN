import torch
from ._ext import nms
import numpy as np

def pth_nms(dets, thresh):
  """
  dets has to be a tensor
  """
  if not dets.is_cuda:

    # print('11111111111111111111111111')
    # fang[-1]

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # print(x1)
    # fang[-1]

    # print(dets.shape) # 300 * 24
    # fang[-1]
    # points = dets[:,5:24]
    # print(points.shape) # 300 * 19
    # fang[-1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]
    # order = torch.from_numpy(np.ascontiguousarray(scores.numpy().argsort()[::-1])).long()

    keep = torch.LongTensor(dets.size(0))
    # print(keep)
    # fang[-1]
    num_out = torch.LongTensor(1)
    # print(num_out)
    # fang[-1]
    nms.cpu_nms(keep, num_out, dets, order, areas, thresh)

    return keep[:num_out[0]]
  else:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # print(dets.shape)
    # points = dets[:,5:24]
    
    # print(points)
    # fang[-1]
    #use cuda

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]
    # order = torch.from_numpy(np.ascontiguousarray(scores.cpu().numpy().argsort()[::-1])).long().cuda()

    dets = dets[order].contiguous()

    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    # keep = torch.cuda.LongTensor(dets.size(0))
    # num_out = torch.cuda.LongTensor(1)
    nms.gpu_nms(keep, num_out, dets, thresh)

    return order[keep[:num_out[0]].cuda()].contiguous()
    # return order[keep[:num_out[0]]].contiguous()

