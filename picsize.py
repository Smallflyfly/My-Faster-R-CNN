import PIL.Image
import os

def get_pic_size(filename):
    w = PIL.Image.open(filename).size[0]
    h = PIL.Image.open(filename).size[1]
    return w, h
picpath = os.path.join('./data', 'VOCdevkit2007', 'VOC2007', 'JPEGImages')
picfiles = os.listdir(picpath)
picsize = []
for picfile in picfiles:
    w, h = get_pic_size(picpath + '/' +picfile)
    p_size = [w, h]
    if p_size not in picsize:
        picsize.append(p_size)
    # print(w, h)
    # fang[-1]

print(picsize)