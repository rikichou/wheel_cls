import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def img_transform(img):
    w,h = img.size
    img = np.array(img)

    scale = np.random.uniform(0.25, 0.35)
    center_h = np.floor((h)*scale).astype(np.int)
    stride_h = np.floor((h - center_h)/2).astype(np.int)
    center_w = np.floor((w)*scale).astype(np.int)
    stride_w = np.floor((w - center_w)/2).astype(np.int)

    img[stride_h:(h-stride_h), stride_w:(w-stride_w)] = (0, 0, 0)

    return Image.fromarray(img.astype(np.uint8))

image_dir = '/xcli/wheels_cls/wheel_imgs/gradcamtest'
dirs = []
for dir in os.listdir(image_dir):
    dirs.append(dir)
dirs.sort()

global_id = 0
for dir in dirs:
    cur_dir = os.path.join(image_dir, dir)
    for jpeg in os.listdir(cur_dir):
        img = Image.open(os.path.join(cur_dir, jpeg))
        img = img_transform(img)

        img.save(os.path.join('/xcli/outputs/masktest', str(global_id))+'.jpg')
        #print('successfully saved')
        global_id += 1


"""
#img_dir = '/xcli/wheels_cls/wheel_imgs/gradcamtest/_02_06_0000_200610 (227).avi_40_738.jpg'
img_dir = '/xcli/wheels_cls/wheel_imgs/gradcamtest/_02_06_0000_200610 (227).avi_85_741.jpg'
img = Image.open(img_dir)
w,h = img.size
img = np.array(img)

scale = np.random.uniform(0.25, 0.35)
center_h = np.floor((h)*scale).astype(np.int)
stride_h = np.floor((h - center_h)/2).astype(np.int)
center_w = np.floor((w)*scale).astype(np.int)
stride_w = np.floor((w - center_w)/2).astype(np.int)

img[stride_h:(h-stride_h), stride_w:(w-stride_w)] = (0, 0, 0)
img.save('transfrom_result1.jpg')
"""


