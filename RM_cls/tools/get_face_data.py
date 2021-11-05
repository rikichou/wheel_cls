import sys
import cv2
import os

ann_file = '/home/ruiming/workspace/pro/facialExpression/data/select/clean_1029/val.txt'
out_img_root_dir = '/home/ruiming/workspace/pro/facialExpression/data/select/clean_1029/faceonly/test'
if not os.path.exists(out_img_root_dir):
    os.makedirs(out_img_root_dir)
src_img_root_dir = '/home/ruiming/workspace/pro/facialExpression/data/select'

with open(ann_file) as f:
    samples = [x.strip().split(';') for x in f.readlines()]

def get_input_face(image, rect):
    sx,sy,ex,ey = rect
    if len(image.shape)<3:
        h,w = image.shape
    else:
        h,w,c = image.shape
    faceh = ey-sy
    facew = ex-sx

    longsize = max(faceh, facew)
    expendw = longsize-facew
    expendh  = longsize-faceh

    sx = sx-(expendw/2)
    ex = ex+(expendw/2)
    sy = sy-(expendh/2)
    ey = ey+(expendh/2)

    sx = int(max(0, sx))
    sy = int(max(0, sy))
    ex = int(min(w-1, ex))
    ey = int(min(h-1, ey))

    if len(image.shape)<3:
        return image[sy:ey, sx:ex]
    else:
        return image[sy:ey, sx:ex, :]

label_map = {0:'angry', 1:'happy', 2:'neutral', 3:'sad'}

count = 0
for filename, gt_label, sx, sy, ex, ey in samples:
    count += 1
    gt_label = int(gt_label)
    if gt_label not in label_map:
        continue

    sx = int(sx)
    sy = int(sy)
    ex = int(ex)
    ey = int(ey)

    # image path
    img_path = os.path.join(src_img_root_dir, filename)
    if not os.path.exists(img_path):
        print(img_path)
        continue

    # read image
    img = cv2.imread(img_path)
    if not img is None:
        face_img = get_input_face(img, (sx,sy,ex,ey))

        # get output path
        out_cls_dir = os.path.join(out_img_root_dir, label_map[gt_label])
        if not os.path.exists(out_cls_dir):
            os.makedirs(out_cls_dir)
        cv2.imwrite(os.path.join(out_cls_dir, os.path.basename(filename)), face_img)

    if count %1000 == 0:
        print("{}/{}".format(count, len(samples)))
