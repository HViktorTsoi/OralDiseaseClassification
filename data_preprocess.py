import os
import sys

import pandas as pd
import cv2
from PIL import Image
import numpy as np

root = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/病例资料的副本'
dest = os.path.join(root, 'images')
for patient in sorted(os.listdir(os.path.join(root, 'raw_images'))):
    # for patient in ['115']:
    # os.mkdir(os.path.join(dest, patient))
    for img_name in os.listdir(os.path.join(root, 'raw_images', patient)):
        img = cv2.imread(os.path.join(root, 'raw_images', patient, img_name))
        if img is None:
            print(patient, img_name, file=sys.stderr)
            continue
        print(patient, img.shape, file=sys.stderr if img.shape[0] > img.shape[1] else sys.stdout)
        # 旋转图像 保证宽大于高
        if img.shape[0] > img.shape[1]:
            img = np.transpose(img, axes=[1, 0, 2])
        img = cv2.resize(img, (400, 240))
        cv2.imwrite(os.path.join(dest, patient, img_name[:-4] + '.png'), img)
