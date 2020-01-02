import os
import sys

import pandas as pd
import cv2

root = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/病例资料的副本'
dest = os.path.join(root, 'images')
for patient in sorted(os.listdir(os.path.join(root, 'raw_images'))):
    # os.mkdir(os.path.join(dest, patient))
    for img_name in os.listdir(os.path.join(root, 'raw_images', patient)):
        img = cv2.imread(os.path.join(root, 'raw_images', patient, img_name))
        if img is None:
            print(patient, img_name, file=sys.stderr)
            continue
        img = cv2.resize(img, (740, 490))
        cv2.imwrite(os.path.join(dest, patient, img_name[:-4] + '.png'), img)
        print(patient, img_name)
