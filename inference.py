import sys, os

from helper import *

import cv2
import pandas as pd
from glob import glob
import numpy as np

from timeit import default_timer as timer


import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

print('IMPORT OK  !!!!')


cfg = dotdict(
    batch_size = 3,
    p_threshold = 0.10,
    cc_threshold = -1,
)

mode = 'submit' # 'local' #

data_dir = \
    '/kaggle/input/blood-vessel-segmentation'

#-----
def file_to_id(f):
    s = f.split('/')
    return s[-3]+'_' + s[-1][:-4]

if 'local' in mode:
    valid_folder = [
        ('kidney_3_sparse', (496, 996+1)),
        #('kidney_1_dense', (0, 1000+1)),
    ] #debug for local development
    
    valid_meta = []
    for image_folder, image_no in valid_folder:
        file = [f'{data_dir}/train/{image_folder}/images/{i:04d}.tif' for i in range(*image_no)]
        H,W = cv2.imread(file[0],cv2.IMREAD_GRAYSCALE).shape
        valid_meta.append(dotdict(
            name  = image_folder,
            file  = file,
            shape = (len(file), H, W),
            id = [file_to_id(f) for f in file],
        ))
        
if 'submit' in mode:
    valid_meta = []
    valid_folder = sorted(glob(f'{data_dir}/test/*'))
    for image_folder in valid_folder:
        file = sorted(glob(f'{image_folder}/images/*.tif'))
        H, W = cv2.imread(file[0], cv2.IMREAD_GRAYSCALE).shape
        valid_meta.append(dotdict(
            name=image_folder,
            file=file,
            shape=(len(file), H, W),
            id=[file_to_id(f) for f in file],
        ))

#     glob_file = glob(f'{data_dir}/kidney_5/images/*.tif')
#     if len(glob_file)==3:
#         mode = 'submit-fake' #fake submission to save gpu time when submitting
#         #todo .....




print('len(valid_meta) :', len(valid_meta))
print(valid_meta[0].file[:3])

print('MODE OK  !!!!')