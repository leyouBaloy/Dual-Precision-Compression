import torch
import time

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
import skimage
from skimage import metrics
import numpy as np
import sys
import torch.nn.functional as F

from utils import calc_lp_cr
from module import Model,LPModel,HPModel
import utils


# print(f"LP压缩率: {(16*3120*4320*32)/(16*234*64*64*np.log2(16)+16*16*32):.6f}")
# nums = [8, 16, 32, 64]
# for i in nums:
#     for j in nums:

#         cr,bitrate = calc_lp_cr(4, 3120, 4320, 234, 64, i, j)
#         cr_nopad,bitrate_nopad = calc_lp_cr(4, 3120, 4320, 221, 64, i, j)
#         print(f"n={i},d={j}时，cr: {cr:.2f}")
#         print(f"n={i},d={j}时，nopad_cr: {cr_nopad:.2f}")
#         print(f"n={i},d={j}时，biterate: {bitrate:.2f}")

# cr,bitrate = calc_lp_cr(4, 3120, 4320, 234, 64, i, j)

cr,bitrate = calc_lp_cr(365, 1200, 480, 10, 64, 32, 32)
print(cr)

# part1 = 8*10*64*64*np.log2(32)+32*32*32
# part2 = 122*10*64*64*np.log2(32)+32*32*32
# part3 = 50*10*64*64*np.log2(32)+32*32*32
# part4 = 185*10*64*64*np.log2(32)+32*32*32
print(f"LP压缩率: {(365*1200*480*32)/(365*10*64*64*np.log2(32)+32*32*32*4):.6f}")