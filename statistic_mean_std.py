#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2024-03-27 10:48:46
# @Author  : Your Name (you@example.org)
# @Link    : link
# @Version : 1.0.0

import os
import glob
import numpy as np
from PIL import Image
""" 统计训练数据库中所有图片的每个通道的均值和标准差 """

if __name__ == '__main__':
    train_files = glob.glob(os.path.join('train', '*', '*.jpg'))
    
    print(f"train_files: {len(train_files)}")
    
    result = []
    
    for file in train_files:
        img = Image.open(file).convert("RGB")
        img = np.array(img).astype(np.uint8)
        img = img / 255.
        result.append(img)
    
    print(np.shape(result)) # (n, 128, 128, 3)
    mean = np.mean(result, axis=(0, 1, 2))
    std = np.std(result, axis=(0, 1, 2))
    print(f"mean: {mean}")
    print(f"std: {std}")
        