#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2024-03-27 09:02:27
# @Author  : Zihao Cheng
# @Link    : link
# @Version : 1.0.0

import os
import glob
import random
import shutil
from PIL import Image

if __name__ == '__main__':
    test_split_ratio = 0.05
    desired_size = 128
    raw_path = './raw'
    
    dirs = glob.glob(os.path.join(raw_path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]
    
    print(f"Totally {len(dirs)} classes")
    
    for path in dirs:
        # 对每个类别单独处理
        path = path.split(r'/')[-1]
        
        os.makedirs(f'train/{path}', exist_ok=True)
        os.makedirs(f"test/{path}", exist_ok=True)

        files = glob.glob(os.path.join(raw_path, path, '*.jpg'))
        files += glob.glob(os.path.join(raw_path, path, '*.JPG'))
        files += glob.glob(os.path.join(raw_path, path, '*.png'))
        
        random.shuffle(files)
        boundary = int(len(files) * test_split_ratio)
        for i, file in enumerate(files):
            img = Image.open(file).convert("RGB")
            
            old_size = img.size
            
            ratio = float(desired_size / max(old_size))
            
            new_size = tuple([int(x * ratio) for x in old_size])
            
            im = img.resize(new_size, Image.LANCZOS)
            
            new_im = Image.new("RGB", (desired_size, desired_size))
            
            new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
            
            assert new_im.mode == "RGB"
            
            if i <= boundary:
                new_im.save(os.path.join(f"test/{path}", file.split('/')[-1].split('.')[0]+'.jpg'))
                
            else:
                new_im.save(os.path.join(f"train/{path}", file.split('/')[-1].split('.')[0]+'.jpg'))
                
    test_files = glob.glob(os.path.join('test', '*', '*.jpg'))
    train_files = glob.glob(os.path.join('train', '*', '*.jpg'))
    
    print(f"Train set has {len(train_files)} images")
    print(f"Test set has {len(test_files)} images")