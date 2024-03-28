#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2024-03-27 10:58:47
# @Author  : Your Name (you@example.org)
# @Link    : link
# @Version : 1.0.0
""" 基于相似度的图像搜索引擎 """
import os
import argparse
import matplotlib.pyplot as plt
import glob
import numpy as np
import sys
from PIL import Image
import torch
import torchvision
import timm
import clip
import tqdm

# device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
def get_args_parser():
    parser = argparse.ArgumentParser("image search engine", add_help=False)
    
    parser.add_argument("--input_size", type=int, default=128,
                        help="input size of the model")
    
    parser.add_argument("--dataset_dir", type=str, default="./train",
                        help="path to the dataset")
    
    parser.add_argument("--test_image_dir", type=str, default="./val",
                        help="images to test")
    
    parser.add_argument("--save_dir", type=str, default="./output",
                        help="path to save the result")
    
    parser.add_argument("--model_name", type=str, default="resnet50",
                        help="model name(resnet50, resnet152, clip)")
    
    parser.add_argument("--freature_dict_file", type=str, default="feature_dict.npy", 
                        help="feature dict file")
    
    parser.add_argument("--top_k", type=int, default=7,
                        help="k most similar images to be picked")
    
    parser.add_argument("--mode", type=str, default="search",
                        help="extract or search")
    return parser
    

def extract_features_by_CLIP(model, preprocess, file):
    image = preprocess(Image.open(file)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features.cpu().numpy()
    
    return image_features

def extract_features_by_timm(args, model, file):
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((args.input_size, args.input_size), Image.LANCZOS)
    image = torchvision.transforms.ToTensor()(image)
    trainset_mean = [0.47208185, 0.43395442, 0.32579166]
    trainset_std = [0.37739209, 0.36132373, 0.34935262]
    image = torchvision.transforms.Normalize(mean=trainset_mean, std=trainset_std)(image)
    image = image.unsqueeze(0).to(device)
    model = model.to(device)
    with torch.no_grad():
        image_features = model.forward_features(image)
        image_features = model.global_pool(image_features)
        image_features = image_features.cpu().numpy()

    
    return image_features

def extrac_all_features(args, model, image_path='', preprocess=None):
    allVectors = {}
    
    for image_file in tqdm.tqdm(glob.glob(os.path.join(image_path, "*", "*.jpg"))):
        if args.model_name == "clip":
            allVectors[image_file] = extract_features_by_CLIP(model, preprocess, image_file)
        else:
            allVectors[image_file] = extract_features_by_timm(args, model, image_file) 

    os.makedirs(f"{args.save_dir}/{args.model_name}", exist_ok=True)
    np.save(f"{args.save_dir}/{args.model_name}/{args.freature_dict_file}", allVectors)

    return allVectors

def getSimilarityMartrix(allVectors):
    v = np.array(list(allVectors.values()))
    if np.ndim(v) == 3:
        v = np.reshape(v, (np.shape(v)[0], np.shape(v)[2]))
    numerator = np.dot(v, v.T)
    denominator = np.dot(np.linalg.norm(v, axis=1, keepdims=True), np.linalg.norm(v, axis=1, keepdims=True).T)
    sim = numerator / denominator
    keys = list(allVectors.keys())
    return sim, keys

def setAxes(ax, title, query=False, value=None):
    if query:
        ax.set_title(f"Query: {title}", fontsize=10)
    else:
        ax.set_title(f"Score: {value:.3f}", fontsize=10)
    ax.axis('off')

def plotSimilarImages(args, image, simImages, simScores, numRow=1, numCol=7):
    fig = plt.figure()
    
    fig.set_size_inches([18.5, 10.5])
    fig.suptitle(f"Top {args.top_k} similar images by {args.model_name}", fontsize=35)
    
    for j in range(0, args.top_k):
        ax = []
        if j == 0:
            img = Image.open(image)
            ax = fig.add_subplot(numRow, numCol, 1)
            setAxes(ax, image.split(os.sep)[-1], query=True)
        else:
            img = Image.open(simImages[j-1])
            ax = fig.add_subplot(numRow, numCol, j+1)
            setAxes(ax, simImages[j-1].split(os.sep)[-1], value=simScores[j-1])
        
        img = img.convert("RGB")
        plt.imshow(img)
        img.close()
    
    fig.savefig(f"{args.save_dir}/{args.model_name}/{image.split(os.sep)[-1]}_similar_images.jpg")
    plt.show()
    
if __name__ == "__main__":
    
    args = get_args_parser()
    args = args.parse_args()
    
    model = None
    preprocess = None
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if args.model_name == "clip":
        model, preprocess = clip.load("ViT-B/32", device=device)
    else:
        model = timm.create_model(args.model_name, pretrained=True)
        num_params = sum(p.numel() for p in model.parameters())
        print("Number of parameters: ", num_params)
        model.eval()
        
    if args.mode == "extract":
        # 第一阶段：图像表征提取
        print(f"Extracting features from {args.dataset_dir} by {args.model_name}")
        allVectors = extrac_all_features(args, model, args.dataset_dir, preprocess)
        
    else:
        # 第二阶段：图像搜索
        print(f"use pretrained model {args.model_name} to search {args.top_k} most similar images")  
        
        test_images = glob.glob(os.path.join(args.test_image_dir, "*.jpg"))
        test_images += glob.glob(os.path.join(args.test_image_dir, "*.png"))
        test_images += glob.glob(os.path.join(args.test_image_dir, "*.jpeg"))
        
        # load feature dict
        allVectors = np.load(f"{args.save_dir}/{args.model_name}/{args.freature_dict_file}", allow_pickle=True)
        allVectors = allVectors.item()
        
        # reading test images
        for image_file in tqdm.tqdm(test_images):
            print(f"loading {image_file}...")
            
            if args.model_name == "clip":
                allVectors[image_file] = extract_features_by_CLIP(model, preprocess, image_file)
            else:           
                allVectors[image_file] = extract_features_by_timm(args, model, image_file)
                
            sim, key = getSimilarityMartrix(allVectors)
            
        result = {}
        for image_file in tqdm.tqdm(test_images):
            idx = key.index(image_file)
            sim_vec = sim[idx]
            indexes = np.argsort(sim_vec)[::-1][1:args.top_k]
            simImages, simScores = [], []
            for index in indexes:
                simImages.append(key[index])
                simScores.append(sim_vec[index])
            result[image_file] = simImages, simScores

        # 展示结果
        for image_file in test_images:
            plotSimilarImages(args, image_file, result[image_file][0], result[image_file][1])
            
            