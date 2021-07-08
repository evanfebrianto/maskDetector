# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:37:14 2021
@author: Evan Febrianto

This code is to split the original PASCAL VOC format to train and val folder
in order to prepare the YOLO format conversion
"""

import random
from os import listdir, makedirs, path
from shutil import copy, rmtree
from tqdm import tqdm
from glob import glob

DATASET_DIRECTORY = "./dataset"
OUTPUT_DIRECTORY = "./cleaned-dataset"
TRAIN_RATIO = 0.8 # range from 0 to 1
SEED = 27

# Remove existing directory and create a new one
if path.exists(path.join(OUTPUT_DIRECTORY,'train')):
    rmtree(path.join(OUTPUT_DIRECTORY,'train'))
if path.exists(path.join(OUTPUT_DIRECTORY,'val')):
    rmtree(path.join(OUTPUT_DIRECTORY,'val'))
makedirs(path.join(OUTPUT_DIRECTORY,'train'))
makedirs(path.join(OUTPUT_DIRECTORY,'val'))

xml_files = [path.join(XML_DIRECTORY,file) 
            for file in listdir(XML_DIRECTORY) 
            if path.isfile(path.join(XML_DIRECTORY,file))]

# Shuffle xml_files
random.seed(SEED)
random.shuffle(xml_files)

# Copy files to the output directory
positive_sample = int(TRAIN_RATIO * len(xml_files))
corrupt_files = []
for xml_file in tqdm(xml_files):
    file_name = path.split(xml_file)[1].split('.')[0]
    img_file = glob(path.join(IMAGE_DIRECTORY,file_name+'.*'))[0]
    isXMLCorrupt, isIMGCorrupt = False, False
    if xml_file in xml_files[:positive_sample]:
        try:
            copy(src=xml_file, 
                dst=path.join(OUTPUT_DIRECTORY,'train'))
        except:
            isXMLCorrupt = True
        try:
            copy(src=img_file, 
                dst=path.join(OUTPUT_DIRECTORY,'train'))
        except:
            isIMGCorrupt = True
    else:
        try:
            copy(src=xml_file, 
                dst=path.join(OUTPUT_DIRECTORY,'val'))
        except:
            isXMLCorrupt = True
        try:
            copy(src=img_file, 
                dst=path.join(OUTPUT_DIRECTORY,'val'))
        except:
            isIMGCorrupt = True
    if isXMLCorrupt:
        corrupt_files.append(xml_file)
    if isIMGCorrupt:
        corrupt_files.append(img_file)
        
if len(corrupt_files) == 0:
    print('DONE! No corrupt files found.')
else:
    print('Some files are having problem, please check these files!')
    print(corrupt_files)