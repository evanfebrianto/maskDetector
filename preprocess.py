import random
from os import makedirs, path
from shutil import rmtree
from tqdm import tqdm
from glob import glob
from PIL import Image
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import configparser


WEBCAM_DIR = "webcam_dataset"
SAMPLED_DATASET = "resized-sampled-dataset"
OUTPUT_DIRECTORY = "dataset"
TRAIN_RATIO = 0.8 # range from 0 to 1
SEED = 27
NUM_SAMPLE = 2500 # -1 for all samples, > 0 for sampling data
RESIZED_SHAPE = (64,64)
classes = ['incorrect_mask', 'no_mask', 'mask']


def process_image(images, target_dir=OUTPUT_DIRECTORY):
    if not path.exists(path.join(target_dir,'train',_class)):
        makedirs(path.join(target_dir,'train',_class))
    if not path.exists(path.join(target_dir,'val',_class)):
        makedirs(path.join(target_dir,'val',_class))
    
    # randomize
    random.seed(SEED)
    random.shuffle(images)
    if NUM_SAMPLE > 0:
        images = images[:NUM_SAMPLE]
    
    # Copy files to the output directory
    positive_sample = int(TRAIN_RATIO * len(images))
    for img in tqdm(images):
        im_pil = Image.open(img)
        imResize = im_pil.resize(RESIZED_SHAPE, Image.ANTIALIAS)
        file_name = path.split(img)[1]
        if img in images[:positive_sample]:
            imResize.save(path.join(target_dir,'train',_class,file_name), 'JPEG', quality=90)
        else:
            imResize.save(path.join(target_dir,'val',_class,file_name), 'JPEG', quality=90)

# Remove existing dataset dir
if path.exists(path.join(OUTPUT_DIRECTORY,'train')):
    rmtree(path.join(OUTPUT_DIRECTORY,'train'))
if path.exists(path.join(OUTPUT_DIRECTORY,'val')):
    rmtree(path.join(OUTPUT_DIRECTORY,'val'))
makedirs(path.join(OUTPUT_DIRECTORY,'train'))
makedirs(path.join(OUTPUT_DIRECTORY,'val'))


for _class in classes:
    print('Processing: {}'.format(_class))
    #======================Processing Webcam Dataset=========================
    images = glob(path.join(WEBCAM_DIR,_class,'*'))
    process_image(images=images, target_dir=OUTPUT_DIRECTORY)

    #======================Processing Public Dataset=========================
    images_train = glob(path.join(SAMPLED_DATASET,'train',_class,'*'))
    images_val = glob(path.join(SAMPLED_DATASET,'val',_class,'*'))    
    process_image(images=images_train+images_val, target_dir=OUTPUT_DIRECTORY)

# Getting Mean and Std
print('\nGetting mean and std . . .')
train_set = ImageFolder(root=OUTPUT_DIRECTORY, transform=transforms.ToTensor())
data = torch.cat([d[0] for d in DataLoader(train_set)])
mean, std = data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
print('mean: {}\nstd: {}\n'.format(list(np.array(mean)),list(np.array(std))))

# Update config.ini
filename = 'config.ini'
config = configparser.ConfigParser()
config.read(filename)
config.set('LIST','MEAN',str(list(np.array(mean))))
config.set('LIST','STD',str(list(np.array(std))))
with open(filename, 'w') as configfile:
    config.write(configfile)
print('Config.ini is updated!\n')