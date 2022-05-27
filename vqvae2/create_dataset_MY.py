import argparse
import h5py
from torchvision import transforms
#from utilities import CXRDataset
import os
import pydicom
from pydicom import dcmread
import numpy as np

#########################
# GENERATE pydicom DATASET #
# ########################

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--crop_size', type=int, default=64)
# parser.add_argument('--CXR_dataset', type=str, default='CheXpert')
args = parser.parse_args()

# IMG_DIR = f'/home/aisinai/data/{args.CXR_dataset}'
# DATA_DIR = f'/home/aisinai/data/{args.CXR_dataset}'
IMG_DIR = 'C:/Users/User/Desktop/philips/vqvae2/dataset_MY/train_init'
DATA_DIR = 'C:/Users/User/Desktop/philips/vqvae2/dataset_MY/train_init'
# HDF5_DIR = '/home/aisinai/work/HDF5_datasets'
PYDICOM_DIR = 'C:/Users/User/Desktop/philips/vqvae2/dataset_MY/train_np/'
os.makedirs(PYDICOM_DIR, exist_ok=True)

num_label = 14
nc = 1  # number of channels; 3 for RGB, 1 for grayscale
# mean = [0.485, 0.456, 0.406]  # ImageNet mean
# std = [0.229, 0.224, 0.225]  # ImageNet std

# for info view jupyter notebook Projections_pydicom.ipynb
mean = 15398
std = 11227
normalization = transforms.Normalize(mean=mean, std=std)
transform_array = [transforms.ToPILImage(),
                   transforms.Resize(args.img_size),
                   # transforms.CenterCrop(args.crop_size),
                   transforms.RandomCrop(args.crop_size),
                   transforms.ToTensor(),
                   normalization]
file_list = os.listdir(DATA_DIR)
for f in file_list:
    dcm = DATA_DIR + '/' + f
    dcm_1 = dcmread(dcm)
    im_1 = dcm_1.pixel_array.astype(np.uint8)
    img = transforms.Compose(transform_array)(im_1)
    np.save(PYDICOM_DIR+f, img)
