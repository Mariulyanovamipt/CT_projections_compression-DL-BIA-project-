import argparse
import h5py
from torchvision import transforms
#from utilities import CXRDataset
import os
import pydicom
from pydicom import dcmread
import torch
import numpy as np
import cv2
from torchvision.utils import save_image

path='C:/Users/User/Desktop/philips/vqvae2/dataset_MY/train_init'

img_filenames = [os.path.join(path, x) for x in os.listdir(path)]
#print(img_filenames)

index = 1
dicom_img = dcmread(img_filenames[index]).pixel_array
mean = 15397.0
std = 11227.0
img_normed = dicom_img / mean + std
print(np.max(img_normed))
print(np.max(dicom_img), 'maximum of not normalized')
img_normed =dicom_img/mean+std
img_torch = torch.from_numpy(img_normed)

transform_ = transforms.RandomCrop(64)
img_cropped = transform_(img_torch)
print(img_cropped.shape)
print(img_cropped.unsqueeze(0).shape)
print(img_cropped.unsqueeze(1).shape)

print(img_cropped.unsqueeze(0).squeeze(0).shape)

n_row=2

save_image(torch.cat([img_torch, img_torch], dim=1).data, f'C:/Users/User/Desktop/philips/vqvae2/results/MY/0/sample/original___5_normed_1row.png',
               nrow=n_row, normalize=True)
n_row=1
save_image(torch.cat([img_torch, img_torch], dim=0).data, f'C:/Users/User/Desktop/philips/vqvae2/results/MY/0/sample/original___5_normed+4row.png',
               nrow=n_row ** 2, normalize=True)
print('4 ^ 2', 4**2)
save_image(torch.cat([img_torch, img_torch], dim=0).data, f'C:/Users/User/Desktop/philips/vqvae2/results/MY/0/sample/original___5_normed+4row.png',
               nrow=n_row ** 2, normalize=True)
#save image saves tensors b_size/nrows x n_rows

save_image(torch.cat([img_torch.unsqueeze(0), img_torch.unsqueeze(0), img_torch.unsqueeze(0), img_torch.unsqueeze(0), img_torch.unsqueeze(0), img_torch.unsqueeze(0)], dim=1).data, f'C:/Users/User/Desktop/philips/vqvae2/results/MY/0/sample/original6___unsqueezed_dim0_3row.png',
               nrow=3, normalize=True)
save_image(torch.cat([img_torch, img_torch, img_torch, img_torch, img_torch], dim=0).data, f'C:/Users/User/Desktop/philips/vqvae2/results/MY/0/sample/original___dim0_6row.png',
               nrow=6, normalize=True)

save_image(torch.cat([img_torch, img_torch, img_torch, img_torch, img_torch], dim=1).data, f'C:/Users/User/Desktop/philips/vqvae2/results/MY/0/sample/original___dim1_4row.png',
               nrow=4, normalize=True)

#cv2.imwrite(img_cropped, f'C:/Users/User/Desktop/philips/vqvae2/results/sample/original', 'png');

n_row=1
save_image(img_cropped.data*255/(torch.max(img_cropped)-torch.min(img_cropped)), f'C:/Users/User/Desktop/philips/vqvae2/results/MY/0/sample/original___5_1.png',
               nrow=5, normalize = True)
