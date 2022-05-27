import os
import random
import shutil
import numpy as np
from PIL import Image
from pydicom import dcmread
import imageio

def get_video_sequences (path):
    #path = 'dataset_MY/train_init'
    img_filenames = [os.path.join(path, x) for x in os.listdir(path)]
    #print(img_filenames)
    n_left = len(img_filenames)
    i = 0
    video_sequences = []
    while (n_left>0):
        n = random.randrange(30, 150)
        #print(n)
        new_video = img_filenames[i:i+n]
        video_sequences.append(new_video)
        i = i+n
        n_left = n_left - n
    return video_sequences

def save_video_dataset(imgs_path, videos_path):
    video_seqs = get_video_sequences(imgs_path)
    for i, seq in enumerate(video_seqs):
        newpath = videos_path+f'/{i}'
        os.makedirs(newpath)
        for f in seq:
            shutil.copy(f, os.path.join(newpath))

def save_png_dataset(path, png_path):
    img_filenames = [x for x in os.listdir(path)]
    #imlist = [os.path.join(path,im) for im in os.listdir(path) if im[-4:]=='.dcm']
    for f in img_filenames:
        arr = np.expand_dims(dcmread(os.path.join(path,f)).pixel_array, axis = 2)
        arr0 = dcmread(os.path.join(path,f)).pixel_array
        array_buffer = dcmread(os.path.join(path,f)).PixelData
        print(len(array_buffer))
        #img = Image.new("I", arr.T.shape)
        #img.frombytes(array_buffer, 'raw', "I;16")
        #img.save(os.path.join(png_path, f[:-4]+'.png'))
        arr3 = np.stack((arr0, arr0, arr0), axis = 2)
        print(arr3.shape)

        imageio.imwrite(os.path.join(png_path, f[:-4]+'.png'),arr3)

if __name__ == "__main__":
    imgs_path = 'dataset_video_MY/1'
    videos_path = 'dataset_video_MY'
    png_path = 'dataset_video_MY/1_png'
    #save_video_dataset(imgs_path, videos_path)
    save_png_dataset(imgs_path, png_path)
    print('Done')
