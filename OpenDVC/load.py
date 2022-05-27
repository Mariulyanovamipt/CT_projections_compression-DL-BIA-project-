import numpy as np
from scipy import misc
from pydicom import dcmread
import os

def load_data(data, frames, batch_size, Height, Width, Channel, folder, I_QP):

    for b in range(batch_size):

        path = folder[np.random.randint(len(folder))] + '/'

        bb = np.random.randint(0, 447 - 256)

        for f in range(frames):

            if f == 0:
                img = misc.imread(path + 'im1_bpg444_QP' + str(I_QP) + '.png')
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]
            else:
                img = misc.imread(path + 'im' + str(f + 1) + '.png')
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]

    return data


def load_data_ssim(data, frames, batch_size, Height, Width, Channel, folder, I_level):

    for b in range(batch_size):

        path = folder[np.random.randint(len(folder))] + '/'

        bb = np.random.randint(0, 447 - 256)

        for f in range(frames):

            if f == 0:
                img = misc.imread(path + 'im1_level' + str(I_level) + '_ssim.png')
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]
            else:
                img = misc.imread(path + 'im' + str(f + 1) + '.png')
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]

    return data

def load_data_ssim_pydicom(data, folder, frames=10, batch_size=4, Height=736, Width=64, Channel=1, I_level=None):

    for b in range(batch_size):

        path = folder[np.random.randint(len(folder))] + '/'

        #bb = np.random.randint(0, 447 - 256)
        imlist = [os.path.join(path, x) for x in os.listdir(path)]

        for f in range(frames):
            img = np.expand_dims(dcmread(imlist[f*b]).pixel_array, axis = 2)
            data[f, b, :, :, :] = img[:, :, :]

    return data

def load_pydicom_one_folder(data, folder, frames=10, batch_size=4, Height=736, Width=64, Channel=1, I_level=None):

    for b in range(batch_size):

        #path = folder[np.random.randint(len(folder)-b*frames)] + '/'

        #bb = np.random.randint(0, 447 - 256)
        path = folder
        imlist = [os.path.join(path, x) for x in os.listdir(path)]
        img_num = 100 + np.random.randint(len(imlist)-(b+1)*frames-100)

        for f in range(frames):
            img = np.expand_dims(dcmread(imlist[img_num+f+b*frames], force = True).pixel_array, axis = 2)
            data[f, b, :, :, :] = img[:, :, :]


    return data


def load_pydicom_one_folder(data, folder, frames=10, batch_size=4, Height=736, Width=64, Channel=1, I_level=None):

    for b in range(batch_size):

        #path = folder[np.random.randint(len(folder)-b*frames)] + '/'

        #bb = np.random.randint(0, 447 - 256)
        path = folder
        imlist = [os.path.join(path, x) for x in os.listdir(path)]
        img_num = 100 + np.random.randint(len(imlist)-(b+1)*frames-100)

        for f in range(frames):
            img = np.expand_dims(dcmread(imlist[img_num+f+b*frames], force = True).pixel_array, axis = 2)
            data[f, b, :, :, :] = img[:, :, :]


    return data

