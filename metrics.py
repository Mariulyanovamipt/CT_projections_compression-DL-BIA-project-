import torch
import numpy as np
from piq import ssim, SSIMLoss, psnr, haarpsi, vsi, ms_ssim

def PSNR(x,y, data_range = 1.0):
    return psnr(x,y, data_range = data_range)

def SSIM(x, y, data_range = 1.0):
    return ssim(x, y, data_range = data_range)

def HaarsPSI(x,y, data_range =1.0):
    return haarpsi(x,y,data_range=data_range)

def VSI(x, y, data_range =1.0):
    return vsi(x, y, data_range=data_range)

#if you have npy tensor with shape N x H x W
def load_npy(path):
    arr = np.load(path)
    return arr

metric_zoo = {'psnr' : psnr, 'ssim' : ssim, 'haarpsi': haarpsi, 'vsi' : vsi}
def Metric_avg(tensor1, tensor2, metric_name, data_range = 1.0):
    metric = metric_zoo[metric_name]
    all_metrics = []
    for i in range(tensor1.shape[0]):
        x = np.expand_dims(tensor1[i, :, :], axis = (0,1))  #Bs x Ch x H x W
        y = np.expand_dims(tensor2[i,:,:], axis = (0,1))

        x = torch.tensor(x.astype('float64'))
        y = torch.tensor(y.astype('float64'))

        #print(x.shape)

        all_metrics.append(metric(x,y, data_range = data_range))
        return np.mean(np.array(all_metrics))

if __name__=='__main__':
    test_on_random_tensor = False
    test_on_npy = True
    #path_original = 'C:/Users/User/Desktop/philips/manifest-1643214718297/LDCT-and-Projection-data/C002.npy'
    path_original ='C:/Users/User/Desktop/philips/OpenDVC/test_good.npy'
    #path_reconstructed = 'C:/Users/User/Desktop/philips/manifest-1643214718297/LDCT-and-Projection-data/C002.npy'
    path_reconstructed ='C:/Users/User/Desktop/philips/OpenDVC/test_bad.npy'
    max_ = 65535.0
    if test_on_random_tensor:
        print('Testing metrics on random torch tensor... ')
        x = torch.rand(4, 3, 256, 256)
        y = torch.rand(4, 3, 256, 256)
        z = torch.zeros((4, 3, 256, 256))

        print('--- IDEAL case ---')
        #print(ms_ssim)
        print('psnr', psnr(y,y))
        print('ssim',ssim(y,y))
        print('haarpsi', haarpsi(y,y))
        print('vsi', vsi(y,y))

        print('--- Two random tensors ---')
        print('psnr', psnr(x,y))
        print('ssim',ssim(x,y))
        print('haarpsi', haarpsi(x,y))
        print('vsi', vsi(x,y))

    if test_on_npy:
        original_images = load_npy(path_original)
        reconstructed_images = load_npy(path_reconstructed)
        print('avg pnsr', Metric_avg(original_images, reconstructed_images, 'psnr' , data_range = max_))
        print('avg haarpsi', Metric_avg(original_images, reconstructed_images, 'haarpsi' , data_range = max_))
        print('avg ssim', Metric_avg(original_images, reconstructed_images, 'ssim' , data_range = max_))
        print('avg vsi', Metric_avg(original_images, reconstructed_images, 'vsi' , data_range = max_))
        #print('avg mssim', Metric_avg(original_images, reconstructed_images, 'mssim' , data_range = max_))


