import os
import pydicom
import torch 
from networks import VQVAE
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from utilities import rgb2gray
import numpy as np
import matplotlib.pyplot  as plt

cuda = True if torch.cuda.is_available() else False
list, list_or = [], []
quant_t_, quant_b_ = [], []
test_dir_path = '/home/komleva/vqvae2/С004_for_training/test'
i = 0
for img in os.listdir(test_dir_path):
    im = pydicom.dcmread(test_dir_path+'/'+img, force=True).pixel_array
    im = im/( 65535)
    list_or.append(im)
    model = VQVAE(first_stride=4, second_stride=2)#.cuda() if cuda else VQVAE()
    model.load_state_dict(torch.load('/home/komleva/vqvae2/out/MY/0/checkpoint/vqvae_100.pt'))
    model.eval()
    
    #model.cuda()
    #original_img = Variable(im.type(Tensor))
    
    with  torch.no_grad():
        i = i+1
        print(i)
        #tempScale = torch.zeros((total, len(scale))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(scale)))
        #tmpScale[:, j] = torch.from_numpy(scale).to(tmpScale)
        un_im = torch.from_numpy(im).unsqueeze(0)
        un_im = un_im.unsqueeze(1)
        quant_t, quant_b, _, id_t, id_b = model.encode(un_im.float())
        quant_t_.append(quant_t)
        quant_b_.append(quant_b)
        out, _ = model( un_im.float())
        if i ==150:
            plt.imsave("150.png", torch.squeeze(out), cmap='gray')
    

        list.append(out)
print(len(list))
np.save('quant_t_.npy', np.stack(quant_t_))
np.save('quant_b_.npy', np.stack(quant_b_))