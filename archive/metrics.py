import torch
import torch.nn as nn
import numpy as np
from torchmetrics import JaccardIndex
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import fid, inception
from torchmetrics.classification import Dice as F1
import config
from torchmetrics.functional import f1_score
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity as lpips


## New Imports

import lpips



def F1_score(z1,z2):
    return f1_score(z1, z2, task='binary')    



def change(test_mask_24, test_mask):
    test_mask_24=test_mask_24.cpu().numpy()
    test_mask=test_mask.cpu().numpy()
    difference_mask2 = np.logical_xor(test_mask_24, test_mask)

    return difference_mask2

def change_absolute(test_mask_24, test_mask):
    difference_mask = torch.sum(torch.sum(torch.sum(torch.abs(test_mask_24 - test_mask),axis=-1),axis=-1),axis=-1) 
    
    return difference_mask

####################################################################################################################################
# LPIPS Metric
class LPIPSMetric:
    def __init__(self):
        #self.loss_fn = lpips.LPIPS(net='alex', verbose=False)
        self.loss_fn = lpips.LPIPS()
        self.reset()

    def reset(self):
        self.sum_lpips = 0
        self.count = 0

    def update(self, output, target):
        loss = self.loss_fn(output, target)
        self.sum_lpips += loss.mean().item()  # Compute mean loss across the batch
        self.count += 1

    def compute(self):
        if self.count == 0:
            return 0  # Return 0 if no examples were processed
        return self.sum_lpips / self.count
    
# SSIM Metric

class SSIM:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_ssim = 0
        self.count = 0

    def update(self, output, target):
        score = ssim(output, target, data_range=255, size_average=False)
        self.sum_ssim += score.mean().item()  # Compute mean score across the batch
        self.count += 1

    def compute(self):
        return self.sum_ssim / self.count
    
# PSNR Metric
class PSNR:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_psnr = 0
        self.count = 0

    def update(self, output, target):
        mse = torch.mean((output - target) ** 2)
        psnr = 20 * torch.log10(255 / torch.sqrt(mse))
        self.sum_psnr += psnr.item()
        self.count += 1

    def compute(self):
        return self.sum_psnr / self.count



if __name__ == "__main__":

    z1 = torch.randint(low=0, high=1, size=(8, 1, 256, 256)).to(torch.float)
    z2 = torch.randint(low=0, high=1, size=(8, 1, 256, 256)).to(torch.float)
    ssim_val = ssim( z1, z2, data_range=1, size_average=False) # return (N,)

    print("result ", ssim_val)
