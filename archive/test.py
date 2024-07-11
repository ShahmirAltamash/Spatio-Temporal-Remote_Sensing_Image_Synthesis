import torch
import os
import matplotlib.pyplot as plt 

import config
import utils
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SpaceNetDataset,SpaceNet_testsetter
from models import Discriminator, Generator, UNet
from metrics import LPIPSMetric, SSIM , PSNR, F1_score
from cleanfid import fid
from cleanfid.features import build_feature_extractor
from torchvision.utils import save_image

def extract(data):

    indices= torch.nonzero(data).flatten()
    return indices

def test(seg , gen, loader, save_path, gt_path, mask_path, flag=0, save = 0):
    loop = tqdm(loader, leave=True)

    gen.eval()
    Lpips_list = []
    ssim_list = []
    psnr_list = []

    Lpips_metric = LPIPSMetric()
    ssim_metric = SSIM()
    psnr_metric = PSNR()
    with torch.no_grad():
        for idx, data in enumerate(loop):
            input_img, y ,ind= data   #input_img = input image + mask @4 channels, y = ground truth image

            input_img, y = input_img.to(config.DEVICE), y.to(config.DEVICE) 

            #input_img, y ,loc, img_number, is_final=input_img.to(config.DEVICE), y.to(config.DEVICE) ,loc.to(config.DEVICE), img_number.to(config.DEVICE), is_final.to(config.DEVICE)
            #y_mask = seg(y) # check image sizes after segmentation 512 size.   
            # day0_seg_mask=seg(input_img)
            #x = torch.cat((input_img,y_mask),dim=1) #size ---> (64,4,512,512)
            # with torch.cuda.amp.autocast():
            masks=input_img[:,3,:,:]
            fake_image = gen(input_img) 
            if save ==1:
                for i in range(input_img.shape[0]):
                    index = int(ind[i])
                    
                    if flag == 1:
                        save_image(masks[i].cpu().detach(), f'{mask_path}_mask_{index}.png', normalize=True)
                        save_image(y[i].cpu().detach(), f'{gt_path}_gt_{index}.png', normalize=True)   #ground truth image

                    save_image(fake_image[i].cpu().detach(), f'{save_path}_generated_{index}.png', normalize=True)

            Lpips_metric.update(fake_image.round().detach().cpu(), y.detach().cpu())
            ssim_metric.update(fake_image.round().detach().cpu(), y.detach().cpu())
            psnr_metric.update(fake_image.round().detach().cpu(), y.detach().cpu())
            
            #Stores values in dictionary for each batch of data
            Lpips_list.append(Lpips_metric.compute())
            ssim_list.append(ssim_metric.compute())
            psnr_list.append(psnr_metric.compute())    

               
    
    return Lpips_list,ssim_list,psnr_list
   
                    
def avg(list):
    return sum(list)/len(list)


def main():


    # generator= Generator(in_channels=4).to(config.DEVICE)
    # segmentation= UNet().to(config.DEVICE)

    # generator = utils.load_model_raw(config.CHECKPOINT_GEN_VAL)
    # utils.load_model(config.CHECKPOINT_SEG,segmentation)

    # test_dataset= SpaceNetDataset(train=2)
    # test_loader= DataLoader(test_dataset, 24, shuffle=False)
    # # save(segmentation,generator, test_loader)

    '''
    Latest code 
    '''
        
    generator= Generator(in_channels=4).to(config.DEVICE)
    segmentation= UNet().to(config.DEVICE)

    gen_path =  "Saved Models/gen_best_val.pth.tar"
    gen2_path = "Saved Models/att_gen_best_epoch_77.pth.tar"
    generator = utils.load_model_raw(gen_path)
    generator2 = utils.load_model_raw(gen2_path)
    segmentation= utils.load_model_raw(config.CHECKPOINT_SEG)

    test_dataset= SpaceNetDataset(train=2)
    test_loader= DataLoader(test_dataset, config.BATCH_SIZE, shuffle=False)

    lpips_r=[]
    ssim_r=[]
    psnr_r=[]
    lpips_s=[]
    ssim_s=[]
    psnr_s=[]
    attn_path="Generated Images\Att GAN Results\\"
    generator_path="Generated Images\cGAN Results\\"
    gt_path="Generated Images\Ground Truths\\"
    mask_path="Generated Images\Masks\\"

    print("UNet Generator")
    lpips_s, ssim_s, psnr_s = test(segmentation,generator,test_loader,generator_path,gt_path,mask_path, flag=0, save =0)
    print( "LPIPS UNet Generator", avg(lpips_s),  "SSIM UNet Generator", avg(ssim_s), "PSNR UNet Generator", avg(psnr_s))
    
    print("Attention UNet Generator")
    lpips_r, ssim_r, psnr_r = test(segmentation,generator2,test_loader,attn_path,gt_path,mask_path, flag=0, save=0)
    print( "LPIPS Attention UNet Generator", avg(lpips_r),  "SSIM Attention UNet Generator", avg(ssim_r), "PSNRAttention UNet Generator", avg(psnr_r))
   

    
    # columns = ('LPIPS Attention UNet Generator', ' LPIPS UNet Generator' , 'SSIM Attention UNet Generator' , ' SSIM UNet Generator', 'PSNR Attention UNet Generator' , ' PSNR UNet Generator')
    # rows = ['values']
    # data = [[avg(lpips_r) , avg(lpips_s) , avg(ssim_r) , avg(ssim_s),avg(psnr_r), avg(psnr_s)]]
    # fig , ax = plt.subplots()
    # ax.axis('tight')
    # ax.axis('off')
    # ax.table(cellText=data, colLabels=columns, rowLabels=rows, loc='center')
    # plt.show()

    # # fid_score = fid.compute_fid(real_path, fake_path, mode="clean", verbose=True, num_workers=0)

    # print(data)

    

if __name__ == "__main__":
    main()