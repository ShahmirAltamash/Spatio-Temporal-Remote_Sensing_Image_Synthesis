import torch
import config
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


class SpaceNetDataset(Dataset):
    def __init__(self, root_dir=config.TRAIN_DIR, mask_root_dir=config.MASK_DIR_TRAIN, img_transform=config.IMG_TRANSFORM, mask_transform=config.MASK_TRANSFORM, train=0):
        if train == 0:
            self.csv = pd.read_csv(config.TRAIN_CSV_DIR)
        elif train == 1:
            self.csv = pd.read_csv(config.VAL_CSV_DIR)
        elif train == 2:
            self.csv = pd.read_csv(config.TEST_CSV_DIR)
        else:
            print("Incorrect datatype. Enter 0, 1, or 2.")
            return -1
        self.working_csv = self.csv.loc[self.csv['mask'].notna()]
        self.root_dir = root_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.mask_root = mask_root_dir

    def __len__(self):
        return len(self.working_csv)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() 
        dir_name = self.working_csv.iloc[idx, 2]
        mask_inf = self.working_csv.iloc[idx, 3]
        # mask_il = mask_inf.str.split("_")
        # mask_num = int(mask_il['fname'][3])-1
        
        temp_df = self.working_csv.loc[self.working_csv['image_dir_name'] == dir_name]
        ind = temp_df[temp_df['fname'] == mask_inf].index[0]
        img_path = self.root_dir/temp_df.iloc[0,9]
        image = Image.open(img_path)
        D0_mask_path = self.mask_root/temp_df.iloc[0,15]
        image_D0_mask = Image.open(D0_mask_path)
        mask_path = self.mask_root/temp_df.loc[ind, "mask_path"]       
        mask = Image.open(mask_path)
        ground_truth_path = self.root_dir/temp_df.loc[ind, "images_masked"] 
        ground_truth = Image.open(ground_truth_path)
    
        d0_img = self.img_transform(image)
        D0_mask = self.mask_transform(image_D0_mask)
        mask = self.mask_transform(mask)
        gt = self.img_transform(ground_truth)
        d0_img = d0_img[:3,:,:]
        p2p_in = torch.cat([d0_img,mask], dim = 0)
        idx =  torch.tensor(idx,dtype=torch.int8)
        return p2p_in,gt[:3,:,:], idx 



class SpaceNet_testsetter(Dataset):
    def __init__(self, csv_file = config.EVAL_CSV_DIR, root_dir = config.EVAL_DIR):
    # def __init__(self, csv_file = 'archive\output_csvs\df_test_untidy.csv', root_dir = config.EVAL_DIR):
        self.csv = pd.read_csv(csv_file)
        #print(self.csv)
        self.root_dir = root_dir

        # Define image and mask transformations
        self.img_transform = config.IMG_TRANSFORM

        

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        location = self.csv.loc[idx, 'image_dir_name']
        image_path = self.root_dir/self.csv.loc[idx, 'images_masked']
        d0_image_path = self.root_dir/self.csv[self.csv['image_dir_name'] == location].iloc[0, 9]
        loc_id = int(self.csv.loc[idx, 'loc_id'])
        image_num = int(self.csv.loc[idx, 'image_num'])
        is_final = self.csv.loc[idx, 'is_final_image']

        image = Image.open(image_path)
        d0_image = Image.open(d0_image_path)

        img_t = self.img_transform(image)
        d0_img_t = self.img_transform(d0_image)

        img_t = img_t[:3,:,:]
        d0_img_t = d0_img_t[:3,:,:]

        return d0_img_t, img_t, loc_id, image_num, is_final




class SN_seg_setter(Dataset):
    def __init__(self, root_dir=config.TRAIN_DIR, mask_root_dir=config.MASK_DIR_TRAIN, img_transform=config.IMG_TRANSFORM, mask_transform=config.MASK_TRANSFORM, train=0):

        if train == 0:
            self.csv = pd.read_csv(config.TRAIN_CSV_DIR)
        elif train == 1:
            self.csv = pd.read_csv(config.VAL_CSV_DIR)
        elif train == 2:
            self.csv = pd.read_csv(config.TEST_CSV_DIR)
        else:
            print("Incorrect datatype. Enter 0, 1, or 2.")
            return -1
        #self.csv = pd.read_csv(csv_file)
        self.working_csv = self.csv.loc[self.csv['mask'].notna()]
        self.root_dir = root_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.mask_root = mask_root_dir
       

    def __len__(self):
        
        return len(self.working_csv)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        img_path = self.root_dir/self.working_csv.iloc[idx,9]
        image = Image.open(img_path)
        mask_path = self.mask_root/self.working_csv.iloc[idx,15]
        mask = Image.open(mask_path)

       
        raw_image = self.img_transform(image)
        mask = self.mask_transform(mask)

        return raw_image[:3,:,:],mask

       


if __name__ == "__main__":
    spacenet_dataset = SpaceNet_testsetter()
    seg_dataset = SN_seg_setter()


    # print(spacenet_dataset[0][0].shape)
    # z = torch.cat([spacenet_dataset[0][0][:3], spacenet_dataset[0][1]], dim=2).detach().numpy().transpose((1, 2, 0))
    # m = spacenet_dataset[0][0][3].detach().numpy()
    # plt.figure(1)
    # plt.imshow(z)
    # plt.show()

    # plt.figure(2)
    # plt.imshow(m, cmap="gray")
    # plt.show()



    