import os
import math
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

'''
def build_train_val_set(basepath, divide_rate = 0.1, shuffle = True):
    # divide whole dataset into two parts
    imglist = get_jpgs(basepath)
    sample_amount = int(len(imglist) * divide_rate)
    val_imglist = random.sample(imglist, sample_amount)
    train_imglist = [item for item in imglist if item not in val_imglist]
    # shuffle
    if shuffle:
        random.shuffle(train_imglist)
        random.shuffle(val_imglist)
    return train_imglist, val_imglist
'''

def build_file_set(basepath, imglist):
    filelist = []
    for item in imglist:
        wholepath = os.path.join(basepath, item)
        filelist.append(wholepath)
    return filelist

class Qbayer2RGB_dataset(Dataset):
    def __init__(self, opt, tag, imglist):
        self.opt = opt
        if tag == 'train':
            self.in_filelist = build_file_set(opt.in_path_train, imglist)
            self.RGBout_filelist = build_file_set(opt.RGBout_path_train, imglist)
        if tag == 'val':
            self.in_filelist = build_file_set(opt.in_path_val, imglist)
            self.RGBout_filelist = build_file_set(opt.RGBout_path_val, imglist)

    # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w
    
    def __getitem__(self, index):

        # Path of RAW and RGB and read images
        in_path = self.in_filelist[index]
        RGBout_path = self.RGBout_filelist[index]
        in_img = cv2.imread(in_path, -1)
        RGBout_img = cv2.imread(RGBout_path, -1)

        # Normalize
        in_img = in_img.astype(np.float) / 255.0
        RGBout_img = RGBout_img.astype(np.float) / 255.0
        # Gamma correction
        #in_img = in_img ** (1 / 2.2)

        # Data augmentation
        # color bias
        if self.opt.color_bias_aug:
            # multiplication item
            blue_coeff = np.random.random_sample() + 1.5
            green_coeff = 1 # np.random.random_sample()
            red_coeff = np.random.random_sample() + 1.5
            # adding item
            blue_offset = np.random.random_sample() * 2 * self.opt.color_bias_level - self.opt.color_bias_level
            green_offset = np.random.random_sample() * 2 * self.opt.color_bias_level - self.opt.color_bias_level
            red_offset = np.random.random_sample() * 2 * self.opt.color_bias_level - self.opt.color_bias_level
            # perform
            in_img[0::2, 0::2, 0] = in_img[0::2, 0::2, 0] + red_coeff * red_offset
            in_img[0::2, 1::2, 1] = in_img[0::2, 1::2, 1] + green_coeff * green_offset
            in_img[1::2, 0::2, 1] = in_img[1::2, 0::2, 1] + green_coeff * green_offset
            in_img[1::2, 1::2, 2] = in_img[1::2, 1::2, 2] + blue_coeff * blue_offset
            in_img = np.clip(in_img, 0, 1)
        # additive noise
        if self.opt.noise_aug:
            noise = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = in_img.shape)
            in_img += noise
            in_img = np.clip(in_img, 0, 1)

        # To tensor
        in_img = torch.from_numpy(in_img).float().permute(2, 0, 1).contiguous()
        RGBout_img = torch.from_numpy(RGBout_img).float().permute(2, 0, 1).contiguous()

        return in_img, RGBout_img

    def __len__(self):
        return len(self.in_filelist)
