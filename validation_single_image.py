import argparse
import os
import cv2
import torch
import numpy as np
import math
import random

import utils

class Pair_Processing():
    def __init__(self, opt):
        # General
        self.opt = opt
        # ISO
        if opt.iso == 400:
            self.A = 0.00000110301
            self.B = -0.0000216196
            self.CC = 0.0000237296
            self.DD = 0.0000385458
            self.t = 0.19191
        if opt.iso == 800:
            self.A = 0.00000188468
            self.B = -0.0000359069
            self.CC = 0.0000433813
            self.DD = 0.0000424857
            self.t = 0.105228
        if opt.iso == 1600:
            self.A = 0.0000110783
            self.B = -0.000179802
            self.CC = 0.000706091
            self.DD = 0.00799151
            self.t = 9.19985
        if opt.iso == 3200:
            self.A = 0.00000689484
            self.B = 0.000239979
            self.CC = -0.000109656 
            self.DD = -0.0000743398
            self.t = 0.335491
        if opt.iso == 6400:
            self.A = 0.0000157458
            self.B = 0.0
            self.CC = 0.00001
            self.DD = 0.0000302316
            self.t = 0.0283857
    
    # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def process(self, in_path, RGBout_img):

        in_img = cv2.imread(in_path, -1)
        RGBout_img = cv2.imread(RGBout_img, -1)

        ### Specify the pos for short and long exposure pixels
        if self.opt.short_expo_per_pattern == 2:
            short_pos = [[0,0], [1,1]]
            long_pos = [[0,1], [1,0]]
        if self.opt.short_expo_per_pattern == 3:
            short_pos = [[0,0], [0,1], [1,0]]
            long_pos = [[1,1]]

        in_img = in_img.astype(np.float) / 65535.0
        RGBout_img = RGBout_img.astype(np.float) / 255.0
   
        if self.opt.color_bias_aug:  ### Only for short exposure pixels
            # multiplication item
            blue_coeff = np.random.random_sample() + 1.5
            green_coeff = 1 # np.random.random_sample()
            red_coeff = np.random.random_sample() + 1.5
            # adding item
            blue_offset = np.random.random_sample() * 2 * self.opt.color_bias_level - self.opt.color_bias_level
            green_offset = np.random.random_sample() * 2 * self.opt.color_bias_level - self.opt.color_bias_level
            red_offset = np.random.random_sample() * 2 * self.opt.color_bias_level - self.opt.color_bias_level
            # perform
            for pos in short_pos:
                in_img[pos[0]::4,pos[1]::4,0] = in_img[pos[0]::4,pos[1]::4,0] + blue_coeff * blue_offset
                in_img[pos[0]::4,pos[1]+2::4,1] = in_img[pos[0]::4,pos[1]+2::4,1] + green_coeff * green_offset
                in_img[pos[0]+2::4,pos[1]::4,1] = in_img[pos[0]+2::4,pos[1]::4,1] + green_coeff * green_offset
                in_img[pos[0]+2::4,pos[1]+2::4,2] = in_img[pos[0]+2::4,pos[1]+2::4,2] + red_coeff * red_offset
            in_img = np.clip(in_img, 0, 1)   

        if self.opt.noise_aug:   ### Only for short exposure pixels
            if self.opt.iso == 100:
                for pos in short_pos:
                    in_img[pos[0]::4,pos[1]::4,0] += np.sqrt(in_img[pos[0]::4,pos[1]::4,0] * 0.000034)
                    in_img[pos[0]::4,pos[1]+2::4,1] += np.sqrt(in_img[pos[0]::4,pos[1]+2::4,1] * 0.000034)
                    in_img[pos[0]+2::4,pos[1]::4,1] += np.sqrt(in_img[pos[0]+2::4,pos[1]::4,1] * 0.000034)
                    in_img[pos[0]+2::4,pos[1]+2::4,2] += np.sqrt(in_img[pos[0]+2::4,pos[1]+2::4,2] * 0.000034)
            else:
                for pos in short_pos:
                    in_img[pos[0]::4,pos[1]::4,0] += np.sqrt(self.A + self.B * np.sqrt(in_img[pos[0]::4,pos[1]::4,0]) + \
                        self.CC * in_img[pos[0]::4,pos[1]::4,0] + self.DD * \
                            (1 - np.power(math.e, - in_img[pos[0]::4,pos[1]::4,0] / self.t)))
                    in_img[pos[0]::4,pos[1]+2::4,1] += np.sqrt(self.A + self.B * np.sqrt(in_img[pos[0]::4,pos[1]+2::4,1]) + \
                        self.CC * in_img[pos[0]::4,pos[1]+2::4,1] + self.DD * \
                            (1 - np.power(math.e, - in_img[pos[0]::4,pos[1]+2::4,1] / self.t)))
                    in_img[pos[0]+2::4,pos[1]::4,1] += np.sqrt(self.A + self.B * np.sqrt(in_img[pos[0]+2::4,pos[1]::4,1]) + \
                        self.CC * in_img[pos[0]+2::4,pos[1]::4,1] + self.DD * \
                            (1 - np.power(math.e, - in_img[pos[0]+2::4,pos[1]::4,1] / self.t)))
                    in_img[pos[0]+2::4,pos[1]+2::4,2] += np.sqrt(self.A + self.B * np.sqrt(in_img[pos[0]+2::4,pos[1]+2::4,2]) + \
                        self.CC * in_img[pos[0]+2::4,pos[1]+2::4,2] + self.DD * \
                            (1 - np.power(math.e, - in_img[pos[0]+2::4,pos[1]+2::4,2] / self.t)))
            in_img = np.clip(in_img, 0, 1)

        if self.opt.cover_long_exposure:
            for pos in long_pos:
                in_img[pos[0]::2,pos[1]::2,:] *= 0

        if self.opt.extra_process_train_data:
            for pos in short_pos:
                in_img[pos[0]::2,pos[1]::2,:] *= 4
        
        if self.opt.random_crop:
            h, w = in_img.shape[:2]
            rand_h, rand_w = self.random_crop_start(h, w, self.opt.crop_size, 4)
            in_img = in_img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]
            RGBout_img = RGBout_img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]

        # gamma correction
        in_img = in_img ** (1/2.2)
        #Qbayerout_img = Qbayerout_img ** (1/2.2)
        #RGBout_img = RGBout_img ** (2.2)

        in_img = torch.from_numpy(in_img).float().permute(2,0,1).contiguous()
        RGBout_img = torch.from_numpy(RGBout_img).float().permute(2,0,1).contiguous()

        return in_img, RGBout_img # RGBout_path, Qbayerout_path

def single_image_process(generator, image_processor, opt, addition):
    # Pre-processing
    in_img, RGBout_img = image_processor.process(opt.in_path, opt.RGB_path)
    in_img = in_img.unsqueeze(0)
    RGBout_img = RGBout_img.unsqueeze(0)

    # forward
    # To device
    in_img = in_img.cuda()
    RGBout_img = RGBout_img.cuda()
    # Forward propagation
    with torch.no_grad():
        out = generator(in_img)
    # Sample data every iter
    img_list = [in_img, out, RGBout_img]
    name_list = ['in', 'pred', 'gt']
    utils.save_sample_png(sample_folder = sample_folder, sample_name = 'single' + str(addition), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
    # PSNR
    val_PSNR_this = utils.psnr(out, RGBout_img, 1)
    print('The image PSNR %.4f' % (val_PSNR_this))

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--finetune_path', type = str, \
        #default = './models/all_loss_GAN_nopercep/G2_iso6400_epoch2000_bs12.pth', \
        default = './models/train_all/G2_iso6400_epoch500_bs1.pth', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the testing batches for single GPU')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--val_path', type = str, default = './val_results', help = 'saving path that is a folder')
    parser.add_argument('--task_name', type = str, default = 'single', help = 'task name for loading networks, saving, and log')
    parser.add_argument('--times', type = int, default = 100, help = 'the overall times of forward')
    # Network initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 32, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Dataset parameters
    parser.add_argument('--in_path', type = str, \
        default = 'E:\\SenseTime\\Quad-Bayer to RGB Mapping\\data\\collect_data_v2\\val\\qbayer_input_16bit\\DSC_4177.png', \
            help = 'input baseroot')
    parser.add_argument('--RGB_path', type = str, \
        default = 'E:\\SenseTime\\Quad-Bayer to RGB Mapping\\data\\collect_data_v2\\val\\srgb_target_8bit\\DSC_4177.png', \
            help = 'target baseroot')
    parser.add_argument('--color_bias_aug', type = bool, default = False, help = 'color_bias_aug')
    parser.add_argument('--color_bias_level', type = bool, default = 0.05, help = 'color_bias_level')
    parser.add_argument('--noise_aug', type = bool, default = True, help = 'noise_aug')
    parser.add_argument('--iso', type = int, default = 6400, help = 'noise_level, according to ISO value')
    parser.add_argument('--random_crop', type = bool, default = True, help = 'random_crop')
    parser.add_argument('--crop_size', type = int, default = 1024, help = 'single patch size')
    parser.add_argument('--extra_process_train_data', type = bool, default = True, help = 'recover short exposure data')
    parser.add_argument('--cover_long_exposure', type = bool, default = False, help = 'set long exposure to 0')
    parser.add_argument('--short_expo_per_pattern', type = int, default = 2, help = 'the number of exposure pixel of 2*2 square')
    opt = parser.parse_args()

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    print(opt.in_path)
    generator = utils.create_generator_val(opt).cuda()
    image_processor = Pair_Processing(opt)
    sample_folder = os.path.join(opt.val_path, opt.task_name)
    utils.check_path(sample_folder)

    # Process
    for i in range(opt.times):
        single_image_process(generator, image_processor, opt, i)
