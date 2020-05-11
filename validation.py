import argparse
import os
import torch

import utils
import dataset

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'pre_train or not')
    parser.add_argument('--finetune_path', type = str, default = './models/train_all/G2_iso6400_epoch400_bs1.pth', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the testing batches for single GPU')
    parser.add_argument('--num_workers', type = int, default = 2, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--val_path', type = str, default = './validation', help = 'saving path that is a folder')
    parser.add_argument('--task_name', type = str, default = 'val_320patch_train_pre', help = 'task name for loading networks, saving, and log')
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
    parser.add_argument('--in_path_val', type = str, \
        default = 'E:\\SenseTime\\Quad-Bayer to RGB Mapping\\data\\collect_data_v2\\val_320patch\\qbayer_input_16bit', \
            help = 'input baseroot')
    parser.add_argument('--RGBout_path_val', type = str, \
        default = 'E:\\SenseTime\\Quad-Bayer to RGB Mapping\\data\\collect_data_v2\\val_320patch\\srgb_target_8bit', \
            help = 'target baseroot')
    parser.add_argument('--Qbayerout_path_val', type = str, \
        default = 'E:\\SenseTime\\Quad-Bayer to RGB Mapping\\data\\collect_data_v2\\val_320patch\\qbayer_target_16bit', \
            help = 'target baseroot')
    parser.add_argument('--shuffle', type = bool, default = False, help = 'the training and validation set should be shuffled')
    parser.add_argument('--color_bias_aug', type = bool, default = False, help = 'color_bias_aug')
    parser.add_argument('--color_bias_level', type = bool, default = 0.05, help = 'color_bias_level')
    parser.add_argument('--noise_aug', type = bool, default = True, help = 'noise_aug')
    parser.add_argument('--iso', type = int, default = 6400, help = 'noise_level, according to ISO value')
    parser.add_argument('--random_crop', type = bool, default = True, help = 'random_crop')
    parser.add_argument('--crop_size', type = int, default = 320, help = 'single patch size')
    parser.add_argument('--extra_process_train_data', type = bool, default = True, help = 'recover short exposure data')
    parser.add_argument('--cover_long_exposure', type = bool, default = False, help = 'set long exposure to 0')
    parser.add_argument('--short_expo_per_pattern', type = int, default = 2, help = 'the number of exposure pixel of 2*2 square')
    opt = parser.parse_args()

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator(opt).cuda()
    namelist = utils.get_jpgs(opt.in_path_val)
    test_dataset = dataset.Qbayer2RGB_dataset(opt, 'val', namelist)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    sample_folder = os.path.join(opt.val_path, opt.task_name)
    utils.check_path(sample_folder)

    # forward
    val_PSNR = 0
    val_SSIM = 0
    for i, (in_img, RGBout_img, path) in enumerate(test_loader):
        # To device
        # A is for input image, B is for target image
        in_img = in_img.cuda()
        RGBout_img = RGBout_img.cuda()
        #print(path)

        # Forward propagation
        with torch.no_grad():
            out = generator(in_img)

        # Sample data every iter
        img_list = [out, RGBout_img]
        name_list = ['pred', 'gt']
        utils.save_sample_png(sample_folder = sample_folder, sample_name = '%d' % (i), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
        
        # PSNR
        val_PSNR_this = utils.psnr(out, RGBout_img, 1) * in_img.shape[0]
        print('The %d-th image PSNR %.4f' % (i, val_PSNR_this))
        val_PSNR = val_PSNR + val_PSNR_this
        # SSIM
        val_SSIM_this = utils.ssim(out, RGBout_img) * in_img.shape[0]
        print('The %d-th image SSIM %.4f' % (i, val_SSIM_this))
        val_SSIM = val_SSIM + val_SSIM_this
        
    val_PSNR = val_PSNR / len(namelist)
    val_SSIM = val_SSIM / len(namelist)
    print('The average PSNR equals to', val_PSNR)
    print('The average SSIM equals to', val_SSIM)
