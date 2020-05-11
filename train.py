import argparse
import os
import torch

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'trianing stage 1 or 2')
    parser.add_argument('--save_path', type = str, default = './models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 10, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 10000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--finetune_path', type = str, default = '', help = 'load the pre-trained model with certain epoch, None for pre-training')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 501, help = 'number of epochs of training')
    parser.add_argument('--train_batch_size', type = int, default = 1, help = 'size of the training batches for single GPU')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'size of the validation batches for single GPU')
    parser.add_argument('--lr_g', type = float, default = 0.0001, help = 'Adam: learning rate for G')
    parser.add_argument('--lr_d', type = float, default = 0.0001, help = 'Adam: learning rate for D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0.0, help = 'weight decay for optimizer')
    parser.add_argument('--grad_clip_norm', type = float, default = 0.1, help = 'weight clipping for optimizer')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'epoch', help = 'lr decrease mode, by_epoch or by_iter')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 100, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 100000, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lambda_pixel', type = float, default = 10, help = 'coefficient for L1 / L2 Loss, please fix it to 1')
    parser.add_argument('--lambda_gan', type = float, default = 1, help = 'coefficient for GAN Loss')
    parser.add_argument('--lambda_percep', type = float, default = 5, help = 'coefficient for perceptual Loss')
    parser.add_argument('--lambda_color', type = float, default = 1, help = 'coefficient for perceptual Loss')
    # GAN parameters
    parser.add_argument('--gan_mode', type = str, default = 'WGAN', help = 'type of GAN: [LSGAN | WGAN], WGAN is recommended')
    parser.add_argument('--additional_training_d', type = int, default = 1, help = 'number of training D more times than G')
    # Network initialization parameters
    parser.add_argument('--net_mode', type = str, default = 'uresnet', help = 'task name for loading networks')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Dataset parameters
    parser.add_argument('--in_path_train', type = str, \
        default = 'E:\\SenseTime\\collect_data_v2\\train\\qbayer_input_16bit', help = 'input baseroot')
    parser.add_argument('--RGBout_path_train', type = str, \
        default = 'E:\\SenseTime\\collect_data_v2\\train\\srgb_target_8bit', help = 'target baseroot')
    parser.add_argument('--in_path_val', type = str, \
        default = 'E:\\SenseTime\\collect_data_v2\\val\\qbayer_input_16bit', help = 'input baseroot')
    parser.add_argument('--RGBout_path_val', type = str, \
        default = 'E:\\SenseTime\\collect_data_v2\\val\\srgb_target_8bit', help = 'target baseroot')
    parser.add_argument('--divide_rate', type = float, default = 0.05, help = 'validation set rate')
    parser.add_argument('--shuffle', type = bool, default = True, help = 'the training and validation set should be shuffled')
    parser.add_argument('--color_bias_aug', type = bool, default = False, help = 'whether to perform color bias aug')
    parser.add_argument('--color_bias_level', type = bool, default = 0.05, help = 'color bias level')
    parser.add_argument('--noise_aug', type = bool, default = True, help = 'whether to perform noise aug')
    parser.add_argument('--noise_level', type = str, default = 0.05, help = 'Gaussian noise level')
    parser.add_argument('--random_crop', type = bool, default = True, help = 'random crop size')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #                  Train
    # ----------------------------------------
    if opt.pre_train:
        trainer.Trainer(opt)
    else:
        trainer.Trainer_GAN(opt)
