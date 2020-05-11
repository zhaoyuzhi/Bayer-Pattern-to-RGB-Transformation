import argparse
import os
import cv2
import numpy as np

import utils

def rgb2bayer(img): # input: BGR image
    bayer = np.zeros(img.shape, dtype = np.uint8)
    bayer[0::2, 0::2, 0] = img[0::2, 0::2, 2]
    bayer[0::2, 1::2, 1] = img[0::2, 1::2, 1]
    bayer[1::2, 0::2, 1] = img[1::2, 0::2, 1]
    bayer[1::2, 1::2, 2] = img[1::2, 1::2, 0]
    return bayer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type = str, \
        default = '', \
            help = 'input baseroot')
    parser.add_argument('--dst_path', type = str, \
        default = '', \
            help = 'target baseroot')
    opt = parser.parse_args()
    print(opt)

    src_imglist = utils.get_jpgs(opt.src_path)

    utils.check_path(opt.dst_path)

    for i in range(len(src_imglist)):
        img_name = src_imglist[i]
        src_name = os.path.join(opt.src_path, img_name)
        dst_name = os.path.join(opt.dst_path, img_name)
        img = cv2.imread(src_name)
        bayer = rgb2bayer(img)
        cv2.imwrite(dst_name, bayer)
