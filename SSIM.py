# -*- coding = utf-8 -*-
# @Time: 2022/7/5 21:01
# @Author: Ruonan Yi
# @File: SSIM.py
# @software: PyCharm

import cv2
import numpy as np
import math
import os
import config as c
from tqdm import trange

def ssim(img1, img2, type='ldr'):  # 接受256，256的参数
    if type == 'hdr':
        L = np.max(img1)  # 动态范围， hdr是取最大值
    elif type == 'ldr':
        L = 255  # ldr 是255
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    img1 = img1.astype(np.float64)  # 转为双精度浮点数float64
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)  # 高斯窗口，参数默认为11 1.5
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # 滑动窗口
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    # ssim 公式， 最后求均值
    return ssim_map.mean()


def calculate_ssim(img1, img2, type, way='getL'):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)  # 如果本来是灰度图，直接计算
    elif img1.ndim == 3:
        if img1.shape[2] == 3:  # 如果是256，256，3 (三通道彩色图)
            if way == 'getL':
                img1 = get_luminance(img1)
                img2 = get_luminance(img2)  # 变成256，256
                return ssim(img1, img2, type)
            if way == 'mean':
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2, type))
                return np.array(ssims).mean()
        elif img1.shape[2] == 1:  # 256,256,1
            return ssim(np.squeeze(img1), np.squeeze(img2))  # 去掉最后一个维度后计算
    else:
        raise ValueError('Wrong input image dimensions.')

def get_luminance(hdr):    #获取亮度，三通道RGB彩色图像
    #L = 0.2126 * hdr[:, :, 2] + 0.7152 * hdr[:, :, 1] + 0.0722 * hdr[:, :, 0]
    #L = 0.299 * hdr[:, :, 2] + 0.587 * hdr[:, :, 1] + 0.114 * hdr[:, :, 0]
    L = 0.0722 * hdr[:, :, 2] + 0.7152 * hdr[:, :, 1] + 0.2126 * hdr[:, :, 0]
    return L


if __name__ == '__main__':
    '''
    f = open('all_SSIM_shuffle.txt', 'a+')
    for num in trange(1000):
        secret = cv2.imread(r'../TestResults/pics/secret_' + str(num) + '.png')
        extract = cv2.imread(r'../TestResults/pics/extract_' + str(num) + '.png')
        ssimIndex = calculate_ssim(secret, extract)
        f.write(str(ssimIndex) + '\n')
    f.close()
    '''
    directory_name_cover = c.TEST_PATH_cover
    directory_name_steg_1 = c.TEST_PATH_steg_1
    directory_name_steg_2 = c.TEST_PATH_steg_2
    f = open('./results_PSNR_SSIM.txt', 'a+')
    ssim_c1_ldr = []
    ssim_c2_ldr = []
    for filename in os.listdir(directory_name_cover):
        if filename[-3:] == 'png':
            img_cover = cv2.imread(directory_name_cover + filename)
            img_steg_1 = cv2.imread(directory_name_steg_1 + filename)
            img_steg_2 = cv2.imread(directory_name_steg_2 + filename)
            img_cover = img_cover[..., ::-1]  # R,G,B
            img_steg_1 = img_steg_1[..., ::-1]
            img_steg_2 = img_steg_2[..., ::-1]
            ssim_temp_c1_ldr = calculate_ssim(img_cover, img_steg_1, type='ldr')
            # print(ssim_temp_c1_ldr)
            ssim_c1_ldr.append(ssim_temp_c1_ldr)
            ssim_temp_c2_ldr = calculate_ssim(img_cover, img_steg_2, type='ldr')
            # print(ssim_temp_c2_ldr)
            ssim_c2_ldr.append(ssim_temp_c2_ldr)
    f.write(
        "[test dataset]  " + c.val_dataset
        + "  [train epoch] " + str(c.trained_epoch) + ", "
        + "C1_ldr average ssim: " + str(np.mean(ssim_c1_ldr)) + ", "
        + "C2_ldr average ssim: " + str(np.mean(ssim_c2_ldr)) + "\n"
    )
    f.close()