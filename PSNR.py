# -*- coding = utf-8 -*-
# @Time: 2022/7/5 20:51
# @Author: Ruonan Yi
# @File: PSNR.py
# @software: PyCharm

import cv2
import numpy as np
import math
import os
import config as c
from tqdm import trange

'''
def psnr1(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0) ** 2 )   #求MSE
    if mse < 1.0e-10:
        return 100
    # max=np.max(img1)
    return 10 * math.log10(10000 ** 2 / mse) #返回PSNR


def psnrLDR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
'''

def psnr(img1, img2, type= 'ldr'):   #img1是cover,img2是stego
    if type == 'hdr':
        maxvalue = np.max(img1)  # 动态范围， hdr是取最大值
    elif type == 'ldr':
        maxvalue = 255  # ldr 是255
    #rmse = math.sqrt(np.sum((img1 - img2) ** 2) / img2.size)
    mse = np.mean((img1 - img2)**2)
    if mse < 10**(-5):
        return float('inf')
    psnr = 20 * math.log10(maxvalue / math.sqrt(mse))
    return psnr


def calculate_psnr(img1, img2, type, way='getL'):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 3:  #3维
        if img1.shape[2] == 3:   # img.shape = (256,256,3)，获取图像的大小
            if way == 'getL':  #转换到YUV空间，计算Y（亮度）分量的PSNR
                img1 = get_luminance(img1)  #变成二维阵列
                img2 = get_luminance(img2)
                return psnr(img1, img2, type)
            if way == 'mean':   #计算RGB三个通道的psnr，再求平均,一般不用
                psnrs = []
                for i in range(3):  #0，1，2
                    psnrs.append(psnr(img1, img2, type))
                return np.array(psnrs).mean()   #求3次的平均
    else:
        raise ValueError('Wrong input image dimensions.')


def get_luminance(hdr):    #获取亮度，三通道RGB彩色图像
    #L = 0.2126 * hdr[:, :, 2] + 0.7152 * hdr[:, :, 1] + 0.0722 * hdr[:, :, 0]
    #L = 0.299 * hdr[:, :, 2] + 0.587 * hdr[:, :, 1] + 0.114 * hdr[:, :, 0]
    L = 0.0722 * hdr[:, :, 2] + 0.7152 * hdr[:, :, 1] + 0.2126 * hdr[:, :, 0]
    return L

if __name__ == '__main__':
    directory_name_cover = c.TEST_PATH_cover
    directory_name_steg_1 = c.TEST_PATH_steg_1
    directory_name_steg_2 = c.TEST_PATH_steg_2
    f = open('./results_PSNR_SSIM.txt', 'a+')
    psnr_c1_ldr = []
    psnr_c2_ldr = []
    for filename in os.listdir(directory_name_cover):
        if filename[-3:] == 'png':
            img_cover = cv2.imread(directory_name_cover + filename)
            img_steg_1 = cv2.imread(directory_name_steg_1 + filename)
            img_steg_2 = cv2.imread(directory_name_steg_2 + filename)
            img_cover = img_cover[...,::-1]  # R,G,B
            img_steg_1 = img_steg_1[...,::-1]
            img_steg_2 = img_steg_2[..., ::-1]
            psnr_temp_c1_ldr = calculate_psnr(img_cover, img_steg_1, type='ldr')
            #print(psnr_temp_c1_ldr)
            psnr_c1_ldr.append(psnr_temp_c1_ldr)
            psnr_temp_c2_ldr = calculate_psnr(img_cover, img_steg_2, type='ldr')
            #print(psnr_temp_c2_ldr)
            psnr_c2_ldr.append(psnr_temp_c2_ldr)
    f.write(
            "[test dataset]  " + c.val_dataset
            + "  [train epoch] " + str(c.trained_epoch) + ", "
            + "C1_ldr average psnr: " + str(np.mean(psnr_c1_ldr)) + ", "
            + "C2_ldr average psnr: " + str(np.mean(psnr_c2_ldr)) + "\n"
    )
    f.close()

