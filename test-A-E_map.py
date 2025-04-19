import warnings
import sys
import math
import os
import shutil
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
import tqdm
import cv2
from model import *
from imp_subnet import *
import config as c
from datasets import *
from os.path import join
import datasets
import modules.module_util as mutil
import modules.Unet_common as common
from PSNR import *
from SSIM import *
from cam_model import CAM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise





net1 = Model_3()  # 第一个IHNN
net2 = Model_3()    # 第二个IHNN
net1.cuda()
net2.cuda()

init_model(net1)
init_model(net2)
net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids)
net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)

params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))
params_trainable2 = (list(filter(lambda p: p.requires_grad, net2.parameters())))

optim1 = torch.optim.Adam(params_trainable1, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
optim2 = torch.optim.Adam(params_trainable2, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
weight_scheduler2 = torch.optim.lr_scheduler.StepLR(optim2, c.weight_step, gamma=c.gamma)

dwt = common.DWT()
iwt = common.IWT()

f = open('./results_PSNR_SSIM.txt', 'a+')
f.write(
            "[test dataset]  " + c.val_dataset
            + "  [train epoch] " + str(c.trained_epoch) + "\n"
    )
# if c.train_next:
#     load(c.MODEL_PATH + c.suffix_load + '_1.pt', net1, optim1)
#     load(c.MODEL_PATH + c.suffix_load + '_2.pt', net2, optim2)
#     load(c.MODEL_PATH + c.suffix_load + '_3.pt', net3, optim3)

if c.pretrain:
    load(c.PRETRAIN_PATH + c.suffix_pretrain + '_1.pt', net1, optim1)
    load(c.PRETRAIN_PATH + c.suffix_pretrain + '_2.pt', net2, optim2)

# clear test results images
names = os.listdir(c.TEST_PATH)
subfolders = []
for name in names:
    name_path = os.path.join(c.TEST_PATH,name)
    if os.path.isdir(name_path):
        subfolders.append(name_path)

for subfolder in subfolders:
    names = os.listdir(subfolder)
    for name in names:
        name_path = os.path.join(subfolder, name)
        if os.path.isfile(name_path):
            os.remove(name_path)
        elif os.path.isdir(name_path):
            shutil.rmtree(name_path)

network_Amap = CAM('resnet50').to(device)
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
import torch,gc
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    psnr_s1 = []
    psnr_s2 = []
    psnr_c1_hdr = []
    psnr_c2_hdr = []
    ssim_s1 = []
    ssim_s2 = []
    ssim_c1_hdr = []
    ssim_c2_hdr = []
    net1.eval()
    net2.eval()
    for i, (cover, secret_1, secret_2) in enumerate(testloader):
        cover = cover.to(device)  # channels = 3
        secret_1 = secret_1.to(device)
        secret_2 = secret_2.to(device)
        cover_RGB = cover[:, :3]
        cover_E = cover[:, 3:4]
        network_Amap.eval()
        with torch.no_grad():
            topk = 1
            prob, cls, cam = network_Amap(cover_RGB[0, :, :, :].unsqueeze(0), topk=topk)
            Amap = cam[0].unsqueeze(0)
        # 求Emap
        Emap = getEdge(cover_E).to(device)

        cover_dwt = dwt(cover_RGB)  # channels = 12
        secret_dwt_1 = dwt(secret_1)
        secret_dwt_2 = dwt(secret_2)

        Amap_dwt = dwt(Amap)  # channels = 4
        Emap_dwt = dwt(Emap)  # channels = 4
        map_dwt = torch.cat((Amap_dwt, Emap_dwt), 1)  # channels = 8
        input_dwt_1 = torch.cat((cover_dwt, map_dwt), 1)  # channels = 20
        input_dwt_1 = torch.cat((input_dwt_1, secret_dwt_1), 1)  # channels = 32

        #################
        #    forward1:   #
        #################
        output_dwt_1 = net1(input_dwt_1)  # channels = 24
        output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in,
                                             output_dwt_1.shape[1] - 4 * c.channels_in)  # channels = 12

        # get steg1
        output_steg_1 = iwt(output_steg_dwt_1)  # channels = 3

        #################
        #    forward2:   #
        #################

        input_dwt_2 = torch.cat((output_steg_dwt_1, map_dwt), 1)  # 20
        input_dwt_2 = torch.cat((input_dwt_2, secret_dwt_2), 1)

        output_dwt_2 = net2(input_dwt_2)  # channels = 36
        output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        output_z_dwt_2 = output_dwt_2.narrow(1, 4 * c.channels_in,
                                             output_dwt_2.shape[1] - 4 * c.channels_in)  # channels = 24

        # get steg2
        output_steg_2 = iwt(output_steg_dwt_2)  # channels = 3

        #################
        #   backward2:   #
        #################

        output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)  # channels = 12
        output_z_guass_2 = gauss_noise(output_z_dwt_2.shape)  # channels = 24

        output_rev_dwt_2 = torch.cat((output_steg_dwt_2, output_z_guass_2), 1)  # channels = 36

        rev_dwt_2 = net2(output_rev_dwt_2, rev=True)  # channels = 36

        rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        rev_secret_dwt_2 = rev_dwt_2.narrow(1, rev_dwt_2.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12

        rev_steg_1 = iwt(rev_steg_dwt_1)  # channels = 3
        rev_secret_2 = iwt(rev_secret_dwt_2)  # channels = 3


        #################
        #   backward1:   #
        #################
        output_rev_dwt_1 = torch.cat((rev_steg_dwt_1, output_z_guass_1), 1)  # channels = 24

        rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 36

        rev_secret_dwt = rev_dwt_1.narrow(1, rev_dwt_1.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
        rev_secret_1 = iwt(rev_secret_dwt)

        # 还原图像
        secret_rev1_255 = ldr_denorm(tensor_to_numpy(rev_secret_1).squeeze())
        secret_rev2_255 = ldr_denorm(tensor_to_numpy(rev_secret_2).squeeze())
        secret_1_255 = ldr_denorm(tensor_to_numpy(secret_1).squeeze())
        secret_2_255 = ldr_denorm(tensor_to_numpy(secret_2).squeeze())  # (H,W,3)   rgb

        cover_255 = rgbe_denorm(tensor_to_numpy(cover).squeeze())  # (H,W,4)  RGBE
        cover_hdr = rgbe2float(cover_255)

        output_steg_1 = torch.cat((output_steg_1, cover_E), 1)
        steg_1_255 = rgbe_denorm(tensor_to_numpy(output_steg_1).squeeze())
        steg_1_hdr = rgbe2float(steg_1_255)
        output_steg_2 = torch.cat((output_steg_2, cover_E), 1)
        steg_2_255 = rgbe_denorm(tensor_to_numpy(output_steg_2).squeeze())
        steg_2_hdr = rgbe2float(steg_2_255)

        cover_ldr = tonemapReinhard.process(cover_hdr)  # R,G,B顺序
        steg_1_ldr = tonemapReinhard.process(steg_1_hdr)
        steg_2_ldr = tonemapReinhard.process(steg_2_hdr)
        # psnr
        psnr_temp1 = calculate_psnr(secret_1_255, secret_rev1_255, type='ldr')
        psnr_s1.append(psnr_temp1)  # secret和sr图像的PSNR
        psnr_temp2 = calculate_psnr(secret_2_255, secret_rev2_255, type='ldr')
        psnr_s2.append(psnr_temp2)
        psnr_temp_c1_hdr = calculate_psnr(cover_hdr, steg_1_hdr, type='hdr')
        psnr_c1_hdr.append(psnr_temp_c1_hdr)  # cover和stego1图像的PSNR
        psnr_temp_c2_hdr = calculate_psnr(cover_hdr, steg_2_hdr, type='hdr')
        psnr_c2_hdr.append(psnr_temp_c2_hdr)
        # ssim
        ssim_temp1 = calculate_ssim(secret_1_255, secret_rev1_255, type='ldr')
        ssim_s1.append(ssim_temp1)  # secret和sr图像的PSNR
        ssim_temp2 = calculate_ssim(secret_2_255, secret_rev2_255, type='ldr')
        ssim_s2.append(ssim_temp2)
        ssim_temp_c1_hdr = calculate_ssim(cover_hdr, steg_1_hdr, type='hdr')
        ssim_c1_hdr.append(ssim_temp_c1_hdr)  # cover和stego1图像的PSNR
        ssim_temp_c2_hdr = calculate_ssim(cover_hdr, steg_2_hdr, type='hdr')
        ssim_c2_hdr.append(ssim_temp_c2_hdr)
        # cover,steg1,steg2 ldr,hdr
        cv2.imwrite(c.TEST_PATH_cover + '%.5d.png' % i, cover_ldr[..., ::-1] * 255)
        cv2.imwrite(c.TEST_PATH_steg_1 + '%.5d.png' % i, steg_1_ldr[..., ::-1] * 255)
        cv2.imwrite(c.TEST_PATH_steg_2 + '%.5d.png' % i, steg_2_ldr[..., ::-1] * 255)
        # cv2.imwrite(c.TEST_PATH_cover + '%.5d.hdr' % i, cover_hdr[..., ::-1])
        # cv2.imwrite(c.TEST_PATH_steg_1 + '%.5d.hdr' % i, steg_1_hdr[..., ::-1])
        # cv2.imwrite(c.TEST_PATH_steg_2 + '%.5d.hdr' % i, steg_2_hdr[..., ::-1])
        # sec1,sec2,sec_rev1,sec_rev2
        cv2.imwrite(c.TEST_PATH_secret_1 + '%.5d.png' % i, secret_1_255[..., ::-1])
        cv2.imwrite(c.TEST_PATH_secret_2 + '%.5d.png' % i, secret_2_255[..., ::-1])
        cv2.imwrite(c.TEST_PATH_secret_rev_1 + '%.5d.png' % i, secret_rev1_255[..., ::-1])
        cv2.imwrite(c.TEST_PATH_secret_rev_2 + '%.5d.png' % i, secret_rev2_255[..., ::-1])

        # Amap
        # A = ldr_denorm(tensor_to_numpy(Amap).squeeze())
        # cv2.imwrite(c.TEST_PATH_A_map + '%.5d.png' % i, A)
        # Emap
        # E = ldr_denorm(tensor_to_numpy(Emap).squeeze())
        # cv2.imwrite(c.TEST_PATH_E_map + '%.5d.png' % i, E)
        # resi_cover_1, resi_cover_2
        # resi_steg_1 = np.abs(cover_ldr - steg_1_ldr)
        # resi_steg_1 = resi_steg_1 * 7
        # resi_steg_2 = np.abs(cover_ldr - steg_2_ldr)
        # resi_steg_2 = resi_steg_2 * 7

        # cv2.imwrite(c.TEST_PATH_resi_steg_1 + '%.5d.png' % i, resi_steg_1[..., ::-1] * 255)
        # cv2.imwrite(c.TEST_PATH_resi_steg_2 + '%.5d.png' % i, resi_steg_2[..., ::-1] * 255)

        # resi_secret_1, resi_secret_2
        # resi_sec_1 = np.abs(secret_1_255 - secret_rev1_255)
        # resi_sec_1 = resi_sec_1 * 7
        # resi_sec_2 = np.abs(secret_2_255 - secret_rev2_255)
        # resi_sec_2 = resi_sec_2 * 7
        # cv2.imwrite(c.TEST_PATH_resi_sec_1 + '%.5d.png' % i, resi_sec_1[..., ::-1])
        # cv2.imwrite(c.TEST_PATH_resi_sec_2 + '%.5d.png' % i, resi_sec_2[..., ::-1])

    f.write(
            "[test]  "
            + "S1 average psnr: " + str(np.mean(psnr_s1)) + ", "
            + "C1_hdr average psnr: " + str(np.mean(psnr_c1_hdr)) + ", "
            + "S2 average psnr: " + str(np.mean(psnr_s2)) + ", "
            + "C2_hdr average psnr: " + str(np.mean(psnr_c2_hdr)) + "\n"
    )
    f.write(
        "[test]  "
        + "S1 average ssim: " + str(np.mean(ssim_s1)) + ", "
        + "C1_hdr average ssim: " + str(np.mean(ssim_c1_hdr)) + ", "
        + "S2 average ssim: " + str(np.mean(ssim_s2)) + ", "
        + "C2_hdr average ssim: " + str(np.mean(ssim_c2_hdr)) + "\n"
    )
    f.close()

        # imp_map = imp_map.cpu().numpy().squeeze() * 255
        # cover = cover.cpu().numpy().squeeze() * 255
        # output_steg_1 = output_steg_1.cpu().numpy().squeeze() * 255
        # print(imp_map)
        # imp_map = imp_map * 1
        # resi_cover_1 = cover - output_steg_1
        # resi_cover_2 = cover - output_steg_2
        # resi_secret_1 = secret_1 - rev_secret_1
        # resi_secret_2 = secret_2 - rev_secret_2
        # resi_cover_1 = resi_cover_1 * 7
        # resi_cover_2 = resi_cover_2 * 7
        # resi_secret_1 = resi_secret_1 * 7
        # resi_secret_2 = resi_secret_2 * 7

        # torchvision.utils.save_image(imp_map, c.TEST_PATH_imp_map + '%.5d.png' % i)
        # torchvision.utils.save_image(resi_cover_1, c.TEST_PATH_resi_cover_1 + '%.5d.png' % i)
        # torchvision.utils.save_image(resi_cover_2, c.TEST_PATH_resi_cover_2 + '%.5d.png' % i)
        # torchvision.utils.save_image(resi_secret_1, c.TEST_PATH_resi_secret_1 + '%.5d.png' % i)
        # torchvision.utils.save_image(resi_secret_2, c.TEST_PATH_resi_secret_2 + '%.5d.png' % i)

        # torchvision.utils.save_image(cover, c.TEST_PATH_cover + '%.5d.png' % i)
        # torchvision.utils.save_image(secret_1, c.TEST_PATH_secret_1 + '%.5d.png' % i)
        # torchvision.utils.save_image(secret_2, c.TEST_PATH_secret_2 + '%.5d.png' % i)

        # torchvision.utils.save_image(output_steg_1, c.TEST_PATH_steg_1 + '%.5d.png' % i)
        # torchvision.utils.save_image(rev_secret_1, c.TEST_PATH_secret_rev_1 + '%.5d.png' % i)

        # torchvision.utils.save_image(output_steg_2, c.TEST_PATH_steg_2 + '%.5d.png' % i)
        # torchvision.utils.save_image(rev_secret_2, c.TEST_PATH_secret_rev_2 + '%.5d.png' % i)


        # torchvision.utils.save_image(rev_secret_dwt.narrow(1, 0, c.channels_in), '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LL/secret-rev_1/' + '%.5d.png' % i)
        # torchvision.utils.save_image(rev_secret_dwt.narrow(1, c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HL/secret-rev_1/' + '%.5d.png' % i)
        # torchvision.utils.save_image(rev_secret_dwt.narrow(1, 2 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LH/secret-rev_1/' + '%.5d.png' % i)
        # torchvision.utils.save_image(rev_secret_dwt.narrow(1, 3 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HH/secret-rev_1/' + '%.5d.png' % i)
        #
        # torchvision.utils.save_image(output_steg_dwt_2.narrow(1, 0, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LL/steg_2/' + '%.5d.png' % i)
        # torchvision.utils.save_image(output_steg_dwt_2.narrow(1, c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HL/steg_2/' + '%.5d.png' % i)
        # torchvision.utils.save_image(output_steg_dwt_2.narrow(1, 2 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LH/steg_2/' + '%.5d.png' % i)
        # torchvision.utils.save_image(output_steg_dwt_2.narrow(1, 3 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HH/steg_2/' + '%.5d.png' % i)
        #
        # torchvision.utils.save_image(output_steg_dwt_1.narrow(1, 0, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LL/steg_1/' + '%.5d.png' % i)
        # torchvision.utils.save_image(output_steg_dwt_1.narrow(1, c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HL/steg_1/' + '%.5d.png' % i)
        # torchvision.utils.save_image(output_steg_dwt_1.narrow(1, 2 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LH/steg_1/' + '%.5d.png' % i)
        # torchvision.utils.save_image(output_steg_dwt_1.narrow(1, 3 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HH/steg_1/' + '%.5d.png' % i)
        #
        # torchvision.utils.save_image(rev_secret_dwt_2.narrow(1, 0, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LL/secret-rev_2/' + '%.5d.png' % i)
        # torchvision.utils.save_image(rev_secret_dwt_2.narrow(1, c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HL/secret-rev_2/' + '%.5d.png' % i)
        # torchvision.utils.save_image(rev_secret_dwt_2.narrow(1, 2 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LH/secret-rev_2/' + '%.5d.png' % i)
        # torchvision.utils.save_image(rev_secret_dwt_2.narrow(1, 3 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HH/secret-rev_2/' + '%.5d.png' % i)
        #
        # torchvision.utils.save_image(cover_dwt.narrow(1, 0, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LL/cover/' + '%.5d.png' % i)
        # torchvision.utils.save_image(cover_dwt.narrow(1, c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HL/cover/' + '%.5d.png' % i)
        # torchvision.utils.save_image(cover_dwt.narrow(1, 2 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LH/cover/' + '%.5d.png' % i)
        # torchvision.utils.save_image(cover_dwt.narrow(1, 3 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HH/cover/' + '%.5d.png' % i)
        #
        # torchvision.utils.save_image(secret_dwt_1.narrow(1, 0, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LL/secret_1/' + '%.5d.png' % i)
        # torchvision.utils.save_image(secret_dwt_1.narrow(1, c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HL/secret_1/' + '%.5d.png' % i)
        # torchvision.utils.save_image(secret_dwt_1.narrow(1, 2 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LH/secret_1/' + '%.5d.png' % i)
        # torchvision.utils.save_image(secret_dwt_1.narrow(1, 3 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HH/secret_1/' + '%.5d.png' % i)
        #
        # torchvision.utils.save_image(secret_dwt_2.narrow(1, 0, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LL/secret_2/' + '%.5d.png' % i)
        # torchvision.utils.save_image(secret_dwt_2.narrow(1, c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HL/secret_2/' + '%.5d.png' % i)
        # torchvision.utils.save_image(secret_dwt_2.narrow(1, 2 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/LH/secret_2/' + '%.5d.png' % i)
        # torchvision.utils.save_image(secret_dwt_2.narrow(1, 3 * c.channels_in, c.channels_in),
        #                              '/home/jjp/cascaded_Hinet/test-image-attention-div2k-DWT/HH/secret_2/' + '%.5d.png' % i)