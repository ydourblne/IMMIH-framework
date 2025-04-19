#!/usr/bin/env python
import sys
import os
import torch
import torch.nn
import torch.optim
import torchvision
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
import json
from model import *
from imp_subnet import *
import torchvision.transforms as T
import config as c
from tensorboardX import SummaryWriter
from datasets import *
import viz
import modules.module_util as mutil
import modules.Unet_common as common
import warnings
import cv2
from vgg_loss import *
from vgg_loss import VGGLoss
from RGBE import *
from PSNR import *
from utils import imload, imshow, imsave, array_to_cam, blend
from cam_model import CAM


warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)  # 返回一个batch中所有样本损失的和，结果为标量
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def imp_loss(output, resi):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, resi)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


def distr_loss(noise):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(noise, torch.zeros(noise.shape).cuda())
    return loss.to(device)


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def init_net3(mod):
    for key, param in mod.named_parameters():
        if param.requires_grad:
            param.data = 0.1 * torch.randn(param.data.shape).cuda()


#####################
# Model initialize: #
#####################
net1 = Model_3()  # 第一个IHNN
net2 = Model_3()    # 第二个IHNN

net1.cuda()
net2.cuda()

init_model(net1)    # 初始化三个网络
init_model(net2)

net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids) # 分布式训练
net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)

para1 = get_parameter_number(net1)  # 计算模型的参数数量
para2 = get_parameter_number(net2)

print(para1)
print(para2)

params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))
params_trainable2 = (list(filter(lambda p: p.requires_grad, net2.parameters())))

optim1 = torch.optim.Adam(params_trainable1, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
optim2 = torch.optim.Adam(params_trainable2, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)   # 等间隔调整学习率，调整间隔为weight_step，调整倍数为gamma
weight_scheduler2 = torch.optim.lr_scheduler.StepLR(optim2, c.weight_step, gamma=c.gamma)


dwt = common.DWT()
iwt = common.IWT()


network_Amap = CAM('resnet50').to(device)
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
if c.train_next:     # 载入训练好的模型（注意修改路径）
    load(c.MODEL_PATH + c.suffix_load + '_1.pt', net1, optim1)
    load(c.MODEL_PATH + c.suffix_load + '_2.pt', net2, optim2)

if c.pretrain:  # 预训练
    load(c.PRETRAIN_PATH + c.suffix_pretrain + '_1.pt', net1, optim1)
    load(c.PRETRAIN_PATH + c.suffix_pretrain + '_2.pt', net2, optim2)


try:
    # writer = SummaryWriter(comment='hinet', filename_suffix="steg")
    f = open('./results.txt', 'a+')

    import torch,gc
    gc.collect()
    torch.cuda.empty_cache()

    for i_epoch in range(c.epochs): # 迭代次数
        i_epoch = i_epoch + c.trained_epoch + 1
        f.write("epoch: " + str(i_epoch) + "\n")
        #print("epoch" , i_epoch)
        loss_history = []
        loss_history_g1 = []
        loss_history_g2 = []
        loss_history_r1 = []
        loss_history_r2 = []
        #################
        #     train:    #
        #################

        vgg = VGGLoss(3, 1, False)
        vgg.to(device)
        pbar = tqdm(trainloader)
        pbar.set_description(f'Epoch {i_epoch}')
        for i_batch, (cover, secret_1, secret_2) in enumerate(pbar): # 一个batch的样本一起计算loss，然后更新参数
            # data preparation
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
            #print("batch",i_batch+1)
            cover = cover.to(device)    # channels = 4
            cover_RGB = cover[:, :3]
            cover_E = cover[:, 3:4]     # [n,1,H,W]
            network_Amap.eval()
            with torch.no_grad():
                topk = 1
                prob, cls, cam = network_Amap(cover_RGB[0, :, :, :].unsqueeze(0), topk=topk)
                Amap = cam[0].unsqueeze(0)
                for j in range(1, c.batchsize_train):
                    prob, cls, cam = network_Amap(cover_RGB[j, :, :, :].unsqueeze(0), topk=topk)
                    cam_ = cam[0].unsqueeze(0)
                    Amap = torch.cat((Amap, cam_), 0).to(device)
            # 求Emap
            Emap = getEdge(cover_E).to(device)
            secret_1 = secret_1.to(device)
            secret_2 = secret_2.to(device)

            cover_dwt = dwt(cover_RGB)  # channels = 12
            cover_dwt_low = cover_dwt.narrow(1, 0, c.channels_in)  # channels = 3，取出低频部分LL,在通道上取[0, c.channels_in]
            secret_dwt_1 = dwt(secret_1)    # channels = 12
            secret_dwt_2 = dwt(secret_2)
            Amap_dwt = dwt(Amap)        # channels = 4
            Emap_dwt = dwt(Emap)       # channels = 4
            map_dwt = torch.cat((Amap_dwt, Emap_dwt), 1)      # channels = 8
            input_dwt_1 = torch.cat((cover_dwt, map_dwt), 1)  # channels = 20
            input_dwt_1 = torch.cat((input_dwt_1, secret_dwt_1), 1)  # channels = 32

            #################
            #    forward1:   # 模型中无IM部分
            #################
            output_dwt_1 = net1(input_dwt_1)  # channels = 32
            output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12，前12个通道为stego
            output_steg_dwt_low_1 = output_steg_dwt_1.narrow(1, 0, c.channels_in)  # channels = 3
            output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in,output_dwt_1.shape[1] - 4 * c.channels_in)  # channels = 20

            # get steg1
            output_steg_1 = iwt(output_steg_dwt_1)  # channels = 3，转换到图像域的stego_1

            #################
            #    forward2:   #
            ################
            input_dwt_2 = torch.cat((output_steg_dwt_1, map_dwt), 1)  # 20
            input_dwt_2 = torch.cat((input_dwt_2, secret_dwt_2), 1)  # 32  第二个invertible hiding module的输入

            output_dwt_2 = net2(input_dwt_2)  # channels = 32
            output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
            output_steg_dwt_low_2 = output_steg_dwt_2.narrow(1, 0, c.channels_in)  # channels = 3
            output_z_dwt_2 = output_dwt_2.narrow(1, 4 * c.channels_in, output_dwt_2.shape[1] - 4 * c.channels_in)  # channels = 20

            # get steg2
            output_steg_2 = iwt(output_steg_dwt_2)  # channels = 3

            #################
            #   backward2:   ########
            #################

            output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)  # channels = 20
            output_z_guass_2 = gauss_noise(output_z_dwt_2.shape)  # channels = 20

            output_rev_dwt_2 = torch.cat((output_steg_dwt_2, output_z_guass_2), 1)  # channels = 32

            rev_dwt_2 = net2(output_rev_dwt_2, rev=True)  # channels = 32

            rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
            rev_secret_dwt_2 = rev_dwt_2.narrow(1, rev_dwt_2.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12

            rev_steg_1 = iwt(rev_steg_dwt_1)  # channels = 3
            rev_secret_2 = iwt(rev_secret_dwt_2)  # channels = 3

            #################
            #   backward1:   #
            #################
            output_rev_dwt_1 = torch.cat((rev_steg_dwt_1, output_z_guass_1), 1)  # channels = 32
            import torch, gc

            gc.collect()
            torch.cuda.empty_cache()
            rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 32

            rev_secret_dwt = rev_dwt_1.narrow(1, rev_dwt_1.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
            rev_secret_1 = iwt(rev_secret_dwt)

            #################
            #     loss:     #
            #################
            ## concealing loss
            g_loss_1 = guide_loss(output_steg_1.cuda(), cover_RGB.cuda())   # concealing loss of IHNN1
            g_loss_2 = guide_loss(output_steg_2.cuda(), cover_RGB.cuda())

            vgg_on_cov = vgg(cover_RGB)
            vgg_on_steg1 = vgg(output_steg_1)
            vgg_on_steg2 = vgg(output_steg_2)

            ## perceptual loss
            perc_loss = guide_loss(vgg_on_cov, vgg_on_steg1) + guide_loss(vgg_on_cov, vgg_on_steg2)

            ## low-frequency wavelet loss
            l_loss_1 = guide_loss(output_steg_dwt_low_1.cuda(), cover_dwt_low.cuda())
            l_loss_2 = guide_loss(output_steg_dwt_low_2.cuda(), cover_dwt_low.cuda())

            ## revealing loss
            r_loss_1 = reconstruction_loss(rev_secret_1, secret_1)
            r_loss_2 = reconstruction_loss(rev_secret_2, secret_2)

            ## total loss
            total_loss = c.lamda_reconstruction_1 * r_loss_1 + c.lamda_reconstruction_2 * r_loss_2 + c.lamda_guide_1 * g_loss_1 + c.lamda_guide_2 * g_loss_2 + c.lamda_low_frequency_1 * l_loss_1 + c.lamda_low_frequency_2 * l_loss_2
            total_loss = total_loss + 0.05 * perc_loss  # 0.05还是0.01
            ## 参数更新
            total_loss.backward()

            if c.optim_step_1:
                optim1.step()

            if c.optim_step_2:
                optim2.step()



            optim1.zero_grad()
            optim2.zero_grad()

            loss_history.append([total_loss.item(), 0.])
            loss_history_g1.append(g_loss_1.item())
            loss_history_g2.append(g_loss_2.item())
            loss_history_r1.append(r_loss_1.item())
            loss_history_r2.append(r_loss_2.item())


        #################
        #     val:    #
        #################
        if i_epoch % c.val_freq == 0:  # 修改？？
            with torch.no_grad():
                psnr_s1 = []
                psnr_s2 = []
                psnr_c1_hdr = []
                psnr_c2_hdr = []

                net1.eval()
                net2.eval()
                for cover, secret_1, secret_2 in testloader:

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
                    output_dwt_1 = net1(input_dwt_1)
                    output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                    output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, output_dwt_1.shape[1] - 4 * c.channels_in)  # channels = 20

                    # get steg1
                    output_steg_1 = iwt(output_steg_dwt_1)  # channels = 3



                    #################
                    #    forward2:   #
                    #################
                    input_dwt_2 = torch.cat((output_steg_dwt_1, map_dwt), 1)  # 20
                    input_dwt_2 = torch.cat((input_dwt_2, secret_dwt_2), 1)


                    output_dwt_2 = net2(input_dwt_2)
                    output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                    output_z_dwt_2 = output_dwt_2.narrow(1, 4 * c.channels_in, output_dwt_2.shape[1] - 4 * c.channels_in)  # channels = 20

                    # get steg2
                    output_steg_2 = iwt(output_steg_dwt_2)  # channels = 3

                    #################
                    #   backward2:   #
                    #################

                    output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)
                    output_z_guass_2 = gauss_noise(output_z_dwt_2.shape)

                    output_rev_dwt_2 = torch.cat((output_steg_dwt_2, output_z_guass_2), 1)

                    rev_dwt_2 = net2(output_rev_dwt_2, rev=True)

                    rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                    rev_secret_dwt_2 = rev_dwt_2.narrow(1, rev_dwt_2.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12

                    rev_steg_1 = iwt(rev_steg_dwt_1)  # channels = 3
                    rev_secret_2 = iwt(rev_secret_dwt_2)  # channels = 3

                    #################
                    #   backward1:   #
                    #################
                    output_rev_dwt_1 = torch.cat((rev_steg_dwt_1, output_z_guass_1), 1)  # channels = 24

                    rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 24

                    rev_secret_dwt = rev_dwt_1.narrow(1, rev_dwt_1.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
                    rev_secret_1 = iwt(rev_secret_dwt)

                    ## 计算性能 PSNR
                    secret_rev1_255 = ldr_denorm(tensor_to_numpy(rev_secret_1).squeeze())
                    secret_rev2_255 = ldr_denorm(tensor_to_numpy(rev_secret_2).squeeze())
                    secret_1_255 = ldr_denorm(tensor_to_numpy(secret_1).squeeze())
                    secret_2_255 = ldr_denorm(tensor_to_numpy(secret_2).squeeze())  # (H,W,3)   rgb

                    cover_255 = rgbe_denorm(tensor_to_numpy(cover).squeeze())   # (H,W,4)  RGBE
                    cover_hdr = rgbe2float(cover_255)       # (H,W,3)

                    output_steg_1 = torch.cat((output_steg_1, cover_E), 1)
                    steg_1_255 = rgbe_denorm(tensor_to_numpy(output_steg_1).squeeze())
                    steg_1_hdr = rgbe2float(steg_1_255)
                    output_steg_2 = torch.cat((output_steg_2, cover_E), 1)
                    steg_2_255 = rgbe_denorm(tensor_to_numpy(output_steg_2).squeeze())
                    steg_2_hdr = rgbe2float(steg_2_255)

                    psnr_temp1 = calculate_psnr(secret_1_255, secret_rev1_255, type='ldr')
                    psnr_s1.append(psnr_temp1)      # secret和sr图像的PSNR
                    psnr_temp2 = calculate_psnr(secret_2_255, secret_rev2_255, type='ldr')
                    psnr_s2.append(psnr_temp2)

                    # 修改
                    psnr_temp_c1_hdr = calculate_psnr(cover_hdr, steg_1_hdr, type='hdr')
                    psnr_c1_hdr.append(psnr_temp_c1_hdr)       # cover和stego1图像的PSNR
                    psnr_temp_c2_hdr = calculate_psnr(cover_hdr, steg_2_hdr, type='hdr')
                    psnr_c2_hdr.append(psnr_temp_c2_hdr)

                    # if i_epoch % (c.epochs/2) == 0:
                    #     cover_ldr = tonemapReinhard.process(cover_hdr)  # R,G,B顺序
                    #     steg_1_ldr = tonemapReinhard.process(steg_1_hdr)
                    #     steg_2_ldr = tonemapReinhard.process(steg_2_hdr)
                    #     psnr_temp_c1_ldr = calculate_psnr(cover_ldr*255, steg_1_ldr*255, type='ldr')
                    #     psnr_c1_ldr.append(psnr_temp_c1_ldr)  # cover和stego1图像的PSNR
                    #     psnr_temp_c2_ldr = calculate_psnr(cover_ldr*255, steg_2_ldr*255, type='ldr')
                    #     psnr_c2_ldr.append(psnr_temp_c2_ldr)



                        # cv2.imwrite(c.TEST_PATH_cover + '%.5d.png' % (i_epoch), cover_ldr[...,::-1] * 255)
                        # cv2.imwrite(c.TEST_PATH_steg_1 + '%.5d.png' % (i_epoch), steg_1_ldr[..., ::-1] * 255)
                        # cv2.imwrite(c.TEST_PATH_steg_2 + '%.5d.png' % (i_epoch), steg_2_ldr[..., ::-1] * 255)
                        # cv2.imwrite(c.TEST_PATH_cover + '%.5d.hdr' % (i_epoch), cover_hdr[...,::-1])
                        # cv2.imwrite(c.TEST_PATH_steg_1 + '%.5d.hdr' % (i_epoch), steg_1_hdr[...,::-1])
                        # cv2.imwrite(c.TEST_PATH_steg_2 + '%.5d.hdr' % (i_epoch), steg_2_hdr[...,::-1])
                        #
                        # # sec1,sec2,
                        # cv2.imwrite(c.TEST_PATH_secret_1 + '%.5d.png' % i_epoch, secret_1_255[..., ::-1])
                        # cv2.imwrite(c.TEST_PATH_secret_2 + '%.5d.png' % i_epoch, secret_2_255[..., ::-1])
                        # cv2.imwrite(c.TEST_PATH_secret_rev_1 + '%.5d.png' % i_epoch, secret_rev1_255[..., ::-1])
                        # cv2.imwrite(c.TEST_PATH_secret_rev_2 + '%.5d.png' % i_epoch, secret_rev2_255[..., ::-1])



                # writer.add_scalars("PSNR", {"S1 average psnr": np.mean(psnr_s1)}, i_epoch)
                # writer.add_scalars("PSNR", {"C1 average psnr": np.mean(psnr_c1_hdr)}, i_epoch)
                # writer.add_scalars("PSNR", {"S2 average psnr": np.mean(psnr_s2)}, i_epoch)
                # writer.add_scalars("PSNR", {"C2 average psnr": np.mean(psnr_c2_hdr)}, i_epoch)
                f.write(
                    "[test]  "
                    + "S1 average psnr: " + str(np.mean(psnr_s1)) + ", "
                    + "C1_hdr average psnr: " + str(np.mean(psnr_c1_hdr)) + ", "
                    + "S2 average psnr: " + str(np.mean(psnr_s2)) + ", "
                    + "C2_hdr average psnr: " + str(np.mean(psnr_c2_hdr)) + "\n"

                )
                print("[test]  S1 average psnr: %.2f, C1_hdr average psnr: %.2f, S2 average psnr: %.2f, C2_hdr average psnr:  %.2f"%(np.mean(psnr_s1),np.mean(psnr_c1_hdr),np.mean(psnr_s2),np.mean(psnr_c2_hdr)))
                # if i_epoch % (c.epochs / 2) == 0:
                #     f.write(
                #         "[test]  "
                #         + "C1_ldr average psnr: " + str(np.mean(psnr_c1_ldr)) + ", "
                #         + "C2_ldr average psnr: " + str(np.mean(psnr_c2_ldr)) + "\n"
                #
                #     )
                #     print(
                #         "[test]  C1_ldr average psnr: %.2f, C2_ldr average psnr:  %.2f" % (
                #             np.mean(psnr_c1_ldr), np.mean(psnr_c2_ldr)))

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        epoch_losses[1] = np.log10(optim1.param_groups[0]['lr'])

        epoch_losses_g1 = np.mean(np.array(loss_history_g1))
        epoch_losses_g2 = np.mean(np.array(loss_history_g2))
        epoch_losses_r1 = np.mean(np.array(loss_history_r1))
        epoch_losses_r2 = np.mean(np.array(loss_history_r2))

        viz.show_loss(epoch_losses)
        # writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)
        # writer.add_scalars("Train", {"g1_Loss": epoch_losses_g1}, i_epoch)
        # writer.add_scalars("Train", {"g2_Loss": epoch_losses_g2}, i_epoch)
        # writer.add_scalars("Train", {"r1_Loss": epoch_losses_r1}, i_epoch)
        # writer.add_scalars("Train", {"r2_Loss": epoch_losses_r2}, i_epoch)

        f.write(
                "[train]  "
                + "lr: " + str(epoch_losses[1]) + ", "
                + "Train_Loss: " + str(epoch_losses[0]) + ", "
                + "concealing_loss_1: " + str(epoch_losses_g1) + ", "
                + "concealing_loss_2: " + str(epoch_losses_g2) + ", "
                + "revealing_loss_1: " + str(epoch_losses_r1) + ", "
                + "revealing_loss_2: " + str(epoch_losses_r2) + "\n "
                )
        print("[train]  lr: %.4f, Train_Loss: %.4f, concealing_loss_1: %.4f, concealing_loss_2: %.4f, revealing_loss_1: %.4f, revealing_loss_2: %.4f" % (
        epoch_losses[1], epoch_losses[0], epoch_losses_g1, epoch_losses_g2, epoch_losses_r1, epoch_losses_r2))
        # 保存模型 checkpoint
        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim1.state_dict(),
                        'net': net1.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i_1' % i_epoch + '.pt')
            torch.save({'opt': optim2.state_dict(),
                        'net': net2.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i_2' % i_epoch + '.pt')

        weight_scheduler1.step()
        weight_scheduler2.step()

        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
    # 保存最终的模型 state为model.state_dict()则仅保存训练模型的参数,为以后恢复模型提供最大的灵活性
    torch.save({'opt': optim1.state_dict(),
                'net': net1.state_dict()}, c.MODEL_PATH + 'model_1' + '.pt')
    torch.save({'opt': optim2.state_dict(),
                'net': net2.state_dict()}, c.MODEL_PATH + 'model_2' + '.pt')

    # writer.close()
    f.close()

except: # 出现错误的处理
    if c.checkpoint_on_error:
        torch.save({'opt': optim1.state_dict(),
                    'net': net1.state_dict()}, c.MODEL_PATH + 'model_ABORT_1' + '.pt')
        torch.save({'opt': optim2.state_dict(),
                    'net': net2.state_dict()}, c.MODEL_PATH + 'model_ABORT_2' + '.pt')

    raise

finally:
    viz.signal_stop()
