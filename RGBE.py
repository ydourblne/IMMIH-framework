# -*- coding = utf-8 -*-
# @Time: 2022/7/17 20:28
# @Author: Ruonan Yi
# @File: RGBE.py
# @software: PyCharm

import cv2
import numpy as np
import torch
import random
from struct import pack

def find_n(x:float)->int:
    '''
    用于RGB格式数据转RGBE数据时，计算v的科学计数表示（m*2^n）
    :param x:
    :return: 第一个为m，第二个为n
    '''
    m=x
    n=0
    while(m<1):
        n=n-1
        m=m*2

    return n+1

#从RGB计算e通道，并且归一化
def get_e_from_float(RGB, normalize='minmax'):
    if RGB.ndim == 3:
        #zeros=np.sum(RGB,axis=2)==0  #不同通道按像素求和
        max_value = np.max(RGB, axis=2)  #不同通道按像素求最大
        zeros = max_value==0
        max_value[zeros] = 1
        #e = np.floor(np.log2(max_value)) + 129
        e = np.floor(np.log2(max_value) + 128)
        e[zeros]=0
        '''
        if normalize == 'minmax':
            e_min = np.min(e)
            e_max = np.max(e)
            e = (e - e_min) / (e_max - e_min)
        elif normalize == 'log':
            e_max = np.max(e)
            e = np.log(e + 1) / np.log(e_max + 1)
        else:
            raise NotImplementedError
        e = np.expand_dims(e, axis=2)
        '''
        return e
    else:
        raise NotImplementedError


def rgbe2float(rgbe: np.ndarray) -> np.ndarray:
    res = np.zeros((rgbe.shape[0], rgbe.shape[1], 3))
    p = rgbe[:, :, 3] > 0 #e通道
    m = 2.0 ** (rgbe[:, :, 3][p] - 136.0)
    res[:, :, 0][p] = rgbe[:, :, 0][p] * m #计算R
    res[:, :, 1][p] = rgbe[:, :, 1][p] * m
    res[:, :, 2][p] = rgbe[:, :, 2][p] * m
    return np.array(res, dtype=np.float32)


def float2rgbe(RGB: np.ndarray) -> np.ndarray:  #多维数组类型
    '''
    从RGB浮点数转换为rgbe表示
    :param RGB: RGB浮点数组，范围应当已经被规范到(0,1)
    :return:
    '''
    rgbe = np.zeros([RGB.shape[0], RGB.shape[1], 4], dtype=float) #创建一个三维的零数组，前面是大小，dtype是数据类型
    p = np.max(RGB, axis=2)   #三维数组共有三个轴,对每个通道每一行求最大，一个通道的放一行，得到：4行，256列
    find_n_v = np.vectorize(find_n)
    p = find_n_v(p)
    p = np.expand_dims(p, 2)
    p = np.array(p, dtype=float)
    rgbe[:, :, :3] = RGB * 256 / (2 ** p)
    rgbe[:, :, 3:4] = p + 128
    # for i in range(RGB.shape[0]):
    #     for j in range(RGB.shape[1]):
    #         _,n=find_mn(np.max(RGB[i,j,:]))
    #         rgbe[i,j,:3]=RGB[i,j,:]*256/(2**n)
    #         rgbe[i,j,3]=128+n

    return rgbe


'''获取HDR图像的四通道表示'''  #!!直接读取hdr图像，转换为rgbe
def readHdr(fileName: str) -> np.ndarray:
    fileinfo = {}
    with open(fileName, 'rb') as fd:
        tline = fd.readline().strip()
        if len(tline) < 3 or tline[:2] != b'#?':
            print('invalid header')
            return
        fileinfo['identifier'] = tline[2:]

        # while(tline[:1]==b'#'):
        tline = fd.readline().strip()

        if (tline[:1] == b'#'):
            tline = fd.readline().strip()
        while tline:
            n = tline.find(b'=')
            if n > 0:
                fileinfo[tline[:n].strip()] = tline[n + 1:].strip()
            tline = fd.readline().strip()

        tline = fd.readline().strip().split(b' ')
        fileinfo['Ysign'] = tline[0][0]
        fileinfo['height'] = int(tline[1])
        fileinfo['Xsign'] = tline[2][0]
        fileinfo['width'] = int(tline[3])

        data = [d for d in fd.read()]
        height, width = fileinfo['height'], fileinfo['width']
        if width < 8 or width > 32767:
            data.resize((height, width, 4))
            print("error")
            return rgbe2float(data)

        img = np.zeros((height, width, 4))
        dp = 0
        # c=0
        for h in range(height):
            if data[dp] != 2 or data[dp + 1] != 2:
                print('this file is not run length encoded')
                print(data[dp:dp + 4])
                return
            if data[dp + 2] * 256 + data[dp + 3] != width:
                print('wrong scanline width')
                return
            dp += 4
            for i in range(4):
                ptr = 0
                while (ptr < width):
                    if data[dp] > 128:
                        count = data[dp] - 128
                        if count == 0 or count > width - ptr:
                            print('bad scanline data')
                        img[h, ptr:ptr + count, i] = data[dp + 1]
                        ptr += count
                        dp += 2
                    else:
                        # if(data[dp]==127):
                        # c=c+1
                        count = data[dp]
                        dp += 1
                        if count == 0 or count > width - ptr:
                            print('bad scanline data')
                        img[h, ptr:ptr + count, i] = data[dp: dp + count]
                        ptr += count
                        dp += count
        # return rgbe2float(img)
        # return img,c
        return img


def saveHdr(filename: str, rgbe: np.ndarray) -> bool:
    '''
    直接将rgbe格式的数据保存为"*.hdr"文件
    这样保存会导致文件大小和opencv等标准库保存的大小不同（即便数据完全不变）,这个问题暂未解决
    :param filename:
    :param rgbe:
    :return:
    '''
    if (rgbe.shape[1] < 8 or rgbe.shape[1] > 32767):
        print("The width of the hdr image must be in range(8,32767)")
        return False

    rgbe = rgbe.astype(int)

    with open(filename, 'wb') as fw:
        fw.write(b'#?RGBE')
        fw.write(b'\n')
        fw.write(b'FORMAT=32-bit_rle_rgbe')
        fw.write(b'\n')
        fw.write(b'\n')

        fw.write(b'-Y ')
        fw.write(bytes(str(rgbe.shape[0]), 'ansi'))
        fw.write(b' +X ')
        fw.write(bytes(str(rgbe.shape[1]), 'ansi'))
        fw.write(b'\n')

        for j in range(rgbe.shape[0]):
            fw.write(pack('B', 2))
            fw.write(pack('B', 2))
            fw.write(pack('B', int(rgbe.shape[1] / 256)))
            fw.write(pack('B', int(rgbe.shape[1] % 256)))

            for i in range(4):
                value = rgbe[j, 0, i]
                same_length = 1
                dif_list = []
                dif_list.append(rgbe[j, 0, i])
                for k in range(1, rgbe.shape[1]):
                    if (rgbe[j, k, i] == value):
                        if (len(dif_list) > 1):
                            dif_list.pop(-1)
                            fw.write(pack('B', len(dif_list)))
                            for _, d in enumerate(dif_list):
                                fw.write(pack('B', d))
                            dif_list.clear()
                            dif_list.append(value)

                        if (same_length < 127):
                            same_length = same_length + 1
                        else:
                            fw.write(pack('B', 255))
                            fw.write(pack('B', value))
                            same_length = 1
                    elif (rgbe[j, k, i] != value and same_length == 1):
                        value = rgbe[j, k, i]
                        if (len(dif_list) < 127):
                            dif_list.append(rgbe[j, k, i])
                        else:
                            fw.write(pack('B', 127))
                            for _, d in enumerate(dif_list):
                                fw.write(pack('B', d))
                            dif_list.clear()
                            dif_list.append(value)
                    elif (rgbe[j, k, i] != value and same_length > 1):
                        fw.write(pack('B', 128 + same_length))
                        fw.write(pack('B', value))
                        value = rgbe[j, k, i]
                        same_length = 1
                        dif_list = [value]

                if (len(dif_list) > 1):
                    fw.write(pack('B', len(dif_list)))
                    for _, d in enumerate(dif_list):
                        fw.write(pack('B', d))
                elif (same_length > 1):
                    fw.write(pack('B', 128 + same_length))
                    fw.write(pack('B', value))
                else:
                    fw.write(pack('B', 1))
                    fw.write(pack('B', value))
    fw.close()
    return True

def getEdge(e):
    # plt.imshow(e)  # 显示e
    # pylab.show()
    edge = torch.zeros(e.shape)
    for k in range(e.shape[0]):
        E = e[k, 0, :, :]
        for i in range(1, E.shape[0]-1):
            for j in range(1, E.shape[1]-1):
                if E[i, j] == E[i, j-1] == E[i-1, j] == E[i, j+1] == E[i+1, j]:
                    edge[k, 0, i, j] = 1  # flat
                else:
                    edge[k, 0, i, j] = 0  # boundary
    #plt.imshow(edge, cmap="gray")
    #plt.savefig('./edge.jpg')
    #plt.show()
    return edge

if __name__ == '__main__':
    test=np.random.randn(2,3,4)
    print(test)
    test=test.reshape([-1,4])
    print(test)

    img=readHdr('./test1.hdr')
    saveHdr('./test.txt',img)
    img2=readHdr('test1.hdr')
    t=img-img2
    t=t.reshape([-1])
    for _,d in enumerate(t):
        if(d!=0):
            print("error")

    rgbe=readHdr('./test1.hdr')
    rgb=rgbe2float(rgbe)
    float=float2rgbe(rgb)
    print(rgbe-float)