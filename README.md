# IMMIH: **High-Capacity Multi-image Hiding in HDR Images using Invertible Neural Networks**

Author: Ruonan Yi

Email address: y1125118639@163.com




## 1. Pre-request
### 1.1 Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)).
- [PyTorch = 1.0.1](https://pytorch.org/) .

### 1.2 Dataset

1. SingleHDR

2. COCO

   https://www.kaggle.com/datasets/hariwh0/ms-coco-dataset/code

3. high-quality hdr datasets

   https://resources.mpi-inf.mpg.de/hdr/gallery.html;
   http://www.anyhere.com/gward/hdrenc/pages/originals.html;
   https://pfstools.sourceforge.net/hdr_gallery.html;
   http://rit-mcsl.org/fairchild//HDRPS/HDRthumbs.html;

4. DIV2K

   https://www.kaggle.com/datasets/joe1995/div2k-dataset

For train or test on your own path, change the code in `config.py`:  

```
#Dataset

PATH = '/home/u2080/RuonanY/train/'
TRAIN_PATH_cover = PATH + 'cover'
TRAIN_PATH_sec1 = PATH + 'sec1'
TRAIN_PATH_sec2 = PATH + 'sec2'
PATH1 = '/home/u2080/RuonanY/'+val_dataset+'/'
VAL_PATH_cover = PATH1 + 'cover'
VAL_PATH_sec1 = PATH1 + 'sec1'
VAL_PATH_sec2 = PATH1 + 'sec2'
```

*In the training phase, 1500 images from SingleHDR dataset are used as HDR cover images, and 3000 images from COCO image dataset are used as LDR secret images, and the image sizes are all randomly cropped to 128 128. In the testing phase, the HDR image datasets with higher image quality is used, and due to the high resolution of HDR images in this dataset, the dataset is expanded by randomly cropping to expand the dataset with 447 HDR images. At the same time, 894 images from the DIV2K dataset were used as secret images, and the test images were randomly cropped to 256 256.*

## 2. Test

1. Here are two test codes, (test-A-E_map.py) with A-E map and (train_M3.py) without A-E map.

2.  In the record subfolder, we provide models for several training scenarios, corresponding to the three scenarios of the ablation experiments in the paper.

   M1: trained model with perceptual loss, Amap and Emap

   M2: trained model with Amap and Emap

   M3: trained model with perceptual loss

3.  Make sure that the paths in the pre-trained model in config.py are the same as the paths you are actually storing.

  ```
  pretrain = True
  PRETRAIN_PATH = './model_save/'
  suffix_pretrain = 'model_checkpoint_00194'
  ```

4. Important images from the test set will be saved and you can set the path to them in config.py.

   ```
   TEST_PATH = './test_results/'   # 保存测试结果  imp_map  A-E_map
   TEST_PATH_cover = TEST_PATH + 'cover/'
   TEST_PATH_secret_1 = TEST_PATH + 'secret_1/'
   TEST_PATH_secret_2 = TEST_PATH + 'secret_2/'
   TEST_PATH_steg_1 = TEST_PATH + 'steg_1/'
   TEST_PATH_steg_2 = TEST_PATH + 'steg_2/'
   TEST_PATH_secret_rev_1 = TEST_PATH + 'secret-rev_1/'
   TEST_PATH_secret_rev_2 = TEST_PATH + 'secret-rev_2/'
   
   TEST_PATH_A_map = TEST_PATH + 'A/'
   TEST_PATH_E_map = TEST_PATH + 'E/'
   
   TEST_PATH_resi_steg_1 = TEST_PATH + 'resi_steg_1/'
   TEST_PATH_resi_steg_2 = TEST_PATH + 'resi_steg_2/'
   TEST_PATH_resi_sec_1 = TEST_PATH + 'resi_sec_1/'
   TEST_PATH_resi_sec_2 = TEST_PATH + 'resi_sec_2/'
   ```

   


## 3. Train

1. After getting the dataset ready, modify the path variables for the training and validation dataset in config.py.
2. Check the optim parameters in `config.py` is correct. Make sure the sub-model(net1, net2...) you want to train is correct.

