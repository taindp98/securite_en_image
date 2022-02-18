#!/usr/bin/env python
# coding: utf-8

# 
# # Sécurité dans les réseaux et protection de contenus audio-visuels
# ## Membre
# 
# | Prénom   |      NOM      |  Nombre etudiant |
# |----------|:-------------:|------:|
# | Duong Phuc Tai |  NGUYEN | 12108339 |
# | Kamilia |    RAHIL   |   12109923 |
# 
# ## TP2 – 2D IMAGE FILTERING
# 1. DCT – Discrete Cosine Transform
# 2. Low-pass filtering
# 3. High-pass filtering
# 4. Wavelet transform
# 5. Conclusion

# ### Importing necessary libraries

# In[1]:


import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

# library wavelet
import pywt


# In[2]:


## set the necessary path

_DATA = './data'
_RESULT = './result'


# ![image.png](attachment:c99950e5-7c04-4065-8c70-94580ff6ac90.png)

# In[3]:


def read_image(img_name, data_fold = _DATA):
    """
    Read the Lena image and represent it as a matrix
    Make the required format conversion
    param:
        img_name: is the image's name file
        data_fold: is the path of data folder which store the images
    return:
        img_arr: is the matrix or array represent the image
    """
    ## read image
    img_bgr = cv2.imread(os.path.join(data_fold, img_name))
    ## convert color space to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_arr = img_gray.copy()
    ## indicate the type of image array
    return img_arr.astype(float)


# In[4]:


lena_arr = read_image('lena.jpg')
baboon_arr = read_image('baboon.jpg')


# In[5]:


def show_grid(list_imgs, titles):
    """
    utils function to display the images as a grid
    param:
        list_imgs: list of images
        titles: list of titles
    return:
    
    """
    fig=plt.figure(figsize=(10, 5))
    columns = len(list_imgs)
    rows = 1
    for i in range(0, columns*rows):
        img = list_imgs[i]
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(titles[i])
#     plt.colorbar()


# ### Calculate DCT 2D

# In[6]:


## DCT transform using OpenCV
lena_dct = cv2.dct(lena_arr)
print(np.min(lena_dct), np.max(lena_dct))
plt.figure(figsize=(10, 10), dpi=80)
plt.imshow(lena_dct[:50,:50], cmap='gray')
plt.colorbar()


# In[7]:


## DCT transform using OpenCV
baboon_dct = cv2.dct(baboon_arr)
print(np.min(baboon_dct), np.max(baboon_dct))
plt.figure(figsize=(10, 10), dpi=80)
plt.imshow(baboon_dct[:50,:50], cmap='gray')
plt.colorbar()


# In[8]:


rand_arr = np.random.randn(512,512)
rand_dct = cv2.dct(rand_arr)
plt.figure(figsize=(10, 10), dpi=80)
plt.imshow(rand_dct, cmap='gray')
plt.colorbar()


# ![image.png](attachment:6320232c-e5b2-4445-94c8-2b9493d3ae51.png)

# In[9]:


## IDCT transform using OpenCV
lena_idct = cv2.idct(lena_dct)
print(np.min(lena_idct), np.max(lena_idct))
plt.imshow(lena_idct, cmap='gray')


# In[10]:


## compare original image and inverse DCT image
print(f'Max difference: {np.max(lena_arr - lena_idct)}')


# ![image.png](attachment:cfb1f70e-4d71-4178-b98c-ed5f9b485ebe.png)

# In[11]:


def low_pass_filter(dct_coef, fc):
    """
    get coefficients in the low-freq area
    param:
        dct_coef: is the coefficients of image after applying DCT transform
        fc: is the cut frequency
    return:
        sign_coef: is the significant coefficients under the cut frequency
    """
    sign_coef = np.zeros(dct_coef.shape)
    sign_coef[:fc, :fc] = dct_coef[:fc, :fc]
    return sign_coef


# ![image.png](attachment:ab4ebfb4-445b-4d73-85a5-a87a0eaead7c.png)

# In[12]:


## survey with many different frequencies
list_exp = np.arange(1, 5, 1)
list_fc = np.power(2, list_exp)
list_coef_lpf = []
list_idct = []
for fc in list_fc:
    lena_dct_lp = low_pass_filter(lena_dct, fc)
    list_coef_lpf.append(lena_dct_lp)
    list_idct.append(cv2.idct(lena_dct_lp))
show_grid(np.array(list_coef_lpf)[:,:20,:20], [f'fc={item}' for item in list(list_fc)])
show_grid(np.array(list_idct), [f'fc={item}' for item in list(list_fc)])


# In[13]:


## survey with many different frequencies
list_exp = np.arange(5, 9, 1)
list_fc = np.power(2, list_exp)
list_coef_lpf = []
list_idct = []
for fc in list_fc:
    lena_dct_lp = low_pass_filter(lena_dct, fc)
    list_coef_lpf.append(lena_dct_lp)
    list_idct.append(cv2.idct(lena_dct_lp))
show_grid(np.array(list_coef_lpf)[:,:20,:20], [f'fc={item}' for item in list(list_fc)])
show_grid(np.array(list_idct), [f'fc={item}' for item in list(list_fc)])


# ![image.png](attachment:b0f2ad17-89b1-4149-a155-9cc919e44ddf.png)

# In[14]:


## survey with many different frequencies
list_exp = np.arange(1, 5, 1)
list_fc = np.power(2, list_exp)
list_coef_lpf = []
list_idct = []
for fc in list_fc:
    baboon_dct_lp = low_pass_filter(baboon_dct, fc)
    list_coef_lpf.append(baboon_dct_lp)
    list_idct.append(cv2.idct(baboon_dct_lp))
show_grid(np.array(list_coef_lpf)[:,:20,:20], [f'fc={item}' for item in list(list_fc)])
show_grid(np.array(list_idct), [f'fc={item}' for item in list(list_fc)])


# In[15]:


## survey with many different frequencies
list_exp = np.arange(5, 9, 1)
list_fc = np.power(2, list_exp)
list_coef_lpf = []
list_idct = []
for fc in list_fc:
    baboon_dct_lp = low_pass_filter(baboon_dct, fc)
    list_coef_lpf.append(baboon_dct_lp)
    list_idct.append(cv2.idct(baboon_dct_lp))
show_grid(np.array(list_coef_lpf)[:,:20,:20], [f'fc={item}' for item in list(list_fc)])
show_grid(np.array(list_idct), [f'fc={item}' for item in list(list_fc)])


# ![image.png](attachment:631f818a-af01-4b56-9f61-52d7bff07142.png)

# In[16]:


def high_pass_filter(dct_coef, fc):
    """
    get coefficients in the high-freq area
    param:
        dct_coef: is the coefficients of image after applying DCT transform
        fc: is the cut frequency
    return:
        less_sign_coef: is the less significant coefficients over the cut frequency
    """
    less_sign_coef = dct_coef.copy()
    less_sign_coef[:fc, :fc] = 0
    return less_sign_coef


# ![image.png](attachment:581b44e7-6b6e-4169-8c55-97c8182f443c.png)

# In[17]:


## survey with many different frequencies
list_exp = np.arange(1, 5, 1)
list_fc = np.power(2, list_exp)
list_coef_hpf = []
list_idct = []
for fc in list_fc:
    lena_dct_hp = high_pass_filter(lena_dct, fc)
    list_coef_hpf.append(lena_dct_hp)
    list_idct.append(cv2.idct(lena_dct_hp))
show_grid(np.array(list_coef_hpf)[:,:20,:20], [f'fc={item}' for item in list(list_fc)])
show_grid(np.array(list_idct), [f'fc={item}' for item in list(list_fc)])


# In[18]:


## survey with many different frequencies
list_exp = np.arange(5, 9, 1)
list_fc = np.power(2, list_exp)
list_coef_hpf = []
list_idct = []
for fc in list_fc:
    lena_dct_hp = high_pass_filter(lena_dct, fc)
    list_coef_hpf.append(lena_dct_hp)
    list_idct.append(cv2.idct(lena_dct_hp))
show_grid(np.array(list_coef_hpf)[:,:20,:20], [f'fc={item}' for item in list(list_fc)])
show_grid(np.array(list_idct), [f'fc={item}' for item in list(list_fc)])


# ![image.png](attachment:de48fd08-536d-46ec-8e0d-96fff6798fe6.png)

# In[19]:


## survey with many different frequencies
list_exp = np.arange(1, 5, 1)
list_fc = np.power(2, list_exp)
list_coef_hpf = []
list_idct = []
for fc in list_fc:
    baboon_dct_hp = high_pass_filter(baboon_dct, fc)
    list_coef_hpf.append(baboon_dct_hp)
    list_idct.append(cv2.idct(baboon_dct_hp))
show_grid(np.array(list_coef_hpf)[:,:20,:20], [f'fc={item}' for item in list(list_fc)])
show_grid(np.array(list_idct), [f'fc={item}' for item in list(list_fc)])


# In[20]:


## survey with many different frequencies
list_exp = np.arange(5, 9, 1)
list_fc = np.power(2, list_exp)
list_coef_hpf = []
list_idct = []
for fc in list_fc:
    baboon_dct_hp = high_pass_filter(baboon_dct, fc)
    list_coef_hpf.append(baboon_dct_hp)
    list_idct.append(cv2.idct(baboon_dct_hp))
show_grid(np.array(list_coef_hpf)[:,:,:], [f'fc={item}' for item in list(list_fc)])
show_grid(np.array(list_idct), [f'fc={item}' for item in list(list_fc)])


# <!-- ![image.png](attachment:58cb999f-2551-47c5-b4be-af1bdceda8a9.png) -->

# ### Wavelet Transform

# In[21]:


lena_wavelet_coef = pywt.dwt2(lena_arr, 'db1')
lena_ll, (lena_lh, lena_hl, lena_hh) = lena_wavelet_coef
plt.figure(figsize=(5, 5), dpi=80)

plt.subplot(2, 2, 1)
plt.imshow(lena_ll,cmap='gray')
plt.title("LL")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(lena_lh,cmap='gray')
plt.title("HL")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(lena_hl,cmap='gray')
plt.title("LH")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(lena_hh,cmap='gray')
plt.title("HH")
plt.axis('off')
plt.savefig('./result/lena_dwt2.png')


# In[22]:


baboon_wavelet_coef = pywt.dwt2(baboon_arr, 'db1')
bb_ll, (bb_lh, bb_hl, bb_hh) = baboon_wavelet_coef
plt.figure(figsize=(5, 5), dpi=80)

plt.subplot(2, 2, 1)
plt.imshow(bb_ll,cmap='gray')
plt.title("LL")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(bb_lh,cmap='gray')
plt.title("HL")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(bb_hl,cmap='gray')
plt.title("LH")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(bb_hh,cmap='gray')
plt.title("HH")
plt.axis('off')
plt.savefig('./result/baboon_dwt2.png')


# In[23]:


lena_idwt = pywt.waverec2(lena_wavelet_coef, 'db1')
print('Max different: {0}'.format(np.max(lena_idwt-lena_arr)))
plt.imshow(lena_idwt, cmap='gray')


# In[24]:


lena2_wavelet_coef = pywt.dwt2(lena_ll, 'db1')
lena2_ll, (lena2_lh, lena2_hl, lena2_hh) = lena2_wavelet_coef

plt.figure(figsize=(5, 5), dpi=80)

plt.subplot(2, 2, 1)
plt.imshow(lena2_ll,cmap='gray')
plt.title("LL")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(lena2_lh,cmap='gray')
plt.title("HL")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(lena2_hl,cmap='gray')
plt.title("LH")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(lena2_hh,cmap='gray')
plt.title("HH")
plt.axis('off')
plt.savefig('./result/lena2_dwt2.png')


# In[25]:


baboon2_wavelet_coef = pywt.dwt2(bb_ll, 'db1')
bb2_ll, (bb2_lh, bb2_hl, bb2_hh) = baboon2_wavelet_coef

plt.figure(figsize=(5, 5), dpi=80)

plt.subplot(2, 2, 1)
plt.imshow(bb2_ll,cmap='gray')
plt.title("LL")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(bb2_lh,cmap='gray')
plt.title("HL")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(bb2_hl,cmap='gray')
plt.title("LH")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(bb2_hh,cmap='gray')
plt.title("HH")
plt.axis('off')
plt.savefig('./result/baboon2_dwt2.png')


# In[26]:


lena_inv_pb = pywt.waverec2(
    (lena_wavelet_coef[0], (np.zeros((256,256)), np.zeros((256,256)), np.zeros((256,256)))),     
     'db1')
print('Max different: {0}'.format(np.max(lena_inv_pb-lena_arr)))
plt.imshow(lena_inv_pb, cmap='gray')


# In[27]:


lena_inv_pb = pywt.waverec2(
    (np.zeros((256,256)), lena_wavelet_coef[1]),
    'db1')
print('Max different: {0}'.format(np.max(lena_inv_pb-lena_arr)))
plt.imshow(lena_inv_pb, cmap='gray')


# In[28]:


lena_inv_pb = pywt.waverec2(
    (np.zeros((256,256)), 
     (lena_wavelet_coef[1][0], lena_wavelet_coef[1][1],
     np.zeros((256,256)))),
    'db1')
print('Max different: {0}'.format(np.max(lena_inv_pb-lena_arr)))
plt.imshow(lena_inv_pb, cmap='gray')


# In[29]:


lena_inv_pb = pywt.waverec2(
    (np.zeros((256,256)), 
     (np.zeros((256,256)), np.zeros((256,256)),
     lena_wavelet_coef[1][-1])),
    'db1')
print('Max different: {0}'.format(np.max(lena_inv_pb-lena_arr)))
plt.imshow(lena_inv_pb, cmap='gray')


# In[30]:


baboon_inv_pb = pywt.waverec2(
    (baboon_wavelet_coef[0], (np.zeros((256,256)), np.zeros((256,256)), np.zeros((256,256)))),     
     'db1')
print('Max different: {0}'.format(np.max(baboon_inv_pb-baboon_arr)))
plt.imshow(baboon_inv_pb, cmap='gray')


# In[31]:


baboon_inv_pb = pywt.waverec2(
    (np.zeros((256,256)), baboon_wavelet_coef[1]),
    'db1')
print('Max different: {0}'.format(np.max(baboon_inv_pb-baboon_arr)))
plt.imshow(baboon_inv_pb, cmap='gray')


# In[32]:


baboon_inv_pb = pywt.waverec2(
    (np.zeros((256,256)), 
     (baboon_wavelet_coef[1][0], baboon_wavelet_coef[1][1],
     np.zeros((256,256)))),
    'db1')
print('Max different: {0}'.format(np.max(baboon_inv_pb-baboon_arr)))
plt.imshow(baboon_inv_pb, cmap='gray')


# In[33]:


baboon_inv_pb = pywt.waverec2(
    (np.zeros((256,256)), 
     (np.zeros((256,256)), np.zeros((256,256)),
     baboon_wavelet_coef[1][-1])),
    'db1')
print('Max different: {0}'.format(np.max(baboon_inv_pb-baboon_arr)))
plt.imshow(baboon_inv_pb, cmap='gray')


# #### Mix the LL sub-band of Lena with HL-LH-HH sub-bands of Baboon

# In[34]:


mix_inv = pywt.waverec2(
    (lena_wavelet_coef[0], baboon_wavelet_coef[1]), 'db1')
plt.imshow(mix_inv, cmap='gray')


# #### Objective measures

# In[35]:


def compute_psnr(ref, dis):
    """
    calculate the PSNR objective measure score
    param:
        ref: is the reference image
        dis: is the distortion image
    return:
        psnr: PSNR score
    """
    mse = np.mean((ref - dis) ** 2)
    if(mse == 0):
        return 100
#     max_pixel = 255.0
    max_pixel = np.max(ref)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compute_ncc(ref, dis):
    """
    calculate the normalize cross-correlation objective measure score
    param:
        ref: is the reference image
        dis: is the distortion image
    return:
        psnr: NCC score
    """
    numerator = np.sum(np.multiply(ref, dis))
    denominator = np.sum(np.multiply(ref, ref))
    
    return numerator/denominator


# In[36]:


lena_inv_pb = pywt.waverec2(lena_wavelet_coef,'db1')
print('Max different: {0}'.format(np.max(lena_inv_pb-lena_arr)))
plt.imshow(lena_inv_pb, cmap='gray')
print('PSNR DWT: {0}'.format(compute_psnr(ref = lena_arr, dis = lena_inv_pb)))


# In[37]:


lena_inv_pb = pywt.waverec2(
    (lena_wavelet_coef[0], 
     (np.zeros((256,256)), 
      np.zeros((256,256)), 
      np.zeros((256,256)))
    ),'db1')
print('Max different: {0}'.format(np.max(lena_inv_pb-lena_arr)))
plt.imshow(lena_inv_pb, cmap='gray')
print('PSNR DWT: {0}'.format(compute_psnr(ref = lena_arr, dis = lena_inv_pb)))
print('NCC DWT: {0}'.format(compute_ncc(ref = lena_arr, dis = lena_inv_pb)))


# ### Devoir

# ![image.png](attachment:06be3c70-bce2-46ad-8c4c-ea4a28080f1f.png)
# ![image.png](attachment:bb229bd1-f4e6-454e-8077-8ec32ae5a12b.png)

# In[38]:


def mix_coeffs(dct_coef_1, dct_coef_2, fc):
    """
    mix coefficients, in the low-freq area take from coef_1, in the high-freq area take from coef_2
    param:
        dct_coef_1, dct_coef_2: is the coefficients of image after applying DCT transform
        fc: is the cut frequency
    return:
        mix_coef: is the mix coefficients with the cut frequency
    """
    if dct_coef_1.shape == dct_coef_2.shape:
        
        mix_coef = np.zeros(dct_coef_1.shape)
        mix_coef[:fc, :fc] = dct_coef_1[:fc, :fc]
        mix_coef[fc:, fc:] = dct_coef_2[fc:, fc:]
        return mix_coef


# In[39]:


print('='*20,'Mix Lena_Baboon','='*20)
list_exp = np.arange(1, 5, 1)
list_fc = np.power(2, list_exp)
list_coef_mix = []
list_idct = []
for fc in list_fc:
    dct_mix = mix_coeffs(lena_dct, baboon_dct, fc)
    idct_mix = cv2.idct(dct_mix)
    list_coef_mix.append(dct_mix)
    list_idct.append(idct_mix)
show_grid(np.array(list_coef_mix)[:,:20,:20], [f'fc={item}' for item in list(list_fc)])
show_grid(np.array(list_idct), [f'fc={item}' for item in list(list_fc)])


# In[40]:


print('='*20,'Mix Lena_Baboon','='*20)
list_exp = np.arange(5, 9, 1)
list_fc = np.power(2, list_exp)
list_coef_mix = []
list_idct = []
for fc in list_fc:
    dct_mix = mix_coeffs(lena_dct, baboon_dct, fc)
    idct_mix = cv2.idct(dct_mix)
    list_coef_mix.append(dct_mix)
    list_idct.append(idct_mix)
show_grid(np.array(list_coef_mix)[:,:20,:20], [f'fc={item}' for item in list(list_fc)])
show_grid(np.array(list_idct), [f'fc={item}' for item in list(list_fc)])


# In[41]:


print('='*20,'Mix Baboon_Lena','='*20)
list_exp = np.arange(1, 5, 1)
list_fc = np.power(2, list_exp)
list_coef_mix = []
list_idct = []
for fc in list_fc:
    dct_mix = mix_coeffs(baboon_dct, lena_dct, fc)
    idct_mix = cv2.idct(dct_mix)
    list_coef_mix.append(dct_mix)
    list_idct.append(idct_mix)
show_grid(np.array(list_coef_mix)[:,:20,:20], [f'fc={item}' for item in list(list_fc)])
show_grid(np.array(list_idct), [f'fc={item}' for item in list(list_fc)])


# In[42]:


print('='*20,'Mix Baboon_Lena','='*20)
list_exp = np.arange(5, 9, 1)
list_fc = np.power(2, list_exp)
list_coef_mix = []
list_idct = []
for fc in list_fc:
    dct_mix = mix_coeffs(baboon_dct, lena_dct, fc)
    idct_mix = cv2.idct(dct_mix)
    list_coef_mix.append(dct_mix)
    list_idct.append(idct_mix)
show_grid(np.array(list_coef_mix)[:,:20,:20], [f'fc={item}' for item in list(list_fc)])
show_grid(np.array(list_idct), [f'fc={item}' for item in list(list_fc)])


# In[43]:


def evaluate_quality_dct(img_orig, img_dct):
    """
    evaluation image quality follow many different cut frequencies
    by objective measures: PSNR and NNC
    param:
        img_orig: original image
        img_dct: DCT transformed image
    return:
        df_quality: dataframe represents a table as the trend of quality
    """
    list_exp = np.arange(1, 9, 1)
    list_fc = np.power(2, list_exp)
    list_psnr = []
    list_ncc = []
    df_quality = pd.DataFrame([])
    for fc in list_fc:
        img_dct_lp = low_pass_filter(img_dct, fc)
        img_idct = cv2.idct(img_dct_lp)
        psnr = compute_psnr(img_orig, img_idct.astype(np.uint8))
        ncc = compute_ncc(img_orig, img_idct.astype(np.uint8))
        
        list_psnr.append(psnr)
        list_ncc.append(ncc)
    df_quality['cut-frequency'] = list_fc
    df_quality['PSNR-score'] = list_psnr
    df_quality['NCC-score'] = list_ncc
    return df_quality


# In[44]:


print('='*10,'Table Comparison Quality between Lena and Lena_DCT','='*10)
evaluate_quality_dct(lena_arr, lena_dct)


# In[45]:


print('='*10,'Table Comparison Quality between Baboon and Baboon_DCT','='*10)
evaluate_quality_dct(baboon_arr, baboon_dct)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script SPRCA_Master3IR-M2-TP2.ipynb')

