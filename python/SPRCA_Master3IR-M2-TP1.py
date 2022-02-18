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
# ## TP1 - ILLUSTRATIONS ON IMAGE ENCRYPTION AND STEGANOGRAPHY
# 1. CAESAR cypher applied to images
# 2. Simple substitution cypher applied to images
# 3. LSB technique
# 4. Conclusion

# ### Importing necessary libraries

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


# In[2]:


## set the necessary path

_DATA = './data'
_RESULT = './result'


# ![image.png](attachment:f81b3e74-5820-4466-bfb7-9853371a1793.png)

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
    fig=plt.figure(figsize=(20, 7))
    columns = len(list_imgs)
    rows = 1
    for i in range(0, columns*rows):
        img = list_imgs[i]
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(titles[i])


# Write the encryption and decryption equations for the CAESAR cypher and and alphabet of 256 symbols (pixels)
# 
# - Calculate Modulo
# 
# `b = a - m. *floor(a./m)`

# In[6]:


list_images = []
list_titles = []
## generate 4 different values of k is [50,100,150,200]
## encrypt for the CAESAR cypher
for k in range(50,250,50):
    lena_lsb_k = np.mod(lena_arr + k, 256)
    list_images.append(lena_lsb_k)
    list_titles.append('K = ' + str(k))
    
show_grid(list_images, list_titles)


# In[7]:


list_images = []
list_titles = []
## generate 4 different values of k is [50,100,150,200]
## encrypt for the CAESAR cypher
for k in range(50,250,50):
    baboon_lsb_k = np.mod(baboon_arr + k, 256)
    list_images.append(baboon_lsb_k)
    list_titles.append('K = ' + str(k))
    
show_grid(list_images, list_titles)


# #### Random generate a Key to map into image as encoding
# 
# each pixel level maps with a value in K, so create 256 K values correspoding to 0 to 255 in 8bits image.
# 
# such as k = [9, 113, 214, ...] and pixel level is [0,1,2,..]. To encode 0 with k[0], 1 with k[1], and 2 with k[2]. So, `y = k[x]`

# In[8]:


## random permutation K
rand_k = np.random.permutation(256)
print(len(rand_k))
plt.plot(rand_k)


# In[9]:


rand_k[:10]


# In[10]:


def encrypt_image(image, key):
    """
    function to encrypt an image by a key
    param:
        image: original image gray scale
        key: key array
    return:
        encrypted image
    """
    enc_img = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            enc_img[i, j] = key[int(image[i][j])]
    return enc_img

def decrypt_image(image, key):
    """
        function to decrypt an image by a key
    param:
        image: encrypted image
        key: key array
    return:
        decrypted image
    """
    dec_img = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            dec_img[i, j] = np.where(key == image[i][j])[0]
    return dec_img


# In[11]:


lena_enc = encrypt_image(lena_arr, rand_k)
print(lena_enc)
plt.imshow(lena_enc, cmap='gray')


# In[12]:


lena_dec = decrypt_image(lena_enc, rand_k)
print(lena_dec)
plt.imshow(lena_dec, cmap='gray')


# In[13]:


rand_k = np.random.permutation(256)
baboon_enc = encrypt_image(baboon_arr, rand_k)
# print(baboon_enc)
plt.imshow(baboon_enc, cmap='gray')


# In[14]:


baboon_dec = decrypt_image(baboon_enc, rand_k)
print(baboon_dec)
plt.imshow(baboon_dec, cmap='gray')


# ![image.png](attachment:aa985d62-2e1c-4089-b793-099183dc7d19.png)

# In[15]:


def compute_lsb(image):
    """
    decouple into 4 areas
    param:
        original image
    return:
        computed lsb image
    """
    enc_image = np.zeros(image.shape)
    w, h = image.shape
    for i in range(w):
        for j in range(h):
            if ((i <= w //2) and ( j <= h //2)) or ((i > w //2) and ( j > h //2)):
                ## lsb_0
                enc_image[i, j] = image[i,j] - np.mod(image[i,j],2)
            else:
                ## lsb_1
                enc_image[i, j] = image[i,j] - np.mod(image[i,j],2) + 1
    return enc_image


# In[16]:


lena_lsb = compute_lsb(lena_arr)
print(lena_lsb)
plt.imshow(lena_lsb, cmap='gray')


# In[17]:


baboon_lsb = compute_lsb(baboon_arr)
print(baboon_lsb)
plt.imshow(baboon_lsb, cmap='gray')


# In[18]:


## write the compressed image with many different quality values
## Q = [100, 99, 75]
for q in [100, 99, 75]:
    img_out_file = os.path.join(_RESULT, f'lena_lsb_{q}.jpg')
    cv2.imwrite(img_out_file, lena_lsb, [int(cv2.IMWRITE_JPEG_QUALITY), q])


# In[19]:


## write the compressed image with many different quality values
## Q = [100, 99, 75]
for q in [100, 99, 75]:
    img_out_file = os.path.join(_RESULT, f'baboon_lsb_{q}.jpg')
    cv2.imwrite(img_out_file, baboon_lsb, [int(cv2.IMWRITE_JPEG_QUALITY), q])


# In[20]:


## check the lena_lsb
plt.imshow(np.mod(lena_lsb,2), cmap='gray')


# In[21]:


## check the lena_lsb
plt.imshow(np.mod(baboon_lsb,2), cmap='gray')


# In[22]:


## check the compressed image
list_images = []
list_titles = []
for q in [100, 99, 75]:
    img_in_file = os.path.join(_RESULT, f'lena_lsb_{q}.jpg')
    img_arr = cv2.imread(img_in_file)[:,:,2]
    img_arr_w_mod = np.mod(img_arr,2)
    list_images.append(img_arr_w_mod)
    list_titles.append('Q='+str(q))
show_grid(list_images, list_titles)


# In[23]:


## check the compressed image
list_images = []
list_titles = []
for q in [100, 99, 75]:
    img_in_file = os.path.join(_RESULT, f'baboon_lsb_{q}.jpg')
    img_arr = cv2.imread(img_in_file)[:,:,2]
    img_arr_w_mod = np.mod(img_arr,2)
    list_images.append(img_arr_w_mod)
    list_titles.append('Q='+str(q))
show_grid(list_images, list_titles)


# In[24]:


def compute_2lsb(image):
    """
    decouple into 4 areas
    param:
        original image
    return:
        computed 2 lsb image
    """
    enc_image = np.zeros(image.shape)
    w, h = image.shape
    for i in range(w):
        for j in range(h):
            if ((i <= w //2) and ( j <= h //2)):
                ## lsb_0
                enc_image[i, j] = image[i,j] - np.mod(image[i,j],4)
            elif ((i > w //2) and ( j <= h //2)):
                enc_image[i, j] = image[i,j] - np.mod(image[i,j],4) + 1
            elif ((i <= w //2) and ( j > h //2)):
                enc_image[i, j] = image[i,j] - np.mod(image[i,j],4) + 2
            else:
                enc_image[i, j] = image[i,j] - np.mod(image[i,j],4) + 3
    return enc_image


# In[25]:


lena_2lsb = compute_2lsb(lena_arr)
print(lena_2lsb)
plt.imshow(lena_2lsb, cmap='gray')


# In[26]:


baboon_2lsb = compute_2lsb(baboon_arr)
print(baboon_2lsb)
plt.imshow(baboon_2lsb, cmap='gray')


# In[27]:


## write the compressed image with many different quality values
## Q = [100, 99, 75]
for q in [100, 99, 75]:
    img_out_file = os.path.join(_RESULT, f'lena_2lsb_{q}.jpg')
    cv2.imwrite(img_out_file, lena_2lsb, [int(cv2.IMWRITE_JPEG_QUALITY), q])


# In[28]:


## write the compressed image with many different quality values
## Q = [100, 99, 75]
for q in [100, 99, 75]:
    img_out_file = os.path.join(_RESULT, f'baboon_2lsb_{q}.jpg')
    cv2.imwrite(img_out_file, baboon_2lsb, [int(cv2.IMWRITE_JPEG_QUALITY), q])


# In[29]:


## check the lena_2lsb
plt.imshow(np.mod(lena_2lsb,2), cmap='gray')


# In[30]:


## check the lena_2lsb
plt.imshow(np.mod(baboon_2lsb,2), cmap='gray')


# In[31]:


## check the compressed image
list_images = []
list_titles = []
for q in [100, 99, 75]:
    img_in_file = os.path.join(_RESULT, f'lena_2lsb_{q}.jpg')
    img_arr = cv2.imread(img_in_file)[:,:,2]
    img_arr_w_mod = np.mod(img_arr,2)
    list_images.append(img_arr_w_mod)
    list_titles.append('Q='+str(q))
show_grid(list_images, list_titles)


# In[32]:


## check the compressed image
list_images = []
list_titles = []
for q in [100, 99, 75]:
    img_in_file = os.path.join(_RESULT, f'baboon_2lsb_{q}.jpg')
    img_arr = cv2.imread(img_in_file)[:,:,2]
    img_arr_w_mod = np.mod(img_arr,2)
    list_images.append(img_arr_w_mod)
    list_titles.append('Q='+str(q))
show_grid(list_images, list_titles)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script SPRCA_Master3IR-M2-TP1.ipynb')

