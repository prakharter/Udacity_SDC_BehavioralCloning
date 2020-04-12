import pandas as pd
import matplotlib.pyplot as plt

data_folder = 'data/'

drive_log = pd.read_csv(data_folder + 'driving_log.csv')
drive_log.head()

drive_log = drive_log[drive_log['steering'] != 0].append(drive_log[drive_log['steering'] == 0].sample(frac=0.2))
print('all data loaded')

import numpy as np
from skimage import io, color, exposure, filters, img_as_ubyte
from skimage.transform import resize
from skimage.util import random_noise

def generate_data(line):
    type2data = {}
    #print('inside geenrate data')
    # center image
    center_img = io.imread(data_folder + line['center'].strip())
    center_ang = line['steering']
    type2data['center'] = (center_img, center_ang)
    
    # flip image if steering is not 0
    if line['steering']:
        flip_img = center_img[:, ::-1]
        flip_ang = center_ang * -1
        type2data['flip'] = (flip_img, flip_ang)
    
    # left image 
    left_img = io.imread(data_folder + line['left'].strip())
    left_ang = center_ang + .2+ .05 * np.random.random()
    left_ang = min(left_ang, 1)
    type2data['left_camera'] = (left_img, left_ang)
    
    # right image
    right_img = io.imread(data_folder + line['right'].strip())
    right_ang = center_ang - .2 - .05 * np.random.random()
    right_ang = max(right_ang, -1)
    type2data['right_camera'] = (right_img, right_ang)
    
    # minus brightness
    aug_img = color.rgb2hsv(center_img)
    aug_img[:, :, 2] *= .5 + .4 * np.random.uniform()
    aug_img = img_as_ubyte(color.hsv2rgb(aug_img))
    aug_ang = center_ang
    type2data['minus_brightness'] = (aug_img, aug_ang)
    
    # equalize_hist
    aug_img = np.copy(center_img)
    for channel in range(aug_img.shape[2]):
        aug_img[:, :, channel] = exposure.equalize_hist(aug_img[:, :, channel]) * 255
    aug_ang = center_ang
    type2data['equalize_hist'] = (aug_img, aug_ang)
    
    # blur image
    blur_img = img_as_ubyte(np.clip(filters.gaussian(center_img, multichannel=True), -1, 1))
    blur_ang = center_ang
    type2data['blur'] = (blur_img, blur_ang)
    
    # noise image
    noise_img = img_as_ubyte(random_noise(center_img, mode='gaussian'))
    noise_ang = center_ang
    type2data['noise'] = (noise_img, noise_ang)
    
    # crop all images
    for name, (img, ang) in type2data.items():
        img = img[60: -25, ...]
        type2data[name] = (img, ang)
    
    return type2data

def show_data(type2data):
    print('isnide showdata')
    col = 4
    row = 1 + len(type2data) // 4
    
    f, axarr = plt.subplots(2, col, figsize=(16, 4))

    for idx, (name, (img, ang)) in enumerate(type2data.items()):
        axarr[idx//col, idx%col].set_title('{}:{:f}'.format(name, ang))
        axarr[idx//col, idx%col].imshow(img)

    plt.show()

type2data = generate_data(drive_log.iloc[0])
#show_data(type2data)


# Generate all rows of data
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print('inside warnings')
    X_train, y_train = [], []
    for idx, row in drive_log.iterrows():
        type2data = generate_data(row)
        for img, ang in type2data.values():
            X_train.append(img)
            y_train.append(ang)
            print('appending xtrain ytrain')

X_train = np.array(X_train)
y_train = np.array(y_train)

import sys

gb = (sys.getsizeof(X_train) + sys.getsizeof(y_train)) / 2**30
print('size: {:f} GB'.format(gb))

#save the training data
np.save('X_train', X_train)
np.save('y_train', y_train)