import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import skimage.io as io
import skimage.color as color
import skimage.morphology as morphology
import skimage.feature as feature
import skimage.measure as measure
import skimage.transform as transform

import mrcnn_utils
import mrcnn_visualize
from mrcnn_visualize import display_images
import mrcnn_model as modellib
from mrcnn_model import log

import carplate


# # 车牌定位

config = carplate.CarplateConfig()
# CARPLATE_DIR = './test_image'

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"



def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


###载入验证集

dataset= carplate.CarplateDataset()
dataset.car_data_train('../data2')
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


###载入maskrcnn模型

# Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir='./', config=config)
# Load weights

print("Loading weights：mask_rcnn_carplate_0001.h5")
model.load_weights('./mask_rcnn_carplate_0001.h5', by_name=True)


###检测

image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))

# mrcnn_visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset.class_names)





results = model.detect([image], verbose=1)
ax = get_ax(1)
r = results[0]
mrcnn_visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")




# 假设只有一个车牌
N = r['rois'].shape[0]  #有多少个框
# for i in range(N):
y1, x1, y2, x2 = r['rois'][0]
img = image[y1:y2, x1:x2]
skimage.io.imshow(img)
print(1)
# p = np.where(r['masks'].flatten() == True)[0]
#
# x0 = np.min([i%image.shape[0] for i in p])
# x1 = np.max([i%image.shape[0] for i in p])
# y0 = np.min([i//image.shape[0] for i in p])
# y1 = np.max([i//image.shape[0] for i in p])




# img = image[y0:y1, x0:x1]
# skimage.io.imshow(img)







######字符分割



# 1. 转换为灰度图像
img2 = color.rgb2gray(img)
io.imshow(img2)





# 2. Canny边缘检测并膨胀
img3 = feature.canny(img2, sigma=3)
img4 = morphology.dilation(img3)
io.imshow(img4)




# 3. 标记并筛选区域
label_img = measure.label(img4)
regions = measure.regionprops(label_img)

fig, ax = plt.subplots()
ax.imshow(img, cmap=plt.cm.gray)

def in_bboxes(bbox, bboxes):
    for bb in bboxes:
        minr0, minc0, maxr0, maxc0 = bb
        minr1, minc1, maxr1, maxc1 = bbox
        if minr1 >= minr0 and maxr1 <= maxr0 and minc1 >= minc0 and maxc1 <= maxc0:
            return True
    return False

bboxes = []
for props in regions:
    y0, x0 = props.centroid
    minr, minc, maxr, maxc = props.bbox
    
    if maxc - minc > img4.shape[1] / 7 or maxr - minr < img4.shape[0] / 3:
        continue
        
    bbox = [minr, minc, maxr, maxc]
    if in_bboxes(bbox, bboxes):
        continue
        
    if abs(y0 - img4.shape[0] / 2) > img4.shape[0] / 4:
        continue
        
    bboxes.append(bbox)
    
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-r', linewidth=2)


# In[37]:


# 4. 提取单个字符图像
bboxes = sorted(bboxes, key=lambda x: x[1])
chars = []
for bbox in bboxes:
    minr, minc, maxr, maxc = bbox
    ch = img2[minr:maxr, minc:maxc]
    chars.append(ch)
    io.imshow(ch)
    plt.show()


# # 字符识别

# In[38]:


DATASET_DIR = 'dataset/carplate'
classes = os.listdir(DATASET_DIR + "/ann/")

num_classes = len(classes)
img_rows, img_cols = 20, 20

if K.image_data_format() == 'channels_first':
    input_shape = [1, img_rows, img_cols]
else:
    input_shape = [img_rows, img_cols, 1]


# In[39]:


model_char = Sequential()
model_char.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model_char.add(Conv2D(64, (3, 3), activation='relu'))
model_char.add(MaxPooling2D(pool_size=(2, 2)))
model_char.add(Dropout(0.25))
model_char.add(Flatten())
model_char.add(Dense(128, activation='relu'))
model_char.add(Dropout(0.5))
model_char.add(Dense(num_classes, activation='softmax'))

model_char.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])




model_char.load_weights("./char_cnn.h5")







def extend_channel(data):
    if K.image_data_format() == 'channels_first':
        data = data.reshape(data.shape[0], 1, img_rows, img_cols)
    else:
        data = data.reshape(data.shape[0], img_rows, img_cols, 1)
        
    return data




chars2 = []
for ch in chars:
    chars2.append(transform.resize(ch, [img_rows, img_cols]))
    
chars2 = np.stack(chars2)




ys = np.unique(classes)

p_test = model_char.predict_classes(extend_channel(chars2))
print(' '.join([ys[p_test[i]] for i in range(len(p_test))]))






