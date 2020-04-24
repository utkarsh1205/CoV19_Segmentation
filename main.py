from model import *
from model2 import *
import matplotlib.pyplot as plt
import os
from os.path import join
import numpy as np

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

#Creating list of image paths
image_dir = "dataset/Train/Image"
imgs = os.listdir(image_dir)
img_paths = [join(image_dir, filename) for filename in imgs]

#Creating list of mask paths
mask_dir = "dataset/Train/Label"
masks = os.listdir(mask_dir)
mask_paths = [join(mask_dir, filename) for filename in masks]

#For Target Size
image_size = 256

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path,
                     color_mode='grayscale',
                     target_size=(img_height, img_width)) for img_path in img_paths]
    
    img_array = np.array([img_to_array(img) for img in imgs])
    img_array /=255
    return(img_array)

def read_and_prep_masks(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path,
                     color_mode='grayscale',
                     target_size=(img_height, img_width)) for img_path in img_paths]
    
    img_array = np.array([img_to_array(img) for img in imgs])
    return(img_array)


#Create input for model
train_imgs = read_and_prep_images(img_paths)
train_masks = read_and_prep_masks(mask_paths)

#One-Hot encode masks
train_masks = train_masks[...,0]
train_masks=to_categorical(train_masks, num_classes=4, dtype='uint8')

#Initializing model
#model = unet()          #unet from model.py
model = unet2()          #unet from model2.py , uses batch normalisation
model_checkpoint = ModelCheckpoint('unet_covid19.hdf5',
                                    monitor='val_loss',verbose=1,
                                    save_best_only=True)

model_EarlyStopping = EarlyStopping(monitor='val_loss',
                                    patience=4,
                                    restore_best_weights=True)

model.fit(train_imgs,
          train_masks,batch_size=2,
          epochs = 20,
          validation_split=0.2,
          shuffle=True,
          callbacks=[model_checkpoint,model_EarlyStopping]
          )

# from data import testGenerator
# testGene = testGenerator("dataset/Test/Image", num_image = 20)
# results = model.predict_generator(testGene,20,verbose=1)

test_im_dir = "dataset/Test/Image"
test_imgs = os.listdir(test_im_dir)
test_img_paths = [join(test_im_dir, filename) for filename in test_imgs]

test_mask_dir = "dataset/Test/Label"
test_masks = os.listdir(test_mask_dir)
test_mask_paths = [join(test_mask_dir, filename) for filename in test_masks]

test_imgs = read_and_prep_images(test_img_paths)
test_masks = read_and_prep_masks(test_mask_paths)

test_masks = test_masks[...,0]
test_masks=to_categorical(test_masks, num_classes=4,dtype='uint8')

results = model.predict(test_imgs)

plt.subplot(1,2,1)
plt.imshow(test_masks[0,:,:,3])
plt.title("Ground truth")
plt.subplot(1,2,2)
plt.imshow(results[0,:,:,3])
plt.title("predicted")

result1 = results[0,:,:,0]

result1 = result1>0.1
plt.imshow(result1)
plt.imshow(test_masks[0,:,:,0])

#saveResult("dataset/Test/Results2",results)
