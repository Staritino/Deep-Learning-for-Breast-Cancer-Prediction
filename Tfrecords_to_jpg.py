#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave


# ### Normal and Abnormal

# In[2]:


import os.path
os.path.exists('training10_0.tfrecords')


# In[3]:


def read_and_decode_single_example(filenames):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
    
    reader = tf.TFRecordReader()
    
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label_normal': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        })
    
    # now return the converted data
    label_normal = features['label_normal']
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [299, 299, 1])
    
    # scale the image
    image = tf.image.per_image_standardization(image)
    
    # random flip image
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    #image = tf.image.random_brightness(image, max_delta=10)
    #image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    
    return label_normal, image


# In[4]:


label_normal, image = read_and_decode_single_example(["training10_0.tfrecords", "training10_1.tfrecords", "training10_2.tfrecords", "training10_3.tfrecords", "training10_4.tfrecords"])

images_batch, labels_batch = tf.train.batch([image, label_normal], batch_size=64, capacity=2000)

global_step = tf.Variable(0, trainable=False)


# In[5]:


dest_dir = "images_10"
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)
if not os.path.exists('dest_dir/0'):    
    os.mkdir(os.path.join(dest_dir, "0"))
if not os.path.exists('dest_dir/1'):
    os.mkdir(os.path.join(dest_dir, "1"))

# how many total images are there?
total_images = 55890
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    counter = 0
    while counter < total_images:
        la_b, im_b = sess.run([labels_batch, images_batch])
        
        # go through each image in the batch and save it to a directory depending on its class
        for image, label_normal in zip(im_b, la_b):
            # create the filename from the counter so each image has a distinct name
            filename = str(counter) + ".jpg"
            
            # put the image in a directory according to its label
            image_dir = os.path.join(dest_dir, str(label_normal))
            
            # reshape the image
            image = image.reshape((299,299))
            
            # save the image
            imsave(os.path.join(image_dir, filename), image)

            counter += 1
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)


# In[5]:


import os.path
os.path.exists('TF_JPG/0')


# ### Splitting

# In[2]:


import os
import numpy as np
import shutil

# # Creating Train / Val / Test folders (One time use)
root_dir = 'TF_JPG'
Zero = 'TF_JPG/Normal'
One = 'TF_JPG/Abnormal'
#Two = 'TF_JPG/Malignant'
#Three = 'TF_JPG/3'
#Four = 'TF_JPG/4'


os.makedirs(root_dir +'/train' + Zero)
os.makedirs(root_dir +'/train' + One)
#os.makedirs(root_dir +'/train' + Two)
#os.makedirs(root_dir +'/train' + Three)
#os.makedirs(root_dir +'/train' + Four)

os.makedirs(root_dir +'/val' + Zero)
os.makedirs(root_dir +'/val' + One)
#os.makedirs(root_dir +'/val' + Two)
#os.makedirs(root_dir +'/val' + Three)
#os.makedirs(root_dir +'/val' + Four)

os.makedirs(root_dir +'/test' + Zero)
os.makedirs(root_dir +'/test' + One)
#os.makedirs(root_dir +'/test' + Two)
#os.makedirs(root_dir +'/test' + Three)
#os.makedirs(root_dir +'/test' + Four)


# In[3]:


Zero1 = 'TF_JPG/Normal'
currentCls = Zero1
src = currentCls # Folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
    [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])


train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))


# In[4]:


One1 = 'TF_JPG/Abnormal'
currentCls = One1
src = currentCls # Folder to copy images from

allFileNames4 = os.listdir(src)
np.random.shuffle(allFileNames4)
train_FileNames4, val_FileNames4, test_FileNames4 = np.split(np.array(allFileNames4),
    [int(len(allFileNames4)*0.7), int(len(allFileNames4)*0.85)])


train_FileNames4 = [src+'/'+ name for name in train_FileNames4.tolist()]
val_FileNames4 = [src+'/' + name for name in val_FileNames4.tolist()]
test_FileNames4 = [src+'/' + name for name in test_FileNames4.tolist()]

print('Total images: ', len(allFileNames4))
print('Training: ', len(train_FileNames4))
print('Validation: ', len(val_FileNames4))
print('Testing: ', len(test_FileNames4))


# In[5]:


for name in train_FileNames:
    shutil.copy(name, "TF_JPG/train"+ Zero1)

for name in val_FileNames:
    shutil.copy(name, "TF_JPG/val"+Zero1)

for name in test_FileNames:
    shutil.copy(name, "TF_JPG/test"+Zero1)
    
####
for name in train_FileNames4:
    shutil.copy(name, "TF_JPG/train"+One1)

for name in val_FileNames4:
    shutil.copy(name, "TF_JPG/val"+One1)

for name in test_FileNames4:
    shutil.copy(name, "TF_JPG/test"+One1)
    
 #####   
#for name in train_FileNames1:
    #shutil.copy(name, "TF_JPG/train"+Two1)

#for name in val_FileNames1:
    ##shutil.copy(name, "TF_JPG/val"+Two1)

#for name in test_FileNames1:
    #shutil.copy(name, "TF_JPG/test"+Two1)


# In[6]:


Three1 = 'TF_JPG/3'
currentCls = Three1
src = currentCls # Folder to copy images from

allFileNames2 = os.listdir(src)
np.random.shuffle(allFileNames2)
train_FileNames2, val_FileNames2, test_FileNames2 = np.split(np.array(allFileNames2),
    [int(len(allFileNames2)*0.5), int(len(allFileNames2)*0.75)])


train_FileNames2 = [src+'/'+ name for name in train_FileNames2.tolist()]
val_FileNames2 = [src+'/' + name for name in val_FileNames2.tolist()]
test_FileNames2 = [src+'/' + name for name in test_FileNames2.tolist()]

print('Total images: ', len(allFileNames2))
print('Training: ', len(train_FileNames2))
print('Validation: ', len(val_FileNames2))
print('Testing: ', len(test_FileNames2))


# In[7]:


Four1 = 'TF_JPG/4'
currentCls = Four1
src = currentCls # Folder to copy images from

allFileNames3 = os.listdir(src)
np.random.shuffle(allFileNames3)
train_FileNames3, val_FileNames3, test_FileNames3 = np.split(np.array(allFileNames3),
    [int(len(allFileNames3)*0.5), int(len(allFileNames3)*0.75)])


train_FileNames3 = [src+'/'+ name for name in train_FileNames3.tolist()]
val_FileNames3 = [src+'/' + name for name in val_FileNames3.tolist()]
test_FileNames3 = [src+'/' + name for name in test_FileNames3.tolist()]

print('Total images: ', len(allFileNames3))
print('Training: ', len(train_FileNames3))
print('Validation: ', len(val_FileNames3))
print('Testing: ', len(test_FileNames3))


# In[8]:


for name in train_FileNames:
    shutil.copy(name, "TF_JPG/train"+ Zero1)

for name in val_FileNames:
    shutil.copy(name, "TF_JPG/val"+Zero1)

for name in test_FileNames:
    shutil.copy(name, "TF_JPG/test"+Zero1)
    
####
for name in train_FileNames4:
    shutil.copy(name, "TF_JPG/train"+One1)

for name in val_FileNames4:
    shutil.copy(name, "TF_JPG/val"+One1)

for name in test_FileNames4:
    shutil.copy(name, "TF_JPG/test"+One1)
    
 #####   
for name in train_FileNames1:
    shutil.copy(name, "TF_JPG/train"+Two1)

for name in val_FileNames1:
    shutil.copy(name, "TF_JPG/val"+Two1)

for name in test_FileNames1:
    shutil.copy(name, "TF_JPG/test"+Two1)
    
####       
for name in train_FileNames2:
    shutil.copy(name, "TF_JPG/train"+Three1)

for name in val_FileNames2:
    shutil.copy(name, "TF_JPG/val"+Three1)

for name in test_FileNames2:
    shutil.copy(name, "TF_JPG/test"+Three1)
    
####      
for name in train_FileNames3:
    shutil.copy(name, "TF_JPG/train"+Four1)

for name in val_FileNames3:
    shutil.copy(name, "TF_JPG/val"+Four1)

for name in test_FileNames3:
    shutil.copy(name, "TF_JPG/test"+Four1)
 ####       


# ### 5 Classes
# #Normal
# #Benign Mass
# #Benign Calcification
# #Malignant Mass
# #Malignant Calcification

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave


# In[2]:


import os.path
os.path.exists('TF_JPG')


# In[9]:


def read_and_decode_single_example(filenames):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
    
    reader = tf.TFRecordReader()
    
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        })
    
    # now return the converted data
    label = features['label']
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [299, 299, 1])
    
    # scale the image
    #image = tf.image.per_image_standardization(image)
    
    # random flip image
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    #image = tf.image.random_brightness(image, max_delta=10)
    #image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    
    return label, image


# In[10]:


label, image = read_and_decode_single_example(["training10_0.tfrecords", "training10_1.tfrecords", "training10_2.tfrecords", "training10_3.tfrecords", "training10_4.tfrecords"])

images_batch, labels_batch = tf.train.batch([image, label], batch_size=64, capacity=2000)

global_step = tf.Variable(0, trainable=False)


# In[ ]:





# In[12]:


dest_dir = "TF_JPG"
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)
if not os.path.exists('dest_dir/0'):    
    os.mkdir(os.path.join(dest_dir, "0"))
if not os.path.exists('dest_dir/1'):
    os.mkdir(os.path.join(dest_dir, "1"))
if not os.path.exists('dest_dir/2'):    
    os.mkdir(os.path.join(dest_dir, "2"))
if not os.path.exists('dest_dir/3'):
    os.mkdir(os.path.join(dest_dir, "3"))
if not os.path.exists('dest_dir/4'):    
    os.mkdir(os.path.join(dest_dir, "4"))

total_images = 55890
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    counter = 0
    while counter < total_images:
        la_b, im_b = sess.run([labels_batch, images_batch])
        
        # go through each image in the batch and save it to a directory depending on its class
        for image, label in zip(im_b, la_b):
            # create the filename from the counter so each image has a distinct name
            filename = str(counter) + ".jpg"
            
            # put the image in a directory according to its label
            image_dir = os.path.join(dest_dir, str(label))
            
            # reshape the image
            image = image.reshape((299,299))
            
            # save the image
            imsave(os.path.join(image_dir, filename), image)

            counter += 1
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)


# In[ ]:




