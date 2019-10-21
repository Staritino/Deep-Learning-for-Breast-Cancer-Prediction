#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
