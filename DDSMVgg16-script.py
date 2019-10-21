#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers
from keras import backend as k
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.optimizers import SGD, rmsprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import numpy
from keras.layers.advanced_activations import LeakyReLU


# In[1]:


#import os.path
#os.path.exists('allmias1/trainall-mias1')


# In[3]:


#Check Directory
#Adjust samples, epochs, batch size
img_width, img_height = 224, 224
train_data_dir = '/scratch/star/SCRIP/Sub/trainTF_JPG'
validation_data_dir = '/scratch/star/SCRIP/Sub/valTF_JPG'
test_data_dir = '/scratch/star/SCRIP/Sub/testTF_JPG'
train_samples = 5643
validation_samples = 2823
test_samples = 2823
epochs = 10
batch_size = 64


# In[4]:


#Model
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential()
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model = Model(inputs=base_model.input, outputs=model(base_model.output))
for layer in base_model.layers:
    layer.trainable = False
print('base_model is now NOT trainable')


# In[ ]:


model.compile(optimizer=SGD(lr=0.00001, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("model compiled")


# In[5]:


train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "sparse", shuffle=True)

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "sparse", shuffle=True)

test_generator = test_datagen.flow_from_directory(
test_data_dir,
target_size = (img_height, img_width),
class_mode = "sparse", shuffle=False)


# In[ ]:


history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples/batch_size,
    epochs=4,
    validation_data=validation_generator,
    validation_steps=validation_samples/batch_size)


# In[6]:


for layer in model.layers[:15]:
    layer.trainable = False
for layer in model.layers[15:]:
    layer.trainable = True
model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=["accuracy"])


# In[7]:


history= model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples/batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_samples/batch_size)


# In[ ]:


##Plot Accuracy
import matplotlib.pyplot as plt
print(history.history.keys())

plt.figure()
plt.plot(history.history['acc'],'orange',label='Training accuracy')
plt.plot(history.history['val_acc'],'blue',label='Validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
saveto = 'images/'
#plt.show
plt.savefig(saveto + 'SubDDVplot5.png')


# In[ ]:


plt.figure()
plt.plot(history.history['loss'],'red',label='Training loss')
plt.plot(history.history['val_loss'],'green',label='Validation loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
saveto = 'images/'
#plt.show
plt.savefig(saveto + 'SubDDVplot6.png')


# In[ ]:


y_true = test_generator.classes
predict = model.predict_generator(test_generator,test_generator.samples / test_generator.batch_size )
#y_pred = predict > 0.5
#Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(predict, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))
print('Classification Report')
class_labels = list(test_generator.class_indices.keys())
print(class_labels)
#target_names = ['Normal', 'Abnormal']
print(classification_report(y_true, y_pred, target_names=class_labels))
accuracy = metrics.accuracy_score(y_true, y_pred)
print('Accuracy: ',accuracy)


# In[ ]:


test_loss, test_acc = model.evaluate_generator(test_generator, test_generator.samples / test_generator.batch_size, verbose=1)
print('test acc:', test_acc)


# ### Evaluation

# In[ ]:


# auc = roc_auc_score(y_true, y_pred)
# print('AUC: %.2f' % auc)
#
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
# roc_auc_score(y_true, y_pred)
# fpr, tpr, tresholds = roc_curve(y_true, y_pred)
# auc_keras = auc(fpr, tpr)
#
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='area = {:.3f}'.format(auc_keras))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.savefig(saveto + 'DDVplot7.png')


# In[ ]:
