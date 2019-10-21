#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers, regularizers, layers, losses, metrics
from keras import backend as K
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
from keras.utils import np_utils, to_categorical
from keras.applications import ResNet50


# In[1]:


#import os.path
#os.path.exists('allmias1/trainall-mias1')


# In[3]:


# conv_base = ResNet50(weights='imagenet',
# include_top=False,
# input_shape=(299, 299, 3))


# In[4]:
K.set_learning_phase(1)

base_model = ResNet50(weights='imagenet', include_top=False)

x = base_model.output

x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

    if layer.name.startswith('bn'):
        layer.call(layer.input, training=False)



# model = Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(5, activation='softmax'))
#model = Model(inputs=conv_base.input, outputs=model(conv_base.output))

for layer in base_model.layers[:]:
    layer.trainable = False

print('conv_base is now NOT trainable')

model.compile(optimizer=SGD(lr=0.0001, momentum=0.8),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("model compiled")


# In[5]:


#Check Directory
#Adjust samples, epochs, batch size
img_width, img_height = 224, 224
train_data_dir = '/scratch/star/SCRIP/Sub/trainTF_JPG'
validation_data_dir = '/scratch/star/SCRIP/Sub/valTF_JPG'
test_data_dir = '/scratch/star/SCRIP/Sub/testTF_JPG'
train_samples = 5643
validation_samples = 2823
test_samples = 2823
epochs = 4
batch_size = 64


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
class_mode = "sparse", shuffle=False)

test_generator = test_datagen.flow_from_directory(
test_data_dir,
target_size = (img_height, img_width),
class_mode = "sparse", shuffle=False)


# In[ ]:


print(train_generator.class_indices)
print(validation_generator.class_indices)
print(test_generator.class_indices)


# In[ ]:


model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples//batch_size,
    epochs=4,
    validation_data=validation_generator,
    validation_steps=validation_samples//batch_size)


# In[6]:


for layer in  base_model.layers[:165]:
    layer.trainable = False
for layer in  base_model.layers[165:]:
    layer.trainable = True
    if layer.name.startswith('bn'):
        layer.call(layer.input, training=False)

print('Last block of the conv_base is now trainable')
model.compile(optimizer=SGD(lr=0.001, momentum=0.8), loss='sparse_categorical_crossentropy', metrics=["accuracy"])


# In[7]:


history= model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples//batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_samples//batch_size)


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
plt.savefig(saveto + 'SubDDRplot5.png')


# In[ ]:


plt.figure()
plt.plot(history.history['loss'],'red',label='Training accuracy')
plt.plot(history.history['val_loss'],'green',label='Validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
saveto = 'images/'
#plt.show
plt.savefig(saveto + 'SubDDRplot6.png')


# In[ ]:


#test_loss, test_acc = model.evaluate_generator(test_generator, steps= 3561 // batch_size, verbose=1)
#print('test acc:', test_acc)


# ### Evaluation

# In[ ]:


y_true = test_generator.classes
predict = model.predict_generator(test_generator, test_generator.samples / test_generator.batch_size)
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

###############################################
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
# plt.savefig(saveto + 'DDRplot7.png')
#plt.show()

# In[ ]:
