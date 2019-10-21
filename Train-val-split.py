
import os
import numpy as np
import shutil

# # Creating Train / Val / Test folders (One time use)
root_dir = 'all-mias1'
Abnormal = 'all-mias1/Abnormal'
Normal = 'all-mias1/Normal'

os.makedirs(root_dir +'/train' + Abnormal)
os.makedirs(root_dir +'/train' + Normal)
os.makedirs(root_dir +'/val' + Abnormal)
os.makedirs(root_dir +'/val' + Normal)
os.makedirs(root_dir +'/test' + Abnormal)
os.makedirs(root_dir +'/test' + Normal)


# In[50]:


# Creating partitions of the data after shufflling
Abnormal1 = 'all-mias1/Abnormal/Thresholded2'
currentCls = Abnormal1
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


# In[51]:


currentCls1 = 'all-mias1/Normal/Thresholded1'
src1 = currentCls1 # Folder to copy images from

allFileNames1 = os.listdir(src1)
np.random.shuffle(allFileNames1)
train_FileNames1, val_FileNames1, test_FileNames1 = np.split(np.array(allFileNames1),
    [int(len(allFileNames1)*0.7), int(len(allFileNames1)*0.85)])


train_FileNames1 = [src1+'/'+ name for name in train_FileNames1.tolist()]
val_FileNames1 = [src1+'/' + name for name in val_FileNames1.tolist()]
test_FileNames1 = [src1+'/' + name for name in test_FileNames1.tolist()]

print('Total images: ', len(allFileNames1))
print('Training: ', len(train_FileNames1))
print('Validation: ', len(val_FileNames1))
print('Testing: ', len(test_FileNames1))


# In[54]:


for name in train_FileNames:
    shutil.copy(name, "all-mias1/train"+Abnormal)

for name in val_FileNames:
    shutil.copy(name, "all-mias1/val"+Abnormal)

for name in test_FileNames:
    shutil.copy(name, "all-mias1/test"+Abnormal)

for name in train_FileNames1:
    shutil.copy(name, "all-mias1/train"+Normal)

for name in val_FileNames1:
    shutil.copy(name, "all-mias1/val"+Normal)

for name in test_FileNames1:
    shutil.copy(name, "all-mias1/test"+Normal)


# In[ ]:
