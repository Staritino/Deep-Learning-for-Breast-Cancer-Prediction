
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import time
from IPython.display import clear_output



CURR_DIR = os.getcwd()

# Point to the PNGs to be used:
IMG_DIR = 'all-mias1/Normal'
if not os.path.exists('all-mias1/Normal/Thresholded1/'):
    SAVE_DIR = os.makedirs('all-mias1/Normal/Thresholded1/')
else:
    SAVE_DIR = 'all-mias1/Normal/Thresholded1/'




def get_hists(image, b):
    hist, bins = np.histogram(img.flatten(), bins=b, range=[0,255])
    cdf = hist.cumsum()
    cdf_normalized = cdf *hist.max()/ cdf.max()

    return [hist, cdf_normalized]


# In[4]:

def plot(img, img_hists):
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')

    plt.subplot(122)
    plt.plot(img_hists[1], color = 'b')
    plt.plot(img_hists[0], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')

    plt.subplots_adjust(top=0.92, bottom=0.08,
                        left=0.10, right=0.95,
                        hspace=0.25, wspace=0.35)


# In[ ]:

# ## Thresholding: Segments the breast from hardware artifacts within the images

# In[5]:


def threshold(img_list, factor = 0.4, select_files = []):
    images_t = []

    def internal(data):
        thresholded = cv2.threshold(data['clahe_img'],
                                    np.median(data['clahe_img']) * factor, 255,
                                    cv2.THRESH_BINARY)[1]     # just the binary image

        _, l, s, _ = cv2.connectedComponentsWithStats(thresholded)
        images_t.append( {'filename': data['filename'],
                          'clahe_img': data['clahe_img'],
                          'thresh_img': thresholded,
                          'factor': factor,
                          'labels':l,                          # labels: contiguous regions in mammogram, labelled
                          'count':s[:, -1]                     # count: count of pixels in each discrete object
                         })

    if not select_files:
        print ('Processing all files')
        for i, data in enumerate(img_list):
            internal(data)

    else:
        print('Processing select files {}'.format(select_files))
        for i, data in enumerate(img_list):
            if data['filename'] in select_files:
                internal(data)
    return images_t


# In[6]:


def save(fn, img, location=SAVE_DIR):
    print('Saving: {}'.format(location + fn))
    cv2.imwrite(location + fn, img)
    time.sleep(1)

def mask(image, labels, region):
    labels = copy.deepcopy(labels)  # create a full, unique copy of labels
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if labels[row, col] != region:
                labels[row, col] = 0  # mask the artifact
            else:
                labels[row, col] = 1  # retain the breast
    return labels


# In[24]:


def clean_art(images_thresh):
    revist = []
    for i, data in enumerate(images_thresh):
        fn, c_img, t_img = data['filename'], data['clahe_img'], data['thresh_img']
        print( 'Processing File: {}'.format(fn))

        plt.subplot(121)
        plt.imshow(c_img, cmap='gray')
        plt.title('Original')
        plt.subplot(122)
        plt.imshow(t_img, cmap='gray')
        plt.title('Binary Threshold')
        plt.show()
        plt.pause(0.1)

        top_regions = np.argpartition(data['count'], -2)[-2:]
        print(len(top_regions))
        top_counts = data['count'][top_regions]
        print ('Top region pixel counts: {}'.format(top_counts))
        my_mask = mask( t_img, data['labels'], region=top_regions[1])
        image = c_img * my_mask

        image = np.array(image, dtype = np.uint8)
        #thresh_image = cv2.threshold(image, np.median(image), 255, cv2.THRESH_BINARY)[1]

        plt.imshow(image, cmap='gray')
        plt.title(fn)
        plt.show()
        #plt.pause(6)

        #input4 = input("Save post processed image (Y/N): ").lower()
        #if input4 == 'y':
        #print(fn)
        save(fn, image)

        clear_output()
    return revist


# In[ ]:





# ## Image enhancement via equalization
# Using CLACHE

# In[19]:


filenames = [ filename for filename in os.listdir(IMG_DIR) if filename.endswith('.pgm')]
#filenames = filenames[18:19]
clahe_images = []
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
os.chdir(IMG_DIR)
for filename in filenames:
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    clahe_images.append({'filename': filename, 'clahe_img': clahe.apply(img)})
os.chdir(CURR_DIR)


#print(filenames)
#print(filenames.index('mdb319.pgm'))

#filenames = filenames[18:19]
# print(filenames)


images_thresh = threshold(clahe_images)
print (len(images_thresh))


# In[ ]:


# print (len(images_thresh[:5]))
# print(images_thresh[0]['filename'])


remaining = clean_art(images_thresh)
remaining_fn = [item['filename'] for item in remaining]





# In[ ]:
