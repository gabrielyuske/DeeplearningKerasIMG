#!/usr/bin/env python
# coding: utf-8

# In[274]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import PIL
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# In[275]:


data = "C:/Users/nhs80192/Desktop/python3/images/Pet_images"

# In[276]:


data_dir = pathlib.Path( data )
data_dir

# In[277]:


image_count = len( list( data_dir.glob( "*/*.jpg" ) ) )
image_count

# In[278]:


# cat Dog
dog = list( data_dir.glob( "Dog/*" ) )
dog[:5]

# In[279]:


PIL.Image.open( str( dog[0] ) )

# In[280]:


pets_images_dict = {
    "dog": list( data_dir.glob( "Dog/*" ) ),
    "cat": list( data_dir.glob( "cat/*" ) )
}

# In[281]:


pets_labels_dict = {
    "dog": 0,
    "cat": 1
}

# In[282]:


# pets_images_dict["dog"][0] # == NG


# In[283]:


str( pets_images_dict["dog"][0] )  # == OK

# In[284]:


img = cv2.imread( str( pets_images_dict["dog"][0] ) )
img

# In[285]:


cv2.resize( img, (180, 180) ).shape

# In[286]:


X, y = [], []
for pet_name, images in pets_images_dict.items():
    for image in images:
        img = cv2.imread( str( image ) )
        resized_img = cv2.resize( img, (180, 180) )
        plt.imshow( img, cmap="gray" )
        print( image )
        plt.show()

# In[287]:


X.append( resized_img )
y.append( pets_labels_dict[pet_name] )

# In[288]:


X = np.array( X )
y = np.array( y )

# In[293]:


y

# In[294]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=0 )

# In[295]:


len( X_train )

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:




