
import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2

data = "C:/Users/nhs80192/Desktop/python3/images/Pet"
categories = ["Dog" ,"Cat"]

for category in categories:
    path = os.path.join(data,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break

print(img_array.shape)

img_size = 50
new_array = cv2.resize(img_array,(img_size,img_size))
plt.show()
