
from skimage import io
import cv2
import os
import matplotlib.pyplot as plt

#load images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)    
    return images

images = load_images_from_folder("./images")   
print("******************* Loaded Data *******************") 
print(images)
print("***************************************************")

