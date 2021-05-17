
import os
from skimage import exposure
from skimage import io
from skimage.transform import rescale
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import threshold_otsu
import numpy as np


file_names = [];
#load images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        file_names.append(filename)
        img = io.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)    
    return images

images = load_images_from_folder("./images")   
print("******************* Loaded Data *******************") 
print(images)
print("***************************************************")

#Size 
i = 0;
resized_images = []
for image in images:
    new_image = rescale(image, 0.3, anti_aliasing=False)
    resized_images.append(new_image)
    io.imsave(f'./scaling/resized-{file_names[i]}',new_image)
    i =  i + 1

print("******************* Resized Data *******************") 
print(resized_images)
print("****************************************************")

#Contrast

j = 0
for image in images:
    logarithmic_corrected = exposure.adjust_log(image, 1)
    io.imsave(f'./contrast/{file_names[j]}',logarithmic_corrected)
    j = j + 1

j = 0
for image in resized_images:
    logarithmic_corrected = exposure.adjust_log(image, 1)
    io.imsave(f'./contrast/resized-{file_names[j]}',logarithmic_corrected)
    j = j + 1   

#Denoising     
patch_kw = dict(patch_size=5,      
                patch_distance=6,  
                multichannel=True)
j = 0
for image in images:
    sigma_est = np.mean(estimate_sigma(image, multichannel=True))
    denoise = denoise_nl_means(image, h=0.8 * sigma_est, fast_mode=True,**patch_kw)
    io.imsave(f'./denoising/{file_names[j]}',denoise)
    j = j + 1

j = 0
for image in resized_images:
    sigma_est = np.mean(estimate_sigma(image, multichannel=True))
    denoise = denoise_nl_means(image, h=0.8 * sigma_est, fast_mode=True,**patch_kw)
    io.imsave(f'./denoising/resized-{file_names[j]}',denoise)
    j = j + 1

#Thresholding 
window_size = 25

j = 0
for image in images:
    binary_global = image > threshold_otsu(image)
    io.imsave(f'./thresholding/{file_names[j]}',binary_global)
    j = j + 1

j = 0
for image in resized_images:
    binary_global = image > threshold_otsu(image)
    io.imsave(f'./thresholding/resized-{file_names[j]}',binary_global)
    j = j + 1