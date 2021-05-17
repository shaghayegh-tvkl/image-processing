import os
from skimage import exposure
from skimage import io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import threshold_otsu
import numpy as np
import skimage.transform

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

#Size
def resize(images,dir):
    i = 0;
    resized_images = []
    for image in images:
        new_shape = (image.shape[0] // 3, image.shape[1] // 3, image.shape[2])
        new_image = skimage.transform.resize(image=image, output_shape=new_shape)
        resized_images.append(new_image)
        io.imsave(f'{dir}{file_names[i]}',new_image)
        i =  i + 1
    return resized_images    

#Contrast
def contrast_adjustment(images,dir):
    adjusted_images = []
    j = 0
    for image in images:
        logarithmic_corrected = exposure.adjust_log(image, 1)
        adjusted_images.append(logarithmic_corrected)
        io.imsave(f'{dir}{file_names[j]}',logarithmic_corrected)
        j = j + 1
    return adjusted_images

#Denoising     
def denoising(images,dir):
    denoise_images = []
    patch_kw = dict(patch_size=5,      
                patch_distance=6,  
                multichannel=True)
    j = 0
    for image in images:
        sigma_est = np.mean(estimate_sigma(image, multichannel=True))
        denoise = denoise_nl_means(image, h=0.8 * sigma_est, fast_mode=True,**patch_kw)
        denoise_images.append(denoise)
        io.imsave(f'{dir}{file_names[j]}',denoise)
        j = j + 1     
    return denoise_images    

#Thresholding 
def thresholding(images,dir):
    threshold_images = []
    j = 0
    for image in images:
        binary_global = image > threshold_otsu(image)
        threshold_images.append(binary_global)
        io.imsave(f'{dir}{file_names[j]}',binary_global)
        j = j + 1
    return threshold_images    


images = load_images_from_folder("./images")

#First
resized_images = resize(images,"./scaling-1/resized-")
adjusted_images = contrast_adjustment(images,"./contrast-1/")
adjusted_images_resized = contrast_adjustment(resized_images,"./contrast-1/resized-")

denoise_images = denoising(adjusted_images,"./denoising-1/")
denoise_images_resized = denoising(adjusted_images_resized,"./denoising-1/resized-")

threshold_images = thresholding(denoise_images,"./thresholding-1/")
threshold_images_resied = thresholding(denoise_images_resized,"./thresholding-1/resized-")


#Second
adjusted_images = contrast_adjustment(images,"./contrast-2/")
denoise_images = denoising(adjusted_images,"./denoising-2/")
threshold_images = thresholding(denoise_images,"./thresholding-2/")
resized_images = resize(threshold_images,"./scaling-2/resized-")
