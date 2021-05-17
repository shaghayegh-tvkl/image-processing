
import os
from skimage import exposure
from skimage import io
from skimage.transform import rescale

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
