# Generate a small dataset for the frontend demo notebook
import os
import random
import shutil
from glob import glob

imgs_covid = glob('../COVID-19_Radiography_Dataset/COVID/images/*.png')
imgs_normal = glob('../COVID-19_Radiography_Dataset/Normal/images/*.png')
imgs_pneumonia = glob('../COVID-19_Radiography_Dataset/Viral Pneumonia/images/*.png')

imgs_covid_sub = random.sample(imgs_covid, 33)
imgs_normal_sub = random.sample(imgs_normal, 33)
imgs_pneumonia_sub = random.sample(imgs_pneumonia, 33)

os.makedirs('./data_subset_imgs/COVID', exist_ok=True)
os.makedirs('./data_subset_imgs/Normal', exist_ok=True)
os.makedirs('./data_subset_imgs/Viral Pneumonia', exist_ok=True)

for img in imgs_covid_sub:
    shutil.copy(img, './data_subset_imgs/COVID')
for img in imgs_normal_sub:
    shutil.copy(img, './data_subset_imgs/Normal')
for img in imgs_pneumonia_sub:
    shutil.copy(img, './data_subset_imgs/Viral Pneumonia')