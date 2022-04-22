# Solution by the following group members
# 1. Agha Ahmad (501119910)
# 2. Hina Shafique Awan (501118831)

# Importing Libraries
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator


# Loading Dataset

train_dir = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train"
test_dir = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/test"

# Exploring Train Dataset Directory

print("Train set:\n========================================")
num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
num_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
num_covid = len(os.listdir(os.path.join(train_dir, 'COVID19')))

print(f"PNEUMONIA={num_pneumonia}")
print(f"NORMAL={num_normal}")
print(f"COVID19={num_covid}")

# Exploring Test Dataset Directory

print("Test set:\n========================================")
print(f"PNEUMONIA = {len(os.listdir(os.path.join(test_dir, 'PNEUMONIA')))}")
print(f"NORMAL = {len(os.listdir(os.path.join(test_dir, 'NORMAL')))}")
print(f"COVID19 = {len(os.listdir(os.path.join(test_dir, 'COVID19')))}")

# Data Visualization

# Exploring Pneumonia Data
# Loading Image Data

pneumonia = os.listdir("/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train/PNEUMONIA")
pneumonia_dir = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train/PNEUMONIA"

# Investigating Pneumonia 9 Chest-Xray Images in GrayScale

plt.figure(figsize=(20, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(pneumonia_dir, pneumonia[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.tight_layout()

# Exploring Normal Data
# Loading Image Data

normal = os.listdir("/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train/NORMAL")
normal_dir = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train/NORMAL"

# Investigating Normal 9 Chest-Xray Images in GrayScale

plt.figure(figsize=(20, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(normal_dir, normal[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.tight_layout()


# Exploring Covid19 Data
# Loading Image Data

covid = os.listdir("/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train/COVID19")
covid_dir = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train/COVID19"

# Investigating Covid19 9 Chest-Xray Images in GrayScale

plt.figure(figsize=(20, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(covid_dir, covid[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.tight_layout()

# Investigating A Single Chest-Xray Image
# Loading Image Data

normal_img = os.listdir("/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train/NORMAL")[0]
normal_dir = "/Users/agha/Desktop/MS Data Science/Winter 2022/Deep Learning/project/Project Code/Data/train/NORMAL"
sample_img = plt.imread(os.path.join(normal_dir, normal_img))

# Explorating the Raw Chest-Xray Image

plt.imshow(sample_img, cmap='gray')
plt.colorbar()
plt.title('Raw Chest X Ray Image')

# Explorating the Dimensions, maximum and mean values of pixels of image

print(f"The dimensions of the image are {sample_img.shape[0]} pixels width and {sample_img.shape[1]} pixels height, one single color channel.")
print(f"The maximum pixel value is {sample_img.max():.4f} and the minimum is {sample_img.min():.4f}")
print(f"The mean value of the pixels is {sample_img.mean():.4f} and the standard deviation is {sample_img.std():.4f}")

# Ivestigating Pixel Value Distribution

sns.distplot(sample_img.ravel(),
             label=f"Pixel Mean {np.mean(sample_img):.4f} & Standard Deviation {np.std(sample_img):.4f}", 
             kde=False)
plt.legend(loc='upper right')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')

# Image Preprocessing

image_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)

# Training Data Generation

train = image_generator.flow_from_directory(train_dir, 
                                            batch_size=8, 
                                            shuffle=True, 
                                            class_mode='binary',
                                            target_size=(244, 244))

# Test Data Generation

test = image_generator.flow_from_directory(test_dir, 
                                            batch_size=1, 
                                            shuffle=False, 
                                            class_mode='binary',
                                            target_size=(244, 244))

# Visualization and Clipping the input data to the valid range for imshow with RGB data

sns.set_style('white')
generated_image, label = train.__getitem__(0)
plt.imshow(generated_image[0], cmap='gray')
plt.colorbar()
plt.title('Raw Chest X Ray Image')

# Explorating the Dimensions, maximum and mean values of pixels of preprocessed chest-xray image

print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height, one single color channel.")
print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")

# Ivestigating Pixel Value Distribution after Preprocessing

sns.distplot(generated_image.ravel(),
             label=f"Pixel Mean {np.mean(generated_image):.4f} & Standard Deviation {np.std(generated_image):.4f}", 
             kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')
