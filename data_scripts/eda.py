import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dataset path
dataset_path = "Training_processed_images"

# List of image files
files = os.listdir(dataset_path)
print(f"Total images: {len(files)}")

# -------------------------------
# Initialize variables for EDA
# -------------------------------
widths, heights = [], []
mean_pixels = []
std_pixels = []
channels_mean = []
channels_std = []

# Sample images to show later
sample_images = []

# Loop through images
for i, file in enumerate(files):
    path = os.path.join(dataset_path, file)
    img = cv2.imread(path)
    
    if img is None:
        continue
    
    # Image size
    h, w, c = img.shape
    widths.append(w)
    heights.append(h)
    
    # Pixel statistics
    mean_pixels.append(np.mean(img))
    std_pixels.append(np.std(img))
    
    # Channel-wise mean and std
    channels_mean.append(np.mean(img, axis=(0,1)))  # BGR
    channels_std.append(np.std(img, axis=(0,1)))
    
    # Save first 6 images for display
    if i < 6:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample_images.append(img_rgb)

# -------------------------------
# 1. Image size distribution
# -------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(widths, bins=20, color='skyblue')
plt.title("Image Width Distribution")
plt.xlabel("Width (pixels)")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.hist(heights, bins=20, color='salmon')
plt.title("Image Height Distribution")
plt.xlabel("Height (pixels)")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 2. Pixel intensity statistics
# -------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(mean_pixels, bins=30, color='green')
plt.title("Mean Pixel Intensity Distribution")
plt.xlabel("Mean Pixel Value")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.hist(std_pixels, bins=30, color='orange')
plt.title("Pixel Std Deviation Distribution")
plt.xlabel("Std of Pixel Values")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 3. Channel-wise mean and std
# -------------------------------
channels_mean = np.array(channels_mean)
channels_std = np.array(channels_std)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.bar(['Blue','Green','Red'], channels_mean.mean(axis=0), color=['blue','green','red'])
plt.title("Average Channel Mean")
plt.ylabel("Pixel Value")

plt.subplot(1,2,2)
plt.bar(['Blue','Green','Red'], channels_std.mean(axis=0), color=['blue','green','red'])
plt.title("Average Channel Std Dev")
plt.ylabel("Pixel Value")
plt.show()

# -------------------------------
# 4. Show sample images
# -------------------------------
plt.figure(figsize=(12,6))
for i, img in enumerate(sample_images):
    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.axis('off')
plt.suptitle("Sample Images from Dataset")
plt.show()
