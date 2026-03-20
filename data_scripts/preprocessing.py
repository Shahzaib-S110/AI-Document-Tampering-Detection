# This script preprocesses images for training.
import os
import cv2
import numpy as np

input_folder = "DocTamperV1-TrainingSubset/images"
output_folder = "Training_processed_images"

os.makedirs(output_folder, exist_ok=True)

IMG_SIZE = 224

for file in os.listdir(input_folder):

    path = os.path.join(input_folder, file)

    img = cv2.imread(path)

    if img is None:
        continue

    # resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # normalize pixel values
    img = img / 255.0

    # convert back to 0-255 for saving
    img = (img * 255).astype(np.uint8)

    save_path = os.path.join(output_folder, file)

    cv2.imwrite(save_path, img)

print("Preprocessing completed")

# This script preprocesses images for testing.
import os
import cv2
import numpy as np

input_folder = "DocTamperV1-TestingSubset/images"
output_folder = "Testing_processed_images"

os.makedirs(output_folder, exist_ok=True)

IMG_SIZE = 224

for file in os.listdir(input_folder):

    path = os.path.join(input_folder, file)

    img = cv2.imread(path)

    if img is None:
        continue

    # resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # normalize pixel values
    img = img / 255.0

    # convert back to 0-255 for saving
    img = (img * 255).astype(np.uint8)

    save_path = os.path.join(output_folder, file)

    cv2.imwrite(save_path, img)

print("Preprocessing completed")
