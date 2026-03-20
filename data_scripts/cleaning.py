# This script removes corrupted and duplicate images from the training subset of DocTamperV1.

import os
import cv2
import hashlib

dataset_path = "DocTamperV1-TrainingSubset/images"
valid_images = []
hashes = set()

for file in os.listdir(dataset_path):

    path = os.path.join(dataset_path, file)

    try:
        img = cv2.imread(path)

        if img is None:
            print("Corrupted image removed:", file)
            os.remove(path)
            continue

        # create hash to detect duplicates
        img_hash = hashlib.md5(img.tobytes()).hexdigest()

        if img_hash in hashes:
            print("Duplicate image removed:", file)
            os.remove(path)
        else:
            hashes.add(img_hash)
            valid_images.append(file)

    except:
        print("Error reading:", file)

print("Cleaning completed")
print("Total valid images:", len(valid_images))

# This script removes corrupted and duplicate images from the testing subset of DocTamperV1.

import os
import cv2
import hashlib

dataset_path = "DocTamperV1-TestingSubset/images"
valid_images = []
hashes = set()

for file in os.listdir(dataset_path):

    path = os.path.join(dataset_path, file)

    try:
        img = cv2.imread(path)

        if img is None:
            print("Corrupted image removed:", file)
            os.remove(path)
            continue

        # create hash to detect duplicates
        img_hash = hashlib.md5(img.tobytes()).hexdigest()

        if img_hash in hashes:
            print("Duplicate image removed:", file)
            os.remove(path)
        else:
            hashes.add(img_hash)
            valid_images.append(file)

    except:
        print("Error reading:", file)

print("Cleaning completed")
print("Total valid images:", len(valid_images))

# This script removes corrupted and duplicate images from the FCD subset of DocTamperV1.

import os
import cv2
import hashlib

dataset_path = "DocTamperV1-FCDSubset/images"
valid_images = []
hashes = set()

for file in os.listdir(dataset_path):

    path = os.path.join(dataset_path, file)

    try:
        img = cv2.imread(path)

        if img is None:
            print("Corrupted image removed:", file)
            os.remove(path)
            continue

        # create hash to detect duplicates
        img_hash = hashlib.md5(img.tobytes()).hexdigest()

        if img_hash in hashes:
            print("Duplicate image removed:", file)
            os.remove(path)
        else:
            hashes.add(img_hash)
            valid_images.append(file)

    except:
        print("Error reading:", file)

print("Cleaning completed")
print("Total valid images:", len(valid_images))

# This script removes corrupted and duplicate images from the SCD subset of DocTamperV1.

import os
import cv2
import hashlib

dataset_path = "DocTamperV1-SCDSubset/images"
valid_images = []
hashes = set()

for file in os.listdir(dataset_path):

    path = os.path.join(dataset_path, file)

    try:
        img = cv2.imread(path)

        if img is None:
            print("Corrupted image removed:", file)
            os.remove(path)
            continue

        # create hash to detect duplicates
        img_hash = hashlib.md5(img.tobytes()).hexdigest()

        if img_hash in hashes:
            print("Duplicate image removed:", file)
            os.remove(path)
        else:
            hashes.add(img_hash)
            valid_images.append(file)

    except:
        print("Error reading:", file)

print("Cleaning completed")
print("Total valid images:", len(valid_images))
