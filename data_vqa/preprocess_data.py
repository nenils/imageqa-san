import numpy as np
import os
import pickle
import shutil
import random

source_folder = "imageqa-san/data"
dest_folder_1 = "imageqa-san/data/train"
dest_folder_2 = "imageqa-san/data/val"
dest_folder_3 = "imageqa-san/data/test"

if not os.path.exists(dest_folder_1):
    os.makedirs(dest_folder_1)
if not os.path.exists(dest_folder_2):
    os.makedirs(dest_folder_2)
if not os.path.exists(dest_folder_3):
    os.makedirs(dest_folder_3)

file_list = os.listdir(source_folder)
random.shuffle(file_list)

num_train = int(len(file_list) * 0.6)
num_val = int(len(file_list) * 0.2)
num_test = len(file_list) - num_train - num_val

train_image_ids = file_list[:num_train]
val_image_ids = file_list[num_train:num_train + num_val]
test_image_ids = file_list[num_train + num_val:]

for filename in train_image_ids:
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(dest_folder_1, filename)
    shutil.copy(src_path, dst_path)

for filename in val_image_ids:
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(dest_folder_2, filename)
    shutil.copy(src_path, dst_path)

for filename in test_image_ids:
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(dest_folder_3, filename)
    shutil.copy(src_path, dst_path)

image_ids = train_image_ids + val_image_ids
image_ids = [int(float(i)) for i in image_ids]
image_feat = np.zeros((len(image_ids), 4096), dtype='float32')