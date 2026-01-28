# encoding:utf-8
import os

import cv2
import numpy as np
import scipy.io

base_path = "/mnt/data/small_dataset/flower/flowers-102/"

labels = scipy.io.loadmat(base_path + "imagelabels.mat")
labels = np.array(labels["labels"][0])
print("labels:", labels)
print("label(min,max):", labels.min(), labels.max())

setid = scipy.io.loadmat(base_path + "setid.mat")
validation = np.array(setid["valid"][0])
np.random.shuffle(validation)
train = np.array(setid["trnid"][0])
np.random.shuffle(train)
test = np.array(setid["tstid"][0])
np.random.shuffle(test)
print("tain", train)
print("validation:", validation)
print("test:", test)
print("validation(min,max):", validation.min(), validation.max())
print("tain(min,max):", train.min(), train.max())
print("test(min,max):", test.min(), test.max())

flower_dir = list()
for img in os.listdir(base_path + "jpg"):
    flower_dir.append(os.path.join(base_path + "jpg", img))
flower_dir.sort()
print("flower_dir:", flower_dir)  # （已排序）所有花的文件名


def split_dataset(train, folder_train):
    for tid in train:
        tid = tid - 1
        path_img = flower_dir[tid]
        lable = labels[tid]

        img2 = cv2.imread(path_img)
        img2 = cv2.resize(img2, (256, 256))

        image_name = os.path.basename(path_img)

        # classes = "class_" + str(lable).zfill(5)
        classes = str(lable)
        class_path = os.path.join(folder_train, classes)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        complete_image_path = os.path.join(class_path, image_name)

        if not os.path.exists(complete_image_path):
            cv2.imwrite(complete_image_path, img2)
        else:
            print(f"File {complete_image_path} already exists, skipping.")


folder_train = base_path + "train"
folder_val = base_path + "val"

split_dataset(train, folder_train)
split_dataset(validation, folder_train)
split_dataset(test, folder_val)
