import os
from random import shuffle as random_shuffle

import cv2
import numpy as np
import torch
from tensorflow.keras.applications.inception_v3 import preprocess_input
from torch.utils.data import TensorDataset, DataLoader


def extract_xs_ys(ill_positive, negatives, index):
    neg = negatives[index]
    xs = np.concatenate((ill_positive[index], neg[:len(ill_positive[index])]))
    ys = np.array(len(ill_positive[index]) * [1] + len(ill_positive[index]) * [0])
    return xs, ys


def prepr(img_arr, img_size=75):
    if img_arr.shape[-2] == img_size:
        return preprocess_input(img_arr)
    else:
        return cv2.resize(preprocess_input(img_arr), dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)


def load_patient(path, img_size=75):
    imgs = []
    imgs_arr = np.load(path, allow_pickle=True)
    for img_arr in imgs_arr:
        imgs.append(prepr(img_arr, img_size))
    return np.array(imgs)


class Coronary:
    def __init__(self, batch_size, threads):
        path = 'DATA_CORONARY/'
        img_size = 32
        i = 0

        ill_path = os.path.join(path, 'ill/npy')
        ill_positive = []
        ill_negative = []

        for file in sorted(os.listdir(ill_path)):
            if 'pos' in file:
                ill_positive.append(load_patient(os.path.join(ill_path, file), img_size))
            else:
                ill_negative.append(load_patient(os.path.join(ill_path, file), img_size))

        healthy_path = os.path.join(path, 'healthy/npy')
        healthy_negative = []
        for file in sorted(os.listdir(healthy_path)):
            healthy_negative.append(load_patient(os.path.join(healthy_path, file), img_size))

        negatives = []
        for ill, healthy in zip(ill_negative, healthy_negative):
            negatives.append(np.concatenate((ill, healthy)))

        test_x, test_y = extract_xs_ys(ill_positive, negatives, -i)
        train_positive = np.concatenate(ill_positive[:-i - 2] + ill_positive[-i:])
        train_negative = np.concatenate(negatives[:-i - 2] + negatives[-i:])

        len_train_pos = len(train_positive)
        train_negative = train_negative[:len_train_pos]

        train = [x for x in
                 zip(np.concatenate((train_positive, train_negative)), [1] * len_train_pos + [0] * len_train_pos)]
        random_shuffle(train)
        train_xs, train_ys = list(zip(*train))

        test_tensor_x = torch.Tensor(test_x)
        test_tensor_y = torch.Tensor(test_y)

        test_dataset = TensorDataset(test_tensor_x, test_tensor_y)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=threads)

        train_tensor_x = torch.Tensor(train_xs)
        train_tensor_y = torch.Tensor(train_ys)

        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=threads)

        self.train = train_dataloader
        self.test = test_dataloader
