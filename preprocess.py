import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.layers import LeakyReLU
import os
import math

def preprocess_data(image_path, box, class_label, input_shape):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))
    image_normalized = image_resized / 255.0

    S = 7
    C = 3
    h, w, _ = image.shape
    x1, y1, x2, y2 = box

    # 中心座標と幅、高さの計算
    x_center = (x1 + x2) / 2 / w
    y_center = (y1 + y2) / 2 / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h

    # グリッドサイズの計算
    grid_size_x = input_shape[1] / S
    grid_size_y = input_shape[0] / S
    x_center_grid = int(x_center * S)
    y_center_grid = int(y_center * S)

    label = np.zeros((S, S, 5 + C))
    label[y_center_grid, x_center_grid, 0:5] = [x_center * S - x_center_grid, y_center * S - y_center_grid, width, height, 1.0]
    label[y_center_grid, x_center_grid, 5:] = np.eye(C)[class_label]

    return image_normalized, label
