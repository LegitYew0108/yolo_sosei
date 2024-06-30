import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# CSVファイルの読み込み
csv_path = 'Annotations/Annotations/dayTrain/dayClip1/frameAnnotationsBOX.csv'  # CSVファイルのパスを指定
df = pd.read_csv(csv_path)

# クラスのマッピング
class_mapping = {'stop': 0, 'go': 1, 'yield': 2}  # クラス名とクラスIDのマッピング

# モデルの構築
def create_yolo_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7),strides=(2,2), activation='leakyrelu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)
    x = layers.Conv2D(192, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)
    x = layers.Conv2D(128, (1, 1), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(256, (1, 1), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2),strides=2)(x)
    x = layers.Conv2D(256, (1, 1), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(256, (1, 1), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(256, (1, 1), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(256, (1, 1), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(512, (1, 1), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2),strides=2)(x)
    x = layers.Conv2D(512, (1, 1), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(512, (1, 1), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), strides=(2, 2), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), activation='leakyrelu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='leakyrelu')(x)
    outputs = layers.Dense(7*7*(5+num_classes), activation='sigmoid')(x) #7*7(gridcell size)*(x,y,w,h,confidence)
    model = models.Model(inputs, outputs)
    return model

# データの前処理
def preprocess_data(image_path, box, class_label, input_shape):
    # load image
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))
    image_normalized = image_resized / 255.0
    
    # バウンディングボックスの前処理
    h, w, _ = image.shape
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / w
    y_center = (y1 + y2) / 2 / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h
    
    label = 
    label = [x_center, y_center, width, height] + [1 if i == class_label else 0 for i in range(num_classes)]
    
    return image_normalized, label

# モデルのコンパイルと学習
input_shape = (448, 448, 3)  # リサイズ後のサイズを指定
num_classes = 3  # クラス数（例: stop, go, yield）

# データの前処理
images = []
labels = []
print(df)
for index, row in df.iterrows():
    # ここで絶対パスを構築します
    image_path = os.path.join('dayTrain/', row['Filename'])
    box = [row['Upper left corner X'], row['Upper left corner Y'], row['Lower right corner X'], row['Lower right corner Y']]
    class_label = class_mapping[row['Annotation tag']]
    image, label = preprocess_data(image_path, box, class_label, input_shape)
    images.append(image)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

model = create_yolo_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(images, labels, epochs=10)

# モデルの保存
model.save('yolo_model.h5')
