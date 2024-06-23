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
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(num_classes * 5, activation='sigmoid')(x) # (x, y, w, h, class_probs)
    model = models.Model(inputs, outputs)
    return model

# データの前処理
def preprocess_data(image_path, box, class_label, input_shape):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))
    image_normalized = image_resized / 255.0
    
    # バウンディングボックスの前処理
    h, w, _ = image.shape
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / w
    y_center = (y1 + y2) / 2 / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h
    
    label = [x_center, y_center, width, height] + [1 if i == class_label else 0 for i in range(num_classes)]
    
    return image_normalized, label

# モデルのコンパイルと学習
input_shape = (224, 224, 3)  # リサイズ後のサイズを指定
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

# ダミーデータでの学習（実際にはデータセットを使用）
model.fit(images, labels, epochs=10)

# モデルの保存
model.save('yolo_model.h5')
