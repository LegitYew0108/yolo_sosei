import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.layers import LeakyReLU
import os
import math

# モデルの構築
def create_yolo_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7),strides=(2,2), activation=LeakyReLU(alpha=0.01), padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)
    x = layers.Conv2D(192, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)
    x = layers.Conv2D(128, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(256, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.MaxPooling2D((2, 2),strides=2)(x)
    x = layers.Conv2D(256, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(256, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(256, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(256, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(512, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.MaxPooling2D((2, 2),strides=2)(x)
    x = layers.Conv2D(512, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(512, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), strides=(2, 2), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Conv2D(1024, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation=LeakyReLU(alpha=0.01))(x)
    x = layers.Dense(7*7*(5+num_classes), activation='sigmoid')(x) #7*7(gridcell size)*(x,y,w,h,confidence,go,stop,yield)
    outputs = layers.Reshape((7,7,5+num_classes))(x)
    model = models.Model(inputs, outputs)
    return model

# データの前処理
def preprocess_data(image_path, box, class_label, input_shape):
    # load image
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))
    image_normalized = image_resized / 255.0
    
    # バウンディングボックスの前処理
    S = 7
    h, w, _ = image.shape
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / w
    y_center = (y1 + y2) / 2 / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h
    grid_size_x = input_shape[1]/S
    grid_size_y = input_shape[0]/S
    x_center_grid = math.floor(x_center/grid_size_x)
    y_center_grid = math.floor(y_center/grid_size_y)
    
    label = np.zeros((7,7,5+3))
    label[x_center_grid,y_center_grid,0:5] = [x_center,y_center,width,height,1.0]

    label[x_center_grid,y_center_grid,5:] = np.eye(3)[class_label]
    
    print(label)
    return image_normalized, label

def yolo_loss(y_true, y_pred):
    # y_true: ラベルデータ (batch_size, S, S, B*5 + C)
    # y_pred: 予測データ (batch_size, S, S, B*5 + C)

    # オブジェクトが存在するマスク
    object_mask = y_true[..., 4:5]

    # バウンディングボックスの損失
    bbox_true = y_true[..., :1*5]
    bbox_pred = y_pred[..., :1*5]
    bbox_loss = tf.reduce_sum(object_mask * tf.square(bbox_true - bbox_pred))

    # クラス確率の損失
    class_true = y_true[..., 1*5:]
    class_pred = y_pred[..., 1*5:]
    class_loss = tf.reduce_sum(object_mask * tf.square(class_true - class_pred))

    # 信頼度スコアの損失
    confidence_true = y_true[..., 4:5]
    confidence_pred = y_pred[..., 4:5]
    confidence_loss_obj = tf.reduce_sum(object_mask * tf.square(confidence_true - confidence_pred))
    confidence_loss_noobj = tf.reduce_sum((1 - object_mask) * tf.square(confidence_true - confidence_pred))
    confidence_loss = confidence_loss_obj + 0.5 * confidence_loss_noobj  # 非オブジェクトの損失に0.5の重みをかける

    # 全体の損失
    total_loss = bbox_loss + class_loss + confidence_loss
    return total_loss


###----------------------------------ここからが実際の処理---------------------------------###

# CSVファイルの読み込み
csv_path = 'Annotations/Annotations/dayTrain/dayClip1/annotations_1.csv'  # CSVファイルのパスを指定
df = pd.read_csv(csv_path)

# クラスのマッピング
class_mapping = {'stop': 0, 'go': 1, 'yield': 2}  # クラス名とクラスIDのマッピング

# モデルのコンパイルと学習
input_shape = (448, 448, 3)  # リサイズ後のサイズを指定
num_classes = 3  # クラス数（例: stop, go, yield）

# データの前処理
images = []
labels = []
for index, row in df.iterrows():
    print(row)
    image_path = os.path.join('dayTrain/', row['Filename'])
    box = [row['Upper left corner X'], row['Upper left corner Y'], row['Lower right corner X'], row['Lower right corner Y']]
    class_label = class_mapping[row['Annotation tag']]
    image, label = preprocess_data(image_path, box, class_label, input_shape)
    images.append(image)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

model = create_yolo_model(input_shape, num_classes)
model.compile(optimizer='adam', loss=yolo_loss)
model.load_weights('./checkpoints/yolo_checkpoint.weights.h5')

model.fit(images, labels, epochs=100)

# モデルの保存
model.save_weights('./checkpoints/yolo_checkpoint.weights.h5')
