import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.layers import LeakyReLU
import os
import math
from model import yolo_v1_model
from loss import YoloLoss
from preprocess import preprocess_data

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

model = yolo_v1_model()
loss_fn = YoloLoss()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

epochs = 30
batch_size = 1
dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch_train, training=True)

            # 損失計算の数値チェック
            loss_value = loss_fn(y_batch_train, y_pred)
            tf.debugging.check_numerics(loss_value, "Loss contains NaN or Inf")

        grads = tape.gradient(loss_value, model.trainable_variables)

        # 勾配の数値チェック
        for grad in grads:
            tf.debugging.check_numerics(grad, "Gradient contains NaN or Inf")

        # 勾配クリッピング
        clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        
        optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))

        if step % 10 == 0:
            print(f'Step {step}, Loss: {loss_value.numpy()}')

# モデルの保存
model.save_weights('./checkpoint3/yolo_checkpoint.weights.h5')
