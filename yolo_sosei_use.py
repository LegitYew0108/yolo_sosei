import cv2
import numpy as np
import tensorflow as tf
import os
from model import yolo_v1_model
from loss import YoloLoss
from preprocess import preprocess_data

# クラスのマッピング
class_mapping = {'stop': 0, 'go': 1, 'yield': 2}  # クラス名とクラスIDのマッピング

# モデルのロード
input_shape = (448, 448, 3)  # リサイズ後のサイズを指定
num_classes = 3  # クラス数（例: stop, go, yield）

model = yolo_v1_model()
model.load_weights('./checkpoint3/yolo_checkpoint.weights.h5')

# 保存先フォルダの作成
os.makedirs('result', exist_ok=True)

def draw_boxes(image, predictions, threshold=0.02):
    height, width, _ = image.shape
    S = 7  # グリッドサイズ
    B = 1  # バウンディングボックスの数
    C = 3  # クラスの数

    for i in range(S):
        for j in range(S):
            cell_predictions = predictions[i, j]
            confidence = cell_predictions[4]
            if confidence > threshold:
                # バウンディングボックスの座標
                x_center = cell_predictions[0] * width
                y_center = cell_predictions[1] * height
                box_width = cell_predictions[2] * width
                box_height = cell_predictions[3] * height
                x1 = int(x_center - (box_width / 2))
                y1 = int(y_center - (box_height / 2))
                x2 = int(x_center + (box_width / 2))
                y2 = int(y_center + (box_height / 2))

                # 座標が画像の範囲内に収まるようにクリッピング
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                # クラスのスコア
                class_scores = cell_predictions[5:]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]

                # バウンディングボックスを描画
                if class_id == 0:
                    color = (0, 0, 255)
                elif class_id == 1:
                    color = (0, 255, 0)
                elif class_id == 2:
                    color = (0, 255, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"Class {class_id} ({class_score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# 指定された範囲の画像に対して予測と描画を行う
for i in range(1, 121):
    image_path = f'dayTrain/dayTraining/dayClip1--{i:05d}.jpg'
    result_path = f'result/dayClip1--{i:05d}.png'
    
    if not os.path.exists(image_path):
        continue

    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))
    image_normalized = image_resized / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)  # モデルに入力するためにバッチ次元を追加

    predictions = model.predict(image_batch)
    predictions = predictions[0]  # バッチ次元を削除

    image_with_boxes = draw_boxes(image, predictions)
    cv2.imwrite(result_path, image_with_boxes)
    print(f"Saved: {result_path}")
