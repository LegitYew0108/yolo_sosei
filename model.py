import tensorflow as tf
from tensorflow.keras import layers, Model

def yolo_v1_model(input_shape=(448, 448, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    # Layer 1: Conv
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)

    # Layer 2: Conv
    x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)

    # Layer 3: Conv
    x = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)

    x = layers.Conv2D(512, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(512, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)

    x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(1024, (3, 3), strides=2, padding='same', activation='relu')(x)

    # Layer 11-12: Conv
    x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(7 * 7 * (num_classes + 5), activation='linear')(x)
    x = layers.Reshape((7, 7, num_classes + 5))(x)

    # Apply sigmoid to x_center, y_center, confidence, and class probabilities
    x_center_y_center = layers.Activation('sigmoid')(x[..., :2])
    confidence = layers.Activation('sigmoid')(x[..., 4:5])
    class_probs = layers.Activation('sigmoid')(x[..., 5:])

    outputs = layers.Concatenate()([x_center_y_center, x[..., 2:4], confidence, class_probs])

    model = Model(inputs, outputs)
    return model
