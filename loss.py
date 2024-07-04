import tensorflow as tf
from tensorflow.keras.losses import Loss

def compute_iou(true_boxes, pred_boxes):
    true_x1 = true_boxes[..., 0] - true_boxes[..., 2] / 2
    true_y1 = true_boxes[..., 1] - true_boxes[..., 3] / 2
    true_x2 = true_boxes[..., 0] + true_boxes[..., 2] / 2
    true_y2 = true_boxes[..., 1] + true_boxes[..., 3] / 2

    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

    xi1 = tf.maximum(true_x1, pred_x1)
    yi1 = tf.maximum(true_y1, pred_y1)
    xi2 = tf.minimum(true_x2, pred_x2)
    yi2 = tf.minimum(true_y2, pred_y2)

    inter_area = tf.maximum(xi2 - xi1, 0) * tf.maximum(yi2 - yi1, 0)

    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

    union_area = true_area + pred_area - inter_area

    iou = inter_area / (union_area + tf.keras.backend.epsilon())
    return iou

class YoloLoss(Loss):
    def __init__(self, S=7, B=1, C=3, **kwargs):
        super(YoloLoss, self).__init__(**kwargs)
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 10
        self.lambda_noobj = 0.1


    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, (-1, self.S, self.S, self.C + 5 * self.B))
        y_pred = tf.reshape(y_pred, (-1, self.S, self.S, self.C + 5 * self.B))

        true_box_confidence = y_true[..., 4:5]
        pred_box_confidence = y_pred[..., 4:5]

        true_boxes = y_true[..., :4]
        pred_boxes = y_pred[..., :4]

        true_classes = y_true[..., 5:]
        pred_classes = y_pred[..., 5:]

        mask_obj = true_box_confidence
        mask_noobj = 1 - mask_obj

        epsilon = 1e-7

        xy_diff = true_boxes[..., 0:2] - pred_boxes[..., 0:2]

        sqrt_true_wh = tf.sqrt(tf.maximum(true_boxes[..., 2:4], epsilon))
        sqrt_pred_wh = tf.sqrt(tf.maximum(pred_boxes[..., 2:4], epsilon))
        wh_diff = sqrt_true_wh - sqrt_pred_wh

        iou = compute_iou(true_boxes, pred_boxes)
        iou = tf.expand_dims(iou, axis=-1)

        loss_xy = self.lambda_coord * tf.reduce_sum(mask_obj * tf.square(xy_diff), axis=[1, 2, 3])
        loss_wh = self.lambda_coord * tf.reduce_sum(mask_obj * tf.square(wh_diff), axis=[1, 2, 3])

        loss_conf_obj = tf.reduce_sum(mask_obj * tf.square(iou-1), axis=[1, 2, 3])
        loss_conf_noobj = self.lambda_noobj * tf.reduce_sum(mask_noobj * tf.square(true_box_confidence - pred_box_confidence), axis=[1, 2, 3])

        loss_class = tf.reduce_sum(mask_obj * tf.square(true_classes - pred_classes), axis=[1, 2, 3])

        total_loss = loss_xy + loss_wh + loss_conf_obj + loss_conf_noobj + loss_class
        return tf.reduce_mean(total_loss)
