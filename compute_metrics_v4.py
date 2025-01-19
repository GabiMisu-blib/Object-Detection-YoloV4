import tensorflow as tf
import json
import matplotlib.pyplot as plt
import os
from yolov4.configs import *
from yolov4.yolov4 import *
import numpy as np
import cv2
import seaborn as sns
import csv

# Încarcă numele claselor dintr-un fișier
def load_class_names(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

class_names = load_class_names("model_data/coco/coco.names")
n_classes = len(class_names)

# Încarcă informațiile din fișierul JSON pentru setul de date COCO
with open('instances_train2017.json', 'r') as f:
    info = json.load(f)

# Încarcă greutățile YOLO
def load_yolo_weights(model, weights_file):
    tf.keras.backend.clear_session()  # Resetează numele straturilor
    range1 = 110
    range2 = [93, 101, 109]
    
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(range1):
            if i > 0:
                conv_layer_name = 'conv2d_%d' %i
            else:
                conv_layer_name = 'conv2d'
                
            if j > 0:
                bn_layer_name = 'batch_normalization_%d' %j
            else:
                bn_layer_name = 'batch_normalization'
            
            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in range2:
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in range2:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'

# Încarcă modelul YOLO
def Load_Yolo_model():
    if YOLO_FRAMEWORK == "tf":  # Detecție cu TensorFlow
        Darknet_weights = YOLO_V4_WEIGHTS
        print("Loading Darknet_weights from:", Darknet_weights)
        yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
        load_yolo_weights(yolo, Darknet_weights)
    return yolo

# Preprocesare imagine pentru modelul YOLO
def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

# Calculul IoU (Intersection over Union) pentru casete de delimitare
def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

# Algoritm NMS (Non-Maximum Suppression)
def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

# Post-procesare casete de delimitare
def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

Yolo = Load_Yolo_model()

# Aplica modelul YOLOv4 pe o imagine
def apply_yolov4(image_path: str):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [YOLO_INPUT_SIZE, YOLO_INPUT_SIZE])
    image_data = image_data[np.newinstance, ...].astype(np.float32)

    pred_bbox = Yolo.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, original_image, YOLO_INPUT_SIZE, 0.3)
    bboxes = nms(bboxes, 0.45, method='nms')

    detected_categories = set()
    for bbox in bboxes:
        detected_categories.add(bbox[5]+1)

    return detected_categories

output_labels = []
true_output_labels = []
folder_path = 'train2017'

# Procesare imagini din folderul specificat
files = os.listdir(folder_path)
for i in range(int(1 * len(files))):
    print(i)
    filename = files[i]
    detected_categories = apply_yolov4(os.path.join(folder_path, filename))
    output_labels.append(detected_categories)

    aux = os.path.splitext(filename)[0]
    file_id = int(aux)

    true_categories = set()
    for annotation in info['annotations']:
        if annotation['image_id'] == file_id:
            true_categories.add(annotation['category_id'])
    true_output_labels.append(true_categories)
    print(i)
    
tp = {_class: 0 for _class in range(n_classes)}
tn = {_class: 0 for _class in range(n_classes)}
fp = {_class: 0 for _class in range(n_classes)}
fn = {_class: 0 for _class in range(n_classes)}

y_true = []
y_pred = []

# Calculare metrici pentru fiecare clasă
for label, true_label in zip(output_labels, true_output_labels):
    print(label)
    print(true_label)
    print()
    for _class in range(n_classes):
        if _class in label and _class in true_label:
            tp[_class] += 1
            y_true.append(_class)
            y_pred.append(_class)
        if _class not in label and _class in true_label:
            fn[_class] += 1
            y_true.append(_class)
            y_pred.append(-1)  # False negative
        if _class in label and _class not in true_label:
            fp[_class] += 1
            y_true.append(-1)  # False positive
            y_pred.append(_class)
        if _class not in label and _class not in true_label:
            tn[_class] += 1

accuracy = {}
precision = {}
recall = {}
f1 = {}

# Calculare metrici (acuratețe, precizie, recall, F1)
for _class in range(n_classes):
    accuracy[_class] = tp[_class] / (tp[_class] + tn[_class] + fp[_class] + fn[_class]) if tp[_class] + tn[_class] + fp[_class] + fn[_class] != 0 else 0
    precision[_class] = tp[_class] / (tp[_class] + fp[_class]) if tp[_class] + fp[_class] != 0 else 0
    recall[_class] = tp[_class] / (tp[_class] + fn[_class]) if tp[_class] + fn[_class] != 0 else 0
    f1[_class] = 2 * precision[_class] * recall[_class] / (precision[_class] + recall[_class]) if precision[_class] + recall[_class] != 0 else 0

# Plot metrici pentru toate clasele
fig, axs = plt.subplots(2, 2, figsize=(20, 15))

# Acuratețe
classes = list(range(n_classes))
axs[0, 0].bar(classes, [accuracy.get(i, 0) for i in classes], color='b')
axs[0, 0].set_title('Accuracy per Class')
axs[0, 0].set_xlabel('Class')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].set_xticks(classes)
axs[0, 0].set_xticklabels(classes, rotation=90)

# Precizie
axs[0, 1].bar(classes, [precision.get(i, 0) for i in classes], color='g')
axs[0, 1].set_title('Precision per Class')
axs[0, 1].set_xlabel('Class')
axs[0, 1].set_ylabel('Precision')
axs[0, 1].set_xticks(classes)
axs[0, 1].set_xticklabels(classes, rotation=90)

# Recall
axs[1, 0].bar(classes, [recall.get(i, 0) for i in classes], color='r')
axs[1, 0].set_title('Recall per Class')
axs[1, 0].set_xlabel('Class')
axs[1, 0].set_ylabel('Recall')
axs[1, 0].set_xticks(classes)
axs[1, 0].set_xticklabels(classes, rotation=90)

# F1 Score
axs[1, 1].bar(classes, [f1.get(i, 0) for i in classes], color='purple')
axs[1, 1].set_title('F1 Score per Class')
axs[1, 1].set_xlabel('Class')
axs[1, 1].set_ylabel('F1 Score')
axs[1, 1].set_xticks(classes)
axs[1, 1].set_xticklabels(classes, rotation=90)

plt.tight_layout()
plt.show()

# Salvare metrici într-un fișier CSV
with open('metrics.csv', 'w', newline='') as csvfile:
    fieldnames = ['Class', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for _class in range(n_classes):
        writer.writerow({
            'Class': _class,
            'Accuracy': accuracy[_class],
            'Precision': precision[_class],
            'Recall': recall[_class],
            'F1 Score': f1[_class]
        })
