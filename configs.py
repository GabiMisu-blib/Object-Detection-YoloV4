# Tipul modelului YOLO utilizat (YOLOv4)
YOLO_TYPE                   = "yolov4" 

# Framework-ul folosit pentru detecție ("tf" pentru TensorFlow sau "trt" pentru TensorRT)
YOLO_FRAMEWORK              = "tf" # "tf" or "trt"

# Calea către fișierul cu greutățile modelului YOLOv4
YOLO_V4_WEIGHTS             = "model_data/yolov4.weights"

# Calea către fișierul cu numele claselor COCO
YOLO_COCO_CLASSES           = "model_data/coco/coco.names"

# Pașii YOLO pentru fiecare scară
YOLO_STRIDES                = [8, 16, 32]

# Pragul de pierdere pentru IoU (Intersection over Union)
YOLO_IOU_LOSS_THRESH        = 0.5

# Numărul de ancore per scară
YOLO_ANCHOR_PER_SCALE       = 3

# Numărul maxim de casete de delimitare per scară
YOLO_MAX_BBOX_PER_SCALE     = 100

# Dimensiunea de intrare pentru modelul YOLO
YOLO_INPUT_SIZE             = 416

# Dacă tipul modelului este "yolov4", definește ancorele utilizate pentru fiecare scară
if YOLO_TYPE                == "yolov4":
    YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]
