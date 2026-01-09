# ==================== CONFIG.PY ====================
"""
Configuration file for Mask & Crowd Monitoring System
"""

# Model paths
MASK_MODEL_PATH = 'models/mask_detection_model.pth'
YOLO_MODEL_PATH = 'models/yolov11n.pt'

# Mask detection settings (classes in alphabetical order - matching dataset folders)
# Dataset folders: cloth, n95, n95v, nfm, srg
MASK_CLASSES = ['Cloth Mask', 'N95 Mask', 'Partial Mask', 'No Mask', 'Surgical Mask']
MASK_TARGET_SIZE = (128, 128)

# Risk percentage mapping
VIRUS_RISK_MAP = {
    'N95 Mask': 5,
    'Surgical Mask': 15,
    'Cloth Mask': 40,
    'Partial Mask': 65,
    'No Mask': 95
}

# Color mapping for boxes
COLOR_MAP = {
    'safe': (0, 255, 0),      # Green - N95, Surgical
    'warning': (0, 165, 255),  # Orange - Cloth
    'danger': (0, 0, 255)      # Red - Partial, No Mask
}

# Crowd monitoring settings
CROWD_THRESHOLD = 20
YOLO_CONFIDENCE = 0.5
PERSON_CLASS_ID = 0  # COCO person class

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 30

# Alert settings
ALERT_TEXT = "Warning! Excessive crowd detected. Please disperse and maintain social distancing."
ALERT_VOLUME = 0.9
ALERT_RATE = 150

# Face detection
FACE_CASCADE_PATH = None  # Uses OpenCV default
FACE_SCALE_FACTOR = 1.3
FACE_MIN_NEIGHBORS = 5
