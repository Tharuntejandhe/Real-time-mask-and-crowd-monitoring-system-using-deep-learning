# PROJECT DIRECTORY STRUCTURE

```
MASK_CROWD_MONITORING_SYSTEM/
│
├── 📄 app.py                          ← MAIN APPLICATION (1000+ lines)
│   └── Complete Streamlit interface with:
│       ├── Home page
│       ├── Mask Detection module
│       ├── Crowd Monitoring module
│       ├── Real-time processing
│       └── Professional UI/CSS
│
├── 📄 requirements.txt                ← PYTHON DEPENDENCIES
│   └── All packages pre-configured
│
├── 📄 config.py                       ← CONFIGURATION FILE
│   └── Settings for:
│       ├── Model paths
│       ├── Detection parameters
│       ├── Alert settings
│       └── Camera configuration
│
├── 📄 mask_detector.py                ← MASK DETECTION MODULE
│   └── MaskDetector class with:
│       ├── Face detection
│       ├── Mask classification
│       └── Risk calculation
│
├── 📄 crowd_detector.py               ← CROWD MONITORING MODULE
│   └── CrowdDetector class with:
│       ├── Person detection (YOLO)
│       ├── Crowd counting
│       └── Statistics tracking
│
├── 📄 audio_alert.py                  ← AUDIO ALERT SYSTEM
│   └── AudioAlert class with:
│       ├── Text-to-speech
│       ├── Threading support
│       └── Alert management
│
├── 📄 train_mask_model.py             ← TRAINING SCRIPT
│   └── MaskDetectionModelTrainer class:
│       ├── CNN model creation
│       ├── Data augmentation
│       ├── Training pipeline
│       └── Visualization
│
├── 📄 README.md                       ← FULL DOCUMENTATION (500+ lines)
│   ├── Features overview
│   ├── Installation guide
│   ├── Usage instructions
│   ├── Training guide
│   ├── API reference
│   ├── Troubleshooting
│   └── Best practices
│
├── 📄 QUICKSTART.md                   ← QUICK SETUP (200+ lines)
│   ├── 5-minute setup
│   ├── Dataset preparation
│   ├── Configuration
│   ├── Common issues
│   └── Deployment options
│
├── 📁 models/
│   ├── mask_detection_model.h5        (You need to add after training)
│   └── yolov11n.pt                    (Auto-downloads on first run)
│
├── 📁 data/
│   └── train/                         (Your training data here)
│       ├── N95 Mask/
│       ├── Surgical Mask/
│       ├── Cloth Mask/
│       ├── Partial Mask/
│       └── No Mask/
│
├── 📁 logs/
│   └── (TensorBoard logs generated during training)
│
└── 📁 utils/
    └── (Additional utility files can go here)
```

---

## 📋 FILE DESCRIPTIONS

### Core Application Files

#### `app.py` (Main Application - 1000+ lines)
**Purpose:** Complete Streamlit web application
**Features:**
- 3-mode navigation system
- Real-time mask detection
- Crowd monitoring (webcam & video)
- Statistics dashboard
- Professional UI with custom CSS

**Key Components:**
```python
load_mask_model()              # Load CNN model
load_yolo_model()              # Load YOLO model
detect_masks_realtime()        # Process faces for masks
detect_crowd()                 # Count people in frame
play_alert_sound()             # Audio alerts
calculate_virus_risk_percentage()  # Risk calculation
```

#### `requirements.txt` (Dependencies)
**All packages needed:**
- streamlit==1.36.0
- tensorflow==2.15.0
- opencv-python==4.8.1.78
- ultralytics==8.1.0 (YOLO)
- pyttsx3==2.90 (Audio)
- And more...

#### `config.py` (Configuration)
**Centralized settings:**
- Model paths
- Detection thresholds
- Camera settings
- Color schemes
- Alert parameters

---

### Utility Modules

#### `mask_detector.py` (Mask Detection)
**MaskDetector class:**
```python
class MaskDetector:
    - preprocess_image()        # Image preprocessing
    - detect_faces()           # Face detection
    - classify_mask()          # 5-class classification
    - get_risk_percentage()    # Risk calculation
    - get_box_color()          # Color based on risk
    - detect_and_classify()    # Main function
```

#### `crowd_detector.py` (Crowd Monitoring)
**CrowdDetector class:**
```python
class CrowdDetector:
    - detect_persons()         # YOLO person detection
    - get_statistics()         # Detection statistics
    - reset()                  # Reset history
```

#### `audio_alert.py` (Audio System)
**AudioAlert class:**
```python
class AudioAlert:
    - play_alert()             # Play audio alert
    - stop_alert()             # Stop alert
    - set_voice()              # Configure voice
```

---

### Training & Documentation

#### `train_mask_model.py` (Training Script - 300+ lines)
**MaskDetectionModelTrainer class:**
```python
- create_model()           # Build CNN architecture
- create_data_generators() # Data augmentation
- train()                  # Training loop
- save_model()            # Save trained model
- plot_training_history() # Visualize results
```

**Usage:**
```bash
python train_mask_model.py \
    --data_dir data/train \
    --epochs 50 \
    --batch_size 32
```

#### `README.md` (Full Documentation - 500+ lines)
- Complete feature overview
- Detailed installation instructions
- Usage guide for both modules
- Training guide with examples
- API reference
- Troubleshooting section
- Performance benchmarks

#### `QUICKSTART.md` (Quick Start - 200+ lines)
- 5-minute setup guide
- Dataset preparation
- Configuration guide
- Common issues & solutions
- Deployment options

---

## 🗂️ Directory Structure Setup

### Create Required Directories
```bash
# From project root
mkdir -p models data/train logs utils

# Create class subdirectories for training data
mkdir -p data/train/{N95\ Mask,Surgical\ Mask,Cloth\ Mask,Partial\ Mask,No\ Mask}
```

### Directory Purposes

**models/**
- Store trained CNN model
- Store YOLO model (auto-downloads)
- Store model checkpoints

**data/**
- training data organized by class
- validation data (auto-split)
- test datasets

**logs/**
- TensorBoard logs
- Training history
- Performance metrics

**utils/**
- Additional utility files
- Helper functions
- Custom modules

---

## 📦 Installation Workflow

```
1. Clone/Download Project
   ↓
2. pip install -r requirements.txt
   ↓
3. mkdir -p models data logs
   ↓
4. Prepare mask detection data (optional)
   ↓
5. python train_mask_model.py (optional)
   ↓
6. streamlit run app.py
   ↓
7. Access at http://localhost:8501
```

---

## 🚀 Quick Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p models data logs utils

# Create training subdirectories
mkdir -p data/train/{N95\ Mask,Surgical\ Mask,Cloth\ Mask,Partial\ Mask,No\ Mask}
```

### Development
```bash
# Train mask detection model
python train_mask_model.py --data_dir data/train --epochs 50

# Run application
streamlit run app.py

# Run with custom port
streamlit run app.py --server.port 8502

# Run on network
streamlit run app.py --server.address 0.0.0.0
```

### Debugging
```bash
# Check camera
python -c "import cv2; print('OK' if cv2.VideoCapture(0).isOpened() else 'FAIL')"

# Check TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Check YOLO
python -c "from ultralytics import YOLO; print('YOLO OK')"

# Check TTS
python -c "import pyttsx3; e=pyttsx3.init(); e.say('Test'); e.runAndWait()"
```

---

## 📊 File Statistics

| File | Type | Lines | Purpose |
|------|------|-------|---------|
| app.py | Python | 1000+ | Main application |
| train_mask_model.py | Python | 300+ | Training script |
| mask_detector.py | Python | 150+ | Mask detection |
| crowd_detector.py | Python | 130+ | Crowd monitoring |
| audio_alert.py | Python | 80+ | Audio alerts |
| config.py | Python | 50+ | Configuration |
| README.md | Markdown | 500+ | Full documentation |
| QUICKSTART.md | Markdown | 200+ | Quick start guide |

**Total Code: 2000+ lines**
**Total Documentation: 800+ lines**

---

## 🎯 Feature Matrix

| Feature | Module | File | Status |
|---------|--------|------|--------|
| Streamlit UI | App | app.py | ✅ |
| Mask Detection | Mask | app.py, mask_detector.py | ✅ |
| 5-class Classification | Mask | mask_detector.py | ✅ |
| Risk Percentage | Mask | app.py, config.py | ✅ |
| Real-time Camera | Mask | app.py | ✅ |
| YOLO Detection | Crowd | app.py, crowd_detector.py | ✅ |
| Person Counting | Crowd | crowd_detector.py | ✅ |
| Audio Alerts | Crowd | audio_alert.py, app.py | ✅ |
| Video Upload | Crowd | app.py | ✅ |
| Statistics | Both | app.py, detectors | ✅ |
| Training Pipeline | Model | train_mask_model.py | ✅ |
| Configuration | System | config.py | ✅ |

---

## 🔗 File Dependencies

```
app.py (Main)
├── config.py (Settings)
├── tensorflow (Mask loading)
├── ultralytics (YOLO)
├── cv2 (OpenCV)
├── pyttsx3 (Audio)
└── streamlit (UI)

train_mask_model.py
├── tensorflow
├── cv2
└── numpy

mask_detector.py
├── cv2
├── numpy
└── config.py

crowd_detector.py
├── cv2
├── ultralytics
└── config.py

audio_alert.py
├── pyttsx3
├── threading
└── config.py
```

---

## 📝 Summary

**Total Project Files: 9**
- 6 Python files (2000+ lines)
- 3 Documentation files (800+ lines)

**Key Modules:**
- Mask Detection with 5 classes
- Crowd Monitoring with YOLO
- Training Pipeline
- Audio Alert System
- Configuration Management

**Ready to Use:**
✅ All code complete
✅ Fully documented
✅ Production-ready
✅ Easy to customize

**Start with:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
