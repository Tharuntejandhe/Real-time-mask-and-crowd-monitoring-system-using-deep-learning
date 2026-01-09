# SETUP & QUICK START GUIDE

## 🎯 Quick Setup (5 minutes)

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Directories
```bash
mkdir -p models data logs utils
```

### 3. Run Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📊 Dataset Preparation Guide

### Option 1: Download Pre-trained Model

#### From Kaggle
1. Go to: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
2. Download the dataset
3. Extract and organize in `data/train/` directory
4. Directory structure should be:
   ```
   data/train/
   ├── N95 Mask/          (images here)
   ├── Surgical Mask/     (images here)
   ├── Cloth Mask/        (images here)
   ├── Partial Mask/      (images here)
   └── No Mask/           (images here)
   ```

#### From GitHub
1. Clone MaskedFace-Net: `https://github.com/cabani/MaskedFace-Net`
2. Download COCO-Mask dataset
3. Organize into 5 classes as above

### Option 2: Create Your Own Dataset

#### Data Collection Steps
1. **Collect Images:**
   - N95 Mask: 200+ clear N95 mask images
   - Surgical Mask: 200+ surgical mask images
   - Cloth Mask: 200+ cloth mask images
   - Partial Mask: 200+ partially worn mask images
   - No Mask: 200+ unmasked face images

2. **Image Requirements:**
   - Minimum resolution: 128x128 pixels
   - Various lighting conditions
   - Different angles and distances
   - Include diverse ethnicities and demographics

3. **Data Augmentation:**
   The training script includes automatic augmentation

---

## 🚀 Training Your Model

### Step 1: Prepare Dataset
```
data/train/
├── N95 Mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Surgical Mask/
│   └── ...
└── ... (other classes)
```

### Step 2: Run Training Script
```bash
python train_mask_model.py \
    --data_dir data/train \
    --epochs 50 \
    --batch_size 32 \
    --save_path models/mask_detection_model.h5
```

### Step 3: Monitor Training
- Training logs in `logs/` directory
- Best model saved in `models/`
- Training history plot saved as `training_history.png`

### Training Tips
- Use GPU for faster training: `CUDA_VISIBLE_DEVICES=0 python train_mask_model.py`
- Start with 50 epochs, increase if validation accuracy not plateaued
- Target accuracy: 85%+ on validation set
- Minimum 100 images per class

---

## 🔧 Configuration Guide

### Edit `config.py` for customization:

```python
# Crowd threshold (trigger alert when people > this value)
CROWD_THRESHOLD = 20

# YOLO detection confidence (0-1, lower = more detections)
YOLO_CONFIDENCE = 0.5

# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Mask detection preprocessing size
MASK_TARGET_SIZE = (128, 128)

# Face detection sensitivity
FACE_SCALE_FACTOR = 1.3        # Lower = more sensitive
FACE_MIN_NEIGHBORS = 5          # Higher = fewer false positives

# Audio alert settings
ALERT_VOLUME = 0.9              # 0-1
ALERT_RATE = 150                # Speech speed (words per minute)
```

---

## 🎭 Using Mask Detection

### Real-time Camera Detection
1. Select "🎭 Mask Detection" from sidebar
2. Click "Start Detection" checkbox
3. Grant camera access
4. System displays:
   - Bounding boxes around detected faces
   - Mask type classification
   - Confidence score
   - Risk percentage

### Understanding Risk Percentages
- **5% (N95):** Excellent protection
- **15% (Surgical):** Very good protection
- **40% (Cloth):** Fair protection
- **65% (Partial):** Poor protection
- **95% (No Mask):** No protection

### Statistics
- See detection counts for each mask type
- View percentage of each class
- Real-time metrics update

---

## 👥 Using Crowd Monitoring

### Webcam Mode
1. Select "👥 Crowd Monitoring" from sidebar
2. Select "📹 Webcam" option
3. Set crowd threshold (default: 20)
4. Click "Start Crowd Detection"
5. System counts people in real-time
6. Audio alert plays when threshold exceeded

### Video Upload Mode
1. Select "👥 Crowd Monitoring"
2. Select "📹 Upload Video"
3. Choose video file (.mp4, .avi, .mov, .mkv)
4. Set crowd threshold
5. Click "Process Video"
6. View results:
   - Max people detected
   - Number of alert triggers
   - Video frame-by-frame analysis

### Video Format Support
- **MP4** ✅ (Recommended)
- **AVI** ✅
- **MOV** ✅
- **MKV** ✅

---

## 🎮 Control Panel

### Camera Settings
- Resolution: 640x480 (can modify in config.py)
- FPS: 30 (automatic)
- Format: BGR (OpenCV standard)

### Alert Settings
- Voice: System default
- Volume: 90%
- Speed: 150 WPM
- Language: English

### Detection Settings
- Mask classes: 5 (configurable in training)
- Person threshold: Adjustable per session
- Confidence: 0.5 (YOLO)

---

## 🔍 Debugging Tips

### Camera Issues
```bash
# Test camera access
python -c "import cv2; cap=cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAILED')"
```

### Model Loading Issues
```bash
# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Check YOLO installation
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

### Audio Issues
```bash
# Test TTS
python -c "import pyttsx3; e=pyttsx3.init(); e.say('Test'); e.runAndWait()"
```

### GPU Support
```bash
# Check CUDA availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Use GPU in training
CUDA_VISIBLE_DEVICES=0 python train_mask_model.py
```

---

## 📈 Performance Optimization

### For Better Speed
1. Use GPU acceleration
2. Reduce camera resolution to 480p
3. Use YOLOv11 nano (already in config)
4. Increase frame skip (every 2nd frame)

### For Better Accuracy
1. Train on more diverse data
2. Use higher resolution (256x256)
3. Increase epochs during training
4. Use YOLOv11 small model

### For Production Deployment
1. Use Docker containerization
2. Set up monitoring and logging
3. Add database storage for alerts
4. Implement multi-threading
5. Use load balancing

---

## 📱 Deployment Options

### Local Machine
```bash
streamlit run app.py
```

### Remote Server
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect at https://share.streamlit.io
3. Deploy from repository

### Docker Deployment
```bash
# Create Dockerfile
docker build -t mask-crowd-monitor .
docker run -p 8501:8501 mask-crowd-monitor
```

---

## 🚨 Important Notes

### Privacy & Legal
- ⚠️ Ensure compliance with GDPR/CCPA
- ⚠️ Get consent before monitoring
- ⚠️ Secure video storage
- ⚠️ Limited data retention

### Limitations
- Mask detection needs clear face visibility
- Performance depends on lighting
- YOLO may struggle with overlapping people
- Requires adequate computational resources

### Best Practices
✅ Regular model retraining
✅ Diverse training data
✅ Multiple camera angles
✅ Audit trails for alerts
✅ Regular accuracy testing

---

## 📞 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Camera not found | Check permissions, try different camera index |
| Mask model not loading | Train model or download pre-trained version |
| Slow detection | Use GPU, reduce resolution, use nano YOLO |
| Audio not working | Check volume, install pyttsx3, check speakers |
| Out of memory | Reduce batch size, use smaller model |
| Low accuracy | Train with more diverse data, increase epochs |

---

## 🎓 Next Steps

1. ✅ Complete setup above
2. 📚 Read main README.md for full documentation
3. 🎯 Collect/download mask detection dataset
4. 🚀 Train your mask detection model
5. 🎭 Test both modules
6. 🚢 Deploy to your environment

---

## 📚 Resources

- **YOLO Docs:** https://docs.ultralytics.com
- **TensorFlow:** https://www.tensorflow.org
- **Streamlit:** https://docs.streamlit.io
- **OpenCV:** https://docs.opencv.org

---

**Ready to start? Begin with `pip install -r requirements.txt` and then `streamlit run app.py`** 🚀
