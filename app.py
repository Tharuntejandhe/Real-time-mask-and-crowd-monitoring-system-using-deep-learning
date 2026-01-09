import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import threading
import os
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import pyttsx3
import time

# Page configuration
st.set_page_config(
    page_title="Mask & Crowd Monitoring System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        color: #FF6B6B;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #4ECDC4;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: bold;
    }
    .status-safe { background-color: #d4edda; color: #155724; }
    .status-warning { background-color: #fff3cd; color: #856404; }
    .status-danger { background-color: #f8d7da; color: #721c24; }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">🎭 Real-Time Mask & Crowd Monitoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Computer Vision for Public Safety</div>', unsafe_allow_html=True)

# Initialize session state
if 'alert_playing' not in st.session_state:
    st.session_state.alert_playing = False
if 'crowd_alert_triggered' not in st.session_state:
    st.session_state.crowd_alert_triggered = False

# ==================== UTILITY FUNCTIONS ====================

@st.cache_resource
def load_mask_model():
    """Load pre-trained PyTorch mask detection model"""
    try:
        import torch
        import torch.nn as nn
        
        # Define the same CNN architecture used for training
        class MaskDetectionCNN(nn.Module):
            def __init__(self, num_classes=5):
                super(MaskDetectionCNN, self).__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32), nn.ReLU(),
                    nn.MaxPool2d(2, 2), nn.Dropout(0.25)
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(2, 2), nn.Dropout(0.25)
                )
                self.conv3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128), nn.ReLU(),
                    nn.MaxPool2d(2, 2), nn.Dropout(0.25)
                )
                self.conv4 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128), nn.ReLU(),
                    nn.MaxPool2d(2, 2), nn.Dropout(0.25)
                )
                self.fc = nn.Sequential(
                    nn.Linear(128 * 8 * 8, 512),
                    nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model_path = 'models/mask_detection_model.pth'
        if os.path.exists(model_path):
            # Detect device
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            # Load model
            model = MaskDetectionCNN(num_classes=5)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            return model, device
        else:
            st.warning("⚠️ Mask detection model not found. Please ensure 'mask_detection_model.pth' is in the 'models/' directory")
            return None, None
    except Exception as e:
        st.error(f"Error loading mask model: {e}")
        return None, None

@st.cache_resource
def load_yolo_model():
    """Load YOLOv11 model for crowd detection"""
    try:
        # YOLO will auto-download the model on first use
        model = YOLO('yolo11n.pt')  # YOLOv11 nano model
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.info("💡 The YOLO model will be automatically downloaded on first use. Please ensure you have an internet connection.")
        return None

def play_alert_sound(text="Please disperse and maintain social distancing"):
    """Play audio alert using text-to-speech"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume 0-1
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Audio error: {e}")

def calculate_virus_risk_percentage(mask_type):
    """Calculate virus spreading risk based on mask type"""
    risk_map = {
        'N95 Mask': 5,      # 5% risk
        'Surgical Mask': 15, # 15% risk
        'Cloth Mask': 40,    # 40% risk
        'Partial Mask': 65,  # 65% risk
        'No Mask': 95        # 95% risk
    }
    return risk_map.get(mask_type, 100)

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for PyTorch mask model"""
    try:
        import torch
        from torchvision import transforms
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Define transform
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transform and add batch dimension
        img_tensor = transform(image_rgb).unsqueeze(0)
        return img_tensor
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

def detect_masks_realtime(frame, model, device):
    """Detect masks in frame using PyTorch CNN"""
    if model is None:
        st.warning("Mask model not loaded")
        return frame, []
    
    try:
        import torch
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        detections = []
        # Classes in alphabetical order matching dataset
        mask_classes = ['Cloth Mask', 'N95 Mask', 'Partial Mask', 'No Mask', 'Surgical Mask']
        
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
            
            # Preprocess and predict
            processed_roi = preprocess_image(roi)
            if processed_roi is not None:
                with torch.no_grad():
                    processed_roi = processed_roi.to(device)
                    outputs = model(processed_roi)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, class_idx = torch.max(probabilities, 1)
                    
                    confidence = float(confidence[0])
                    class_idx = int(class_idx[0])
                    mask_class = mask_classes[class_idx]
                
                risk_percentage = calculate_virus_risk_percentage(mask_class)
                
                # Color coding based on risk
                if risk_percentage <= 15:
                    color = (0, 255, 0)  # Green
                elif risk_percentage <= 40:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 0, 255)  # Red
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Put text with mask info
                label = f"{mask_class}: {confidence:.2%}"
                cv2.putText(frame, label, (x, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Put risk percentage
                risk_label = f"Risk: {risk_percentage}%"
                cv2.putText(frame, risk_label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detections.append({
                    'mask_type': mask_class,
                    'confidence': confidence,
                    'risk_percentage': risk_percentage,
                    'bbox': (x, y, w, h)
                })
        
        return frame, detections
    except Exception as e:
        st.error(f"Error in mask detection: {e}")
        return frame, []

def detect_crowd(frame, yolo_model, threshold=20):
    """Detect people in frame using YOLO and count crowd"""
    if yolo_model is None:
        st.warning("YOLO model not loaded")
        return frame, 0, False
    
    try:
        # Run YOLO detection
        results = yolo_model(frame, conf=0.5, verbose=False)
        
        # Count persons (class 0 in COCO)
        person_count = 0
        
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Class 0 is 'person' in COCO
                    person_count += 1
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Alert if crowd exceeds threshold
        alert_triggered = person_count > threshold
        
        # Put text on frame
        status_text = f"People Count: {person_count}"
        status_color = (0, 255, 0) if not alert_triggered else (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        if alert_triggered:
            cv2.putText(frame, "CROWD ALERT!", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame, person_count, alert_triggered
    except Exception as e:
        st.error(f"Error in crowd detection: {e}")
        return frame, 0, False

# ==================== MAIN APP LOGIC ====================

# Sidebar navigation
st.sidebar.markdown("---")
option = st.sidebar.radio(
    "🎯 Select Mode",
    ["🏠 Home", "🎭 Mask Detection", "👥 Crowd Monitoring"],
    label_visibility="collapsed"
)

if option == "🏠 Home":
    # Home Page
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 📋 Features
        
        ### 🎭 Mask Detection Module
        - **Real-time detection** using CNN (5-class classification)
        - **5 Detection Classes:**
          - ✅ N95 Mask
          - ✅ Surgical Mask  
          - ✅ Cloth Mask
          - ⚠️ Partial Mask
          - ❌ No Mask
        - **Virus Risk Percentage** display for each class
        - **Live camera feed** processing
        
        ### 👥 Crowd Monitoring Module
        - **YOLO11 Detection** for person tracking
        - **Real-time counting** of people in frame
        - **Threshold-based alerting** (>20 people)
        - **Audio alerts** with disperse instructions
        - **Video upload support** for CCTV footage
        """)
    
    with col2:
        st.markdown("""
        ## 🚀 How to Use
        
        ### Mask Detection:
        1. Select **Mask Detection** mode
        2. Grant camera access
        3. System will detect faces and masks
        4. View risk percentages in real-time
        5. Analytics shown below video feed
        
        ### Crowd Monitoring:
        1. Select **Crowd Monitoring** mode
        2. Choose between camera or video upload
        3. Set crowd threshold (default: 20)
        4. System counts people and alerts
        5. Audio alert triggers on threshold
        
        ## ⚙️ Requirements
        - Python 3.8+
        - Webcam or video file
        - Modern browser
        - Audio output for alerts
        """)
    
    st.markdown("---")
    st.markdown("""
    ## 📊 Risk Assessment Scale
    """)
    
    risk_cols = st.columns(5)
    risk_data = [
        ("N95 Mask", 5, "🟢"),
        ("Surgical", 15, "🟢"),
        ("Cloth", 40, "🟡"),
        ("Partial", 65, "🔴"),
        ("No Mask", 95, "🔴")
    ]
    
    for col, (mask, risk, emoji) in zip(risk_cols, risk_data):
        with col:
            st.metric(f"{emoji} {mask}", f"{risk}%")

elif option == "🎭 Mask Detection":
    st.subheader("🎭 Real-Time Mask Detection")
    
    # Load PyTorch model
    mask_model, device = load_mask_model()
    
    if mask_model is None:
        st.error("❌ Mask detection model not found!")
        st.info("""
        **To train the mask detection model:**
        1. Collect or download mask dataset (5 classes)
        2. Use the provided training script
        3. Save as `models/mask_detection_model.pth`
        
        **Dataset sources:**
        - Kaggle: Face Mask Detection Dataset
        - GitHub: MaskedFace-Net Dataset
        """)
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📹 Camera Feed")
            run = st.checkbox("Start Detection", value=False)
            
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            if run:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                detection_stats = defaultdict(int)
                frame_count = 0
                
                while run and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access camera")
                        break
                    
                    # Flip frame for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    processed_frame, detections = detect_masks_realtime(frame, mask_model, device)
                    
                    # Update statistics
                    for det in detections:
                        detection_stats[det['mask_type']] += 1
                    
                    frame_count += 1
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, use_column_width=True)
                    
                    # Show current detections
                    if detections:
                        det_info = "**Current Detections:**\n"
                        for det in detections:
                            det_info += f"- {det['mask_type']}: {det['confidence']:.1%} (Risk: {det['risk_percentage']}%)\n"
                        stats_placeholder.markdown(det_info)
                    
                    # Small delay for processing
                    time.sleep(0.1)
                
                cap.release()
        
        with col2:
            st.markdown("### 📊 Statistics")
            if 'detection_stats' in locals() and detection_stats:
                for mask_type, count in sorted(detection_stats.items()):
                    risk = calculate_virus_risk_percentage(mask_type)
                    st.metric(mask_type, count)
                    st.progress(risk/100, text=f"Risk: {risk}%")
            else:
                st.info("Start detection to see statistics")

elif option == "👥 Crowd Monitoring":
    st.subheader("👥 Real-Time Crowd Monitoring")
    
    # Load YOLO model
    yolo_model = load_yolo_model()
    
    if yolo_model is None:
        st.error("❌ YOLO model failed to load!")
    else:
        # Crowd settings
        col1, col2 = st.columns([2, 1])
        
        with col1:
            mode = st.radio("Select Input", ["📹 Webcam", "📹 Upload Video"], horizontal=True)
        
        with col2:
            crowd_threshold = st.number_input(
                "Crowd Threshold (people)",
                min_value=5,
                max_value=100,
                value=20
            )
        
        st.markdown("---")
        
        if mode == "📹 Webcam":
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📹 Camera Feed")
                run_crowd = st.checkbox("Start Crowd Detection", value=False)
                
                frame_placeholder = st.empty()
                
                if run_crowd:
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    alert_triggered_prev = False
                    
                    while run_crowd and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to access camera")
                            break
                        
                        frame = cv2.flip(frame, 1)
                        
                        # Detect crowd
                        processed_frame, person_count, alert_triggered = detect_crowd(
                            frame, yolo_model, crowd_threshold
                        )
                        
                        # Play alert if triggered and not previously triggered
                        if alert_triggered and not alert_triggered_prev:
                            threading.Thread(
                                target=play_alert_sound,
                                args=("Warning! Excessive crowd detected. Please disperse and maintain social distancing.",),
                                daemon=True
                            ).start()
                        
                        alert_triggered_prev = alert_triggered
                        
                        # Display frame
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, use_column_width=True)
                        
                        time.sleep(0.1)
                    
                    cap.release()
            
            with col2:
                st.markdown("### ⚙️ Status")
                st.info("Detecting crowd...")
        
        else:  # Upload Video
            st.markdown("### 📹 Upload Video")
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv']
            )
            
            if uploaded_file is not None:
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Processing Video")
                    run_video = st.checkbox("Process Video", value=False)
                    
                    frame_placeholder = st.empty()
                    
                    if run_video:
                        cap = cv2.VideoCapture(tmp_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        progress_bar = st.progress(0)
                        frame_idx = 0
                        
                        max_crowd_count = 0
                        alert_count = 0
                        alert_triggered_prev = False
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Detect crowd
                            processed_frame, person_count, alert_triggered = detect_crowd(
                                frame, yolo_model, crowd_threshold
                            )
                            
                            max_crowd_count = max(max_crowd_count, person_count)
                            
                            if alert_triggered and not alert_triggered_prev:
                                alert_count += 1
                                play_alert_sound("Warning! Excessive crowd detected.")
                            
                            alert_triggered_prev = alert_triggered
                            
                            # Display frame
                            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, use_column_width=True)
                            
                            # Update progress
                            frame_idx += 1
                            progress_bar.progress(min(frame_idx / total_frames, 1.0))
                        
                        cap.release()
                        
                        # Show results
                        st.success("✅ Video processing complete!")
                
                with col2:
                    st.markdown("### 📊 Results")
                    if 'max_crowd_count' in locals():
                        st.metric("Max People Detected", max_crowd_count)
                        st.metric("Alert Triggers", alert_count)
                        
                        if max_crowd_count > crowd_threshold:
                            st.warning(f"⚠️ Crowd exceeded threshold {crowd_threshold}")
                        else:
                            st.success(f"✅ Crowd within threshold {crowd_threshold}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🎭 Real-Time Mask & Crowd Monitoring System v1.0</p>
    <p>Built with Streamlit, OpenCV, PyTorch & YOLO</p>
</div>
""", unsafe_allow_html=True)
