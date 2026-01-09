# ==================== MASK_DETECTOR.PY ====================
"""
Mask Detection Module (PyTorch)
Detects 5 classes of masks and calculates virus risk
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from config import (
    MASK_CLASSES, MASK_TARGET_SIZE, VIRUS_RISK_MAP,
    FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS
)

class MaskDetector:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        if self.model is not None:
            self.model.eval()  # Set to evaluation mode
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Define the same preprocessing transform used during training
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(MASK_TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Apply transforms
            img_tensor = self.transform(image_rgb)
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def detect_faces(self, frame):
        """Detect faces in frame using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=(30, 30)
        )
        return faces
    
    def classify_mask(self, roi):
        """Classify mask type in ROI"""
        if self.model is None:
            return None, 0
        
        processed_roi = self.preprocess_image(roi)
        if processed_roi is None:
            return None, 0
        
        try:
            with torch.no_grad():
                processed_roi = processed_roi.to(self.device)
                outputs = self.model(processed_roi)
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, class_idx = torch.max(probabilities, 1)
                
                confidence = float(confidence[0])
                class_idx = int(class_idx[0])
                mask_class = MASK_CLASSES[class_idx]
                
                return mask_class, confidence
        except Exception as e:
            print(f"Classification error: {e}")
            return None, 0
    
    def get_risk_percentage(self, mask_class):
        """Get virus spreading risk for mask type"""
        return VIRUS_RISK_MAP.get(mask_class, 100)
    
    def get_box_color(self, mask_class):
        """Get bounding box color based on risk level"""
        risk = self.get_risk_percentage(mask_class)
        
        if risk <= 15:
            return (0, 255, 0)      # Green - Safe
        elif risk <= 40:
            return (0, 165, 255)    # Orange - Warning
        else:
            return (0, 0, 255)      # Red - Danger
    
    def detect_and_classify(self, frame):
        """Detect masks in frame and return annotated frame + detections"""
        if self.model is None:
            return frame, []
        
        detections = []
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
            
            # Classify mask
            mask_class, confidence = self.classify_mask(roi)
            
            if mask_class is None:
                continue
            
            risk_percentage = self.get_risk_percentage(mask_class)
            color = self.get_box_color(mask_class)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{mask_class}: {confidence:.2%}"
            cv2.putText(
                frame, label, (x, y-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            
            # Draw risk
            risk_label = f"Risk: {risk_percentage}%"
            cv2.putText(
                frame, risk_label, (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            detections.append({
                'mask_type': mask_class,
                'confidence': confidence,
                'risk_percentage': risk_percentage,
                'bbox': (x, y, w, h)
            })
        
        return frame, detections
