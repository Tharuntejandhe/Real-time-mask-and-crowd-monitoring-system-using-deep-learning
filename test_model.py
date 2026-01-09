"""
Test the trained PyTorch mask detection model
Run this to verify the model works correctly
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# Define model architecture (same as training)
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

def load_model(model_path='models/mask_detection_model.pth'):
    """Load the trained model"""
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    model = MaskDetectionCNN(num_classes=5)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully from {model_path}")
    return model, device

def test_on_sample_images():
    """Test model on sample images from the dataset"""
    import os
    import random
    
    print("\n" + "="*60)
    print("TESTING MODEL ON SAMPLE IMAGES")
    print("="*60)
    
    # Load model
    model, device = load_model()
    
    # Class names in alphabetical order
    class_names = ['Cloth Mask', 'N95 Mask', 'Partial Mask', 'No Mask', 'Surgical Mask']
    
    # Define preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get sample images from each class
    dataset_path = 'data/Dataset/train'
    class_folders = ['cloth', 'n95', 'n95v', 'nfm', 'srg']
    
    print("\nTesting on random samples from each class:\n")
    
    for idx, folder in enumerate(class_folders):
        folder_path = os.path.join(dataset_path, folder)
        images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if images:
            # Pick a random image
            sample_image = random.choice(images)
            image_path = os.path.join(folder_path, sample_image)
            
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, class_idx = torch.max(probabilities, 1)
            
            actual_class = class_names[idx]
            predicted_class = class_names[class_idx.item()]
            confidence_score = confidence.item()
            
            # Check if correct
            is_correct = "✅" if predicted_class == actual_class else "❌"
            
            print(f"{is_correct} Folder: {folder:6s} | Actual: {actual_class:15s} | Predicted: {predicted_class:15s} | Confidence: {confidence_score:.2%}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

def test_webcam():
    """Test model on webcam feed"""
    print("\n" + "="*60)
    print("TESTING MODEL ON WEBCAM")
    print("="*60)
    print("Press 'q' to quit\n")
    
    # Load model
    model, device = load_model()
    
    # Class names
    class_names = ['Cloth Mask', 'N95 Mask', 'Partial Mask', 'No Mask', 'Surgical Mask']
    
    # Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Webcam opened. Looking for faces...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Process each face
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
            
            # Convert BGR to RGB and preprocess
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img_tensor = transform(roi_rgb).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, class_idx = torch.max(probabilities, 1)
            
            mask_class = class_names[class_idx.item()]
            conf_score = confidence.item()
            
            # Determine color based on class
            if mask_class in ['N95 Mask', 'Surgical Mask']:
                color = (0, 255, 0)  # Green
            elif mask_class == 'Cloth Mask':
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{mask_class}: {conf_score:.2%}"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display frame
        cv2.imshow('Mask Detection Test', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam test complete!")

if __name__ == '__main__':
    import sys
    
    print("\n🎭 MASK DETECTION MODEL TESTER")
    print("="*60)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--webcam':
        test_webcam()
    else:
        test_on_sample_images()
        print("\n💡 To test with webcam, run: python test_model.py --webcam")
