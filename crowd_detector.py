# ==================== CROWD_DETECTOR.PY ====================
"""
Crowd Monitoring Module
Detects people using YOLO and counts crowd
"""

import cv2
import numpy as np
from config import YOLO_CONFIDENCE, PERSON_CLASS_ID, CROWD_THRESHOLD

class CrowdDetector:
    def __init__(self, yolo_model):
        self.model = yolo_model
        self.person_count = 0
        self.alert_triggered = False
        self.detection_history = []
    
    def detect_persons(self, frame, threshold=CROWD_THRESHOLD):
        """Detect people in frame using YOLO"""
        if self.model is None:
            return frame, 0, False
        
        try:
            # Run YOLO inference
            results = self.model(frame, conf=YOLO_CONFIDENCE, verbose=False)
            
            person_count = 0
            detections = []
            
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == PERSON_CLASS_ID:  # Person class
                        person_count += 1
                        
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame, f"Person {conf:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf
                        })
            
            # Check alert condition
            alert_triggered = person_count > threshold
            
            # Draw status on frame
            status_text = f"People Count: {person_count}/{threshold}"
            status_color = (0, 255, 0) if not alert_triggered else (0, 0, 255)
            
            cv2.putText(
                frame, status_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3
            )
            
            if alert_triggered:
                cv2.putText(
                    frame, "CROWD ALERT!", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
                )
                cv2.putText(
                    frame, "DISPERSE IMMEDIATELY!", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            
            self.person_count = person_count
            self.alert_triggered = alert_triggered
            self.detection_history.append({
                'count': person_count,
                'alert': alert_triggered,
                'detections': len(detections)
            })
            
            return frame, person_count, alert_triggered
        
        except Exception as e:
            print(f"Error in crowd detection: {e}")
            return frame, 0, False
    
    def get_statistics(self):
        """Get statistics from detection history"""
        if not self.detection_history:
            return {}
        
        counts = [h['count'] for h in self.detection_history]
        alerts = [h['alert'] for h in self.detection_history]
        
        return {
            'max_people': max(counts),
            'min_people': min(counts),
            'avg_people': sum(counts) / len(counts),
            'total_alerts': sum(alerts),
            'frames_processed': len(self.detection_history)
        }
    
    def reset(self):
        """Reset detection history"""
        self.detection_history = []
        self.person_count = 0
        self.alert_triggered = False
