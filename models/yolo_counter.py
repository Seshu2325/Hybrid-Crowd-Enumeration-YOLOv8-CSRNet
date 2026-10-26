"""
YOLOv8 Counter - Detection-based crowd counting
Uses Ultralytics YOLOv8 for person detection
"""

import torch
import cv2
import numpy as np
import time
from ultralytics import YOLO
import config


class YOLOCounter:
    """
    YOLOv8-based person counter
    """
    
    def __init__(self, model_path=config.YOLO_MODEL, device=None):
        """
        Initialize YOLO counter
        
        Args:
            model_path: Path to YOLO weights
            device: torch.device (defaults to config.DEVICE)
        """
        self.device = device if device else config.DEVICE
        self.img_size = config.YOLO_IMG_SIZE
        self.conf_threshold = config.YOLO_CONF_THRESHOLD
        self.iou_threshold = config.YOLO_IOU_THRESHOLD
        
        print(f"Loading YOLOv8: {model_path}")
        self.model = YOLO(model_path)
        
        # Move to device
        if self.device.type == 'cuda':
            self.model.to(self.device)
        
        # Warm-up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.predict(dummy, return_boxes=False)
        
        print(f"✓ YOLOv8 initialized on {self.device}")
    
    def predict(self, image, return_boxes=True):
        """
        Detect people in image
        
        Args:
            image: Input image (numpy array, BGR)
            return_boxes: Whether to return bounding boxes
        
        Returns:
            dict: {
                'count': int,
                'boxes': list (optional),
                'confidences': list (optional),
                'fps': float,
                'inference_time': float
            }
        """
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            image,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],  # Only person class
            half=config.USE_FP16 and self.device.type == 'cuda',
            verbose=False,
            device=self.device
        )
        
        # Extract results
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else np.array([])
        confidences = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else np.array([])
        count = len(boxes)
        
        inference_time = time.time() - start_time
        
        output = {
            'count': count,
            'inference_time': inference_time,
            'fps': 1.0 / inference_time if inference_time > 0 else 0
        }
        
        if return_boxes:
            output['boxes'] = boxes.tolist()
            output['confidences'] = confidences.tolist()
        
        return output
    
    def draw_boxes(self, image, boxes, confidences=None):
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            boxes: List of bounding boxes [x1, y1, x2, y2]
            confidences: Optional confidence scores
        
        Returns:
            Image with boxes drawn
        """
        img = image.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence if available
            if confidences is not None and i < len(confidences):
                conf = confidences[i]
                label = f'{conf:.2f}'
                cv2.putText(img, label, (x1, y1 - 5),
                           config.FONT, config.FONT_SCALE - 0.2,
                           (0, 255, 0), config.FONT_THICKNESS - 1)
        
        return img


def load_yolo_model(model_path=config.YOLO_MODEL, device=None):
    """Factory function to load YOLO model"""
    return YOLOCounter(model_path, device)


if __name__ == '__main__':
    print("Testing YOLOCounter...")
    
    # Create test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize
    counter = YOLOCounter()
    
    # Predict
    result = counter.predict(test_img)
    
    print(f"Count: {result['count']}")
    print(f"FPS: {result['fps']:.1f}")
    print("✓ YOLOCounter works!")
