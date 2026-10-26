"""
Configuration file for Hybrid Crowd Counting System
All parameters in one place
"""
import torch
import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_FP16 = False  # Enable half precision (faster on modern GPUs)

# ============================================================================
# YOLO CONFIGURATION
# ============================================================================
YOLO_MODEL = os.path.join(BASE_DIR, "yolov8n.pt")  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.35
YOLO_IMG_SIZE = 1280

# ============================================================================
# CSRNET CONFIGURATION
# ============================================================================
# Download pretrained weights from:
# https://drive.google.com/file/d/1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru/view
# (ShanghaiTech Part A - 550MB)
# Save as: weights/csrnet_shanghaitech_a.pth


# Absolute path to weights
CSRNET_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "csrnet_shanghaitech_a.pth")

# ============================================================================
# HYBRID MODE CONFIGURATION
# ============================================================================
# For basic hybrid mode
CSRNET_INTERVAL = 10  # Run CSRNet every N frames

# For adaptive mode
SPARSE_THRESHOLD = 20   # Below this: use YOLO only
DENSE_THRESHOLD = 100    # Above this: use CSRNet only

# ============================================================================
# DENSITY CLASSIFICATION
# ============================================================================
DENSITY_LOW = 20
DENSITY_MEDIUM = 50

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
SHOW_BOXES = True
SHOW_DENSITY = True
OVERLAY_ALPHA = 0.4

# Text display
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)

# ============================================================================
# PERFORMANCE
# ============================================================================
CACHE_CLEAR_INTERVAL = 100  # Clear CUDA cache every N frames

# ============================================================================
# OUTPUT PATHS
# ============================================================================
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate configuration"""
    print("="*60)
    print("SYSTEM CONFIGURATION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"\nYOLO Model: {YOLO_MODEL}")
    print(f"CSRNet Weights: {CSRNET_WEIGHTS_PATH}")
    print(f"FP16 Enabled: {USE_FP16}")
    print(f"\nThresholds:")
    print(f"  Sparse: < {SPARSE_THRESHOLD}")
    print(f"  Dense: > {DENSE_THRESHOLD}")
    print("="*60)
    
    return True

if __name__ == '__main__':
    validate_config()
