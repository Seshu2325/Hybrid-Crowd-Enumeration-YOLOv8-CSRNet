# 🚀 QUICK START EXAMPLES

## Complete Usage Examples for All Scenarios

---

## 📸 Single Image Mode

### Example 1: Basic Image Processing

```bash
python main.py --image crowd_photo.jpg --mode adaptive
```

**What happens:**
- Loads the image
- Runs adaptive hybrid system
- Displays result with overlay
- Shows count and FPS

### Example 2: Image with Ground Truth

```bash
python main.py --image stadium.jpg --ground_truth 850 --mode adaptive
```

**Output:**
```
Processing: stadium.jpg

Results:
  Active Model: CSRNET
  Count: 847.3
  FPS: 18.2
  Ground Truth: 850
  Error: 2.7
```

### Example 3: Save Output

```bash
python main.py --image crowd.jpg --mode adaptive --save
```

**Saves to:** `outputs/result_crowd.jpg`

---

## 📁 Directory Mode

### Example 1: Process All Images

```bash
python main.py --directory images/ --mode adaptive
```

**Processes:**
- All .jpg, .jpeg, .png files
- Runs adaptive system
- Shows progress

### Example 2: Batch with Ground Truth

**Create ground_truth.json:**
```json
{
  "image1.jpg": 120,
  "image2.jpg": 85,
  "image3.jpg": 200
}
```

**Run:**
```bash
python main.py --directory images/ --ground_truth_file ground_truth.json --mode adaptive
```

**Output:**
```
[1/3] image1.jpg
  Count: 118.5
  Ground Truth: 120

[2/3] image2.jpg
  Count: 87.2
  Ground Truth: 85

[3/3] image3.jpg
  Count: 195.8
  Ground Truth: 200

============================================================
EVALUATION METRICS
============================================================

YOLOv8:
  MAE:  15.30
  MSE:  280.50
  RMSE: 16.75

CSRNet:
  MAE:  3.23
  MSE:  12.50
  RMSE: 3.54

Best Model (by MAE): CSRNet
============================================================
```

### Example 3: Save All Results

```bash
python main.py --directory images/ --mode adaptive --save
```

**Saves:**
- `outputs/result_image1.jpg`
- `outputs/result_image2.jpg`
- etc.

---

## 🎥 Video Mode

### Example 1: Process Video File

```bash
python main.py --video crowd_footage.mp4 --mode adaptive
```

**What happens:**
- Opens video file
- Processes frame by frame
- Shows live preview
- Press 'Q' to quit

### Example 2: Save Output Video

```bash
python main.py --video input.mp4 --mode adaptive --save
```

**Saves to:** `outputs/result_input.mp4`

### Example 3: Disable Overlays

```bash
python main.py --video input.mp4 --mode adaptive --no_boxes --no_density
```

**Shows:**
- Just the count and FPS
- No bounding boxes
- No density map

---

## 📹 Webcam Mode

### Example 1: Basic Webcam

```bash
python main.py --webcam --mode adaptive
```

**Controls:**
- `Q` - Quit
- `S` - Save frame
- `B` - Toggle boxes
- `D` - Toggle density

### Example 2: Specific Camera

```bash
python main.py --webcam --camera 1 --mode adaptive
```

**Use camera index 1** (if you have multiple cameras)

### Example 3: Clean View

```bash
python main.py --webcam --mode adaptive --no_boxes
```

**Shows:**
- Density map only
- No bounding boxes
- Cleaner visualization

---

## 🔄 Mode Comparison

### Adaptive Mode (Recommended)

```bash
python main.py --webcam --mode adaptive
```

**Display shows:**
```
ADAPTIVE HYBRID SYSTEM
Active: YOLO
Count: 15 people | FPS: 45.2
Density: Sparse
```

**Or when dense:**
```
ADAPTIVE HYBRID SYSTEM
Active: CSRNET  
Count: 245 people | FPS: 18.5
Density: Dense
```

### Basic Hybrid Mode

```bash
python main.py --webcam --mode basic_hybrid
```

**Display shows:**
```
HYBRID COUNTING SYSTEM
YOLO: 12 | 45.2 FPS
CSRNet: 11.8 | 18.5 FPS
Density: Sparse
```

**Shows both counts simultaneously**

---

## 🎯 Real-World Scenarios

### Scenario 1: Event Monitoring

**Task:** Monitor entrance to an event

```bash
python main.py --webcam --mode adaptive --save
```

**Setup:**
1. Position camera at entrance
2. Run system
3. Press 'S' periodically to save snapshots
4. Review `outputs/webcam_frame_*.png` later

### Scenario 2: Mall Traffic Analysis

**Task:** Analyze recorded mall footage

```bash
python main.py --video mall_footage.mp4 --mode adaptive --save
```

**Result:**
- Processed video with counts
- Saved to `outputs/result_mall_footage.mp4`
- Can analyze later

### Scenario 3: Dataset Evaluation

**Task:** Evaluate on benchmark dataset

```bash
# Create ground_truth.json with your labels
python main.py --directory dataset/test/ --ground_truth_file dataset/test_labels.json --mode adaptive
```

**Gets:**
- MAE, MSE, RMSE
- FPS comparison
- Model comparison

### Scenario 4: Live Security Monitoring

**Task:** Real-time security camera feed

```bash
python main.py --webcam --camera 0 --mode adaptive
```

**Features:**
- Real-time counting
- Automatic mode switching
- Visual density indication
- Save suspicious frames with 'S'

---

## ⚙️ Configuration Examples

### High Accuracy Setup

**Edit config.py:**
```python
YOLO_MODEL = 'yolov8s.pt'  # Larger model
SPARSE_THRESHOLD = 15       # Lower threshold
DENSE_THRESHOLD = 40        # Earlier switching
USE_FP16 = False            # Full precision
```

### High Speed Setup

**Edit config.py:**
```python
YOLO_MODEL = 'yolov8n.pt'  # Fastest model
SPARSE_THRESHOLD = 25       # Higher threshold
DENSE_THRESHOLD = 60        # Later switching
USE_FP16 = True             # Half precision
CSRNET_INTERVAL = 15        # Longer cache
```

### Balanced Setup (Default)

```python
YOLO_MODEL = 'yolov8n.pt'
SPARSE_THRESHOLD = 20
DENSE_THRESHOLD = 50
USE_FP16 = True
CSRNET_INTERVAL = 10
```

---

## 🧪 Testing Examples

### Test 1: Verify Installation

```bash
# Test configuration
python config.py
```

**Should show:**
```
============================================================
SYSTEM CONFIGURATION
============================================================
Device: cuda
CUDA Available: True
GPU: NVIDIA GeForce RTX 4050
...
```

### Test 2: Test Models Individually

```bash
# Test YOLO
python -c "from models import YOLOCounter; y = YOLOCounter(); print('✓ YOLO works')"

# Test CSRNet
python -c "from models import CSRNetCounter; c = CSRNetCounter(); print('✓ CSRNet works')"
```

### Test 3: Quick Webcam Test

```bash
python main.py --webcam --mode adaptive
# Should open window immediately
# Press 'Q' to quit
```

---

## 📊 Output Examples

### Console Output

```
============================================================
HYBRID CROWD COUNTING SYSTEM
============================================================

[1/2] Loading YOLOv8...
Loading YOLOv8: yolov8n.pt
✓ YOLOv8 initialized on cuda

[2/2] Loading CSRNet...
Building CSRNet model...
Loading CSRNet weights from: weights/csrnet_shanghaitech_a.pth
✓ CSRNet weights loaded successfully (strict=True)
✓ FP16 enabled for CSRNet
✓ CSRNet warm-up completed
✓ CSRNet initialized on cuda

[3/3] Creating Adaptive Hybrid System...
✓ Adaptive Hybrid System initialized
  Thresholds: Sparse < 20, Dense > 50

============================================================
✓ SYSTEM READY
============================================================
```

### Metrics Output

```
============================================================
EVALUATION METRICS
============================================================

YOLOv8:
  MAE:  18.50
  MSE:  425.30
  RMSE: 20.62
  Avg FPS: 45.30

CSRNet:
  MAE:  8.20
  MSE:  85.40
  RMSE: 9.24
  Avg FPS: 18.50

Best Model (by MAE): CSRNet
============================================================
```

---

## 🎬 Step-by-Step Walkthrough

### First Time Setup

```bash
# 1. Create environment
conda create -n hybrid_crowd python=3.10 -y
conda activate hybrid_crowd

# 2. Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
cd hybrid_crowd_system
pip install -r requirements.txt

# 4. Download CSRNet weights
# Go to: https://drive.google.com/file/d/1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru/view
# Save to: weights/csrnet_shanghaitech_a.pth

# 5. Test
python config.py

# 6. Run!
python main.py --webcam --mode adaptive
```

### Daily Usage

```bash
# Activate environment
conda activate hybrid_crowd

# Navigate to project
cd hybrid_crowd_system

# Run based on your need:

# Webcam
python main.py --webcam --mode adaptive

# Process images
python main.py --directory images/ --mode adaptive --save

# Process video
python main.py --video footage.mp4 --mode adaptive --save
```

---

## 💡 Tips & Tricks

### Tip 1: Faster Processing

```bash
# Use smaller YOLO model
python main.py --video long_video.mp4 --mode adaptive

# Edit config.py first:
# YOLO_MODEL = 'yolov8n.pt'
# USE_FP16 = True
```

### Tip 2: Better Accuracy

```bash
# Use larger YOLO model
# Edit config.py:
# YOLO_MODEL = 'yolov8s.pt'

python main.py --image test.jpg --mode basic_hybrid
```

### Tip 3: Save Everything

```bash
# Process and save all
python main.py --directory dataset/ --ground_truth_file labels.json --mode adaptive --save
```

### Tip 4: Quick Counts

```bash
# Fastest possible (YOLO only for sparse)
python main.py --webcam --mode adaptive --no_density
```

---

## ✅ Success Indicators

You'll know it's working when:

1. ✅ `python config.py` shows "✓ SYSTEM READY"
2. ✅ Webcam opens with live counting
3. ✅ FPS > 30 on adaptive mode
4. ✅ Counts look reasonable
5. ✅ Mode switches automatically (adaptive)
6. ✅ Density overlay appears (if enabled)
7. ✅ No errors in console

---

**For full documentation, see README.md**
