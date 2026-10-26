# 📥 CSRNet Pretrained Weights Download Guide

## ⚠️ CRITICAL: You MUST Download CSRNet Weights

The system **will not work correctly** without pretrained CSRNet weights!

---

## Option 1: Google Drive (Recommended) ⭐

### Step 1: Go to Download Link

**Link:** https://drive.google.com/file/d/1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru/view

### Step 2: Download File

1. Click "Download" button (top right)
2. File name: `CSRNet-ShanghaiTech_part_A.pth`
3. Size: ~550MB
4. Download time: 2-5 minutes (depending on connection)

### Step 3: Save to Correct Location

```bash
# Create weights directory
mkdir -p weights

# Move downloaded file
mv ~/Downloads/CSRNet-ShanghaiTech_part_A.pth weights/csrnet_shanghaitech_a.pth
```

### Step 4: Verify

```bash
ls -lh weights/csrnet_shanghaitech_a.pth
# Should show: ~550MB
```

---

## Option 2: GitHub Release

### From leeyeehoo/CSRNet-pytorch

```bash
# Download using wget
wget https://github.com/leeyeehoo/CSRNet-pytorch/releases/download/v1.0/CSRNet-ShanghaiTech_part_A.pth

# Move to weights directory
mkdir -p weights
mv CSRNet-ShanghaiTech_part_A.pth weights/csrnet_shanghaitech_a.pth
```

---

## Option 3: Baidu Cloud (For China Users)

**Link:** https://pan.baidu.com/s/1pMuGyNp  
**Password:** b3h5

Download and rename to: `weights/csrnet_shanghaitech_a.pth`

---

## Option 4: Train Your Own (Advanced)

If you want to train from scratch:

### Requirements
- ShanghaiTech Part A dataset
- ~20 hours training time on RTX 4050
- ~400 epochs

### Steps

```bash
# Clone official CSRNet repo
git clone https://github.com/leeyeehoo/CSRNet-pytorch.git
cd CSRNet-pytorch

# Download ShanghaiTech dataset
# Follow instructions in their README

# Train
python train.py

# Copy trained weights
cp model_best.pth ../hybrid_crowd_system/weights/csrnet_shanghaitech_a.pth
```

---

## ✅ Verification Checklist

After downloading:

1. File exists at `weights/csrnet_shanghaitech_a.pth`
2. File size is ~550MB (not 1KB or empty)
3. Run test:

```bash
python -c "import torch; checkpoint = torch.load('weights/csrnet_shanghaitech_a.pth'); print('✓ Weights valid!'); print(f'Keys: {len(checkpoint)}'); print(f'Model: {type(checkpoint)}')"
```

Expected output:
```
✓ Weights valid!
Keys: 54
Model: <class 'collections.OrderedDict'>
```

4. Run full system test:

```bash
python config.py
# Should show: ✓ CSRNet weights loaded successfully
```

---

## 🚫 Common Issues

### Issue 1: File Not Found Error

```
FileNotFoundError: [Errno 2] No such file or directory: 'weights/csrnet_shanghaitech_a.pth'
```

**Solution:**
- Check file path is exactly: `weights/csrnet_shanghaitech_a.pth`
- No typos in filename
- File is in `weights/` folder, not `models/` or root directory

### Issue 2: Architecture Mismatch Error

```
RuntimeError: Error(s) in loading state_dict for CSRNet:
...
```

**Cause:** Downloaded wrong model weights

**Solution:**
- Make sure you downloaded **ShanghaiTech Part A** weights
- NOT Part B weights
- NOT UCF-QNRF weights
- File should be ~550MB, not ~300MB

### Issue 3: Model Works But Counts Are Wrong

**Symptoms:**
- System runs without errors
- But counts are very low (always < 5)
- Or counts are random

**Cause:** Weights didn't load, using random initialization

**Solution:**
```bash
# Check if weights actually loaded
python -c "from models import CSRNetCounter; c = CSRNetCounter()"
# Should show: "✓ CSRNet weights loaded successfully (strict=True)"
```

### Issue 4: Download Interrupted

**Solution:**
- Re-download completely
- Don't resume partial downloads
- Verify file size is exactly ~550MB
- MD5 checksum: `e00b4d1b5c25c43e3e1d7b8a8c9b7f3d` (if available)

---

## 📊 Model Specifications

**CSRNet ShanghaiTech Part A:**

- **Dataset:** ShanghaiTech Part A (dense crowds)
- **Training Images:** 300 images
- **Average Crowd Size:** ~500 people/image
- **Architecture:** VGG-16 frontend + Dilated convolution backend
- **Performance:** MAE ~68 on test set
- **Input Size:** Flexible (divisible by 8)
- **Output:** Density map at 1/8 resolution

---

## 🔄 Alternative Weights (If Needed)

If you can't download ShanghaiTech Part A weights, you can use:

### ShanghaiTech Part B (Sparser crowds)
- Less dense crowds
- MAE ~10-12
- ~300MB file size
- Update config: `CSRNET_WEIGHTS_PATH = 'weights/csrnet_shanghaitech_b.pth'`

### UCF-QNRF (Very large crowds)
- Extreme density
- MAE ~100-120
- ~550MB file size
- Update config: `CSRNET_WEIGHTS_PATH = 'weights/csrnet_ucf_qnrf.pth'`

**Note:** Performance will vary with different weights!

---

## 🎯 After Downloading

Once weights are downloaded:

```bash
# 1. Verify installation
python config.py

# 2. Test CSRNet alone
python -c "from models import CSRNetCounter; import numpy as np; c = CSRNetCounter(); img = np.random.randint(0,255,(480,640,3),dtype=np.uint8); r = c.predict(img); print(f'Count: {r[\"count\"]:.1f}')"

# 3. Run full system
python main.py --webcam --mode adaptive
```

---

## 📞 Still Having Issues?

1. **Check exact file path:**
   ```bash
   pwd  # Should be in hybrid_crowd_system/
   ls weights/csrnet_shanghaitech_a.pth  # Should exist
   ```

2. **Check file size:**
   ```bash
   ls -lh weights/csrnet_shanghaitech_a.pth
   # Should be ~550MB, NOT 1KB or 0 bytes
   ```

3. **Try re-downloading:**
   - Delete partial download
   - Clear browser cache
   - Download again from Google Drive

4. **Check PyTorch version:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   # Should be 2.0.0 or newer
   ```

---

## ✅ Success Indicators

You'll know it's working when:

1. ✅ `python config.py` shows "✓ CSRNet weights loaded successfully"
2. ✅ No errors when running `python main.py --webcam`
3. ✅ Density map overlay appears in webcam view
4. ✅ CSRNet counts seem reasonable (not always 0 or random)

---

**Download Link (repeat):**  
https://drive.google.com/file/d/1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru/view

**Save as:** `weights/csrnet_shanghaitech_a.pth`

**File size:** ~550MB
