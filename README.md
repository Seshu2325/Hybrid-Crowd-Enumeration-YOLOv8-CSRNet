# Hybrid Crowd Enumeration using YOLOv8 and CSRNet

An academic computer-vision project that combines YOLOv8 person detection with CSRNet density-map estimation to explore crowd counting across different scene conditions.

> **Project status:** This is an experimental project. Quantitative accuracy and performance claims are not included because a reproducible benchmark has not yet been documented in this repository.

## Features

- Detects people using Ultralytics YOLOv8 (COCO person class).
- Uses CSRNet to estimate a crowd count from a density map.
- Provides an adaptive mode that uses YOLO detections and image-edge statistics as heuristics for deciding when to run CSRNet.
- Provides a basic hybrid mode that runs both models and averages their counts.
- Supports image, video, and webcam input.
- Can save annotated image and video results.
- Includes utilities for drawing bounding boxes, visualizing density maps, and calculating evaluation metrics.

## Project structure

```text
Hybrid-Crowd-Enumeration-YOLOv8-CSRNet/
├── main.py
├── config.py
├── requirements.txt
├── yolov8n.pt
├── sample_ground_truth.json
├── models/
│   ├── adaptive_hybrid.py
│   ├── csrnet_counter.py
│   └── yolo_counter.py
├── utils/
│   ├── metrics.py
│   └── visualization.py
├── README.md
├── EXAMPLES.md
├── TECHNICAL_DOC.md
└── WEIGHTS_DOWNLOAD.md
```

The exact files in your local copy may include additional helper files.

## Requirements

- Python with compatible versions of PyTorch, Torchvision, OpenCV, NumPy, Ultralytics, and tqdm.
- A webcam and microphone are not required for image or video-file inference; a webcam is required for webcam mode.
- A compatible CSRNet checkpoint is required for meaningful CSRNet predictions.

## Installation (Windows)

Open Command Prompt in the repository root directory.

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation can depend on your Python version and whether you need CPU or CUDA support. If installing from `requirements.txt` fails, consult the official PyTorch installation selector and install a compatible PyTorch/Torchvision pair before installing the remaining dependencies.

## Model weights

The YOLO checkpoint `yolov8n.pt` is included in this repository.

The CSRNet checkpoint is not included because of its file size. The current project configuration expects it at:

```text
weights/csrnet_shanghaitech_a.pth
```

Create a `weights` folder in the repository root and place a compatible checkpoint at that path before relying on CSRNet predictions. The download URL currently referenced by `config.py` is:

[CSRNet checkpoint link (Google Drive)](https://drive.google.com/file/d/1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru/view)

Verify that the downloaded checkpoint is compatible with this CSRNet implementation. If it is missing or incompatible, CSRNet results should not be treated as valid.

## Usage

Run the commands from the repository root. Replace the example input paths with paths to your own files.

### Show command-line options

```bat
python main.py --help
```

### Process an image

```bat
python main.py --image "path\to\image.jpg" --mode adaptive
```

Save the visualization:

```bat
python main.py --image "path\to\image.jpg" --mode adaptive --save
```

### Process a video

```bat
python main.py --video "path\to\video.mp4" --mode adaptive
```

Save the processed video:

```bat
python main.py --video "path\to\video.mp4" --mode adaptive --save
```

### Use a webcam

```bat
python main.py --webcam --camera 0 --mode adaptive
```

Press **Q** to exit video or webcam display.

### Run basic hybrid mode

```bat
python main.py --image "path\to\image.jpg" --mode basic_hybrid --save
```

When `--save` is used, results are written to the configured output directory (by default, `outputs`).

## How it works

1. **YOLOv8 detector:** Detects individual people and returns bounding boxes and a detection-based count.
2. **CSRNet counter:** Produces a density map and estimates a count. The current implementation applies CSRNet to overlapping image tiles.
3. **Adaptive mode:** Runs YOLO first and uses detection-related statistics and Canny edge density as heuristics to decide whether to run CSRNet.
4. **Basic hybrid mode:** Runs both counters and returns the arithmetic mean of their estimated counts.
5. **Visualization:** Draws available bounding boxes and overlays a density map when provided.

The adaptive rules and tile-overlap correction are heuristic design choices and need evaluation on labelled data before being described as accuracy improvements.

## Evaluation

The repository includes evaluation utilities and a sample ground-truth file. To report performance responsibly, evaluate predictions on a documented held-out dataset and record the dataset/split, model checkpoint, configuration, and metrics. For crowd counting, useful metrics include mean absolute error (MAE) and root mean squared error (RMSE). Compare YOLOv8, CSRNet, and the hybrid output on the same images before claiming an improvement.

## Limitations

- Results depend on the quality of the input, the selected model weights, and the configured thresholds.
- Edge density can be affected by backgrounds and textures, so it is only a heuristic for crowd density.
- The basic hybrid mean and tile-overlap correction have not been established as optimal without evaluation.
- Inference speed depends on the hardware, image size, selected model, and whether CUDA is available.

## Future improvements

- Benchmark all modes on a labelled dataset using MAE and RMSE.
- Validate and tune adaptive-switching thresholds using held-out data.
- Measure end-to-end inference time and memory use on documented hardware.
- Improve checkpoint validation and error messages.

## Technologies

Python, PyTorch, Torchvision, OpenCV, NumPy, Ultralytics YOLOv8, CSRNet.
