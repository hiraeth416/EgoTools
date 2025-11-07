# Hand Pose Extraction Guide

This guide covers extracting hand pose (MANO parameters and 3D joints) from video segments using HaMeR or MediaPipe.

## Table of Contents

- [Overview](#overview)
- [Model Comparison](#model-comparison)
- [Basic Usage](#basic-usage)
- [Arguments](#arguments)
- [Output Structure](#output-structure)
- [Model Details](#model-details)
- [Advanced Usage](#advanced-usage)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Overview

Hand pose extraction provides:
- **2D keypoints**: Hand joint locations in image space
- **3D joints**: Hand joint positions in 3D space (HaMeR only)
- **MANO parameters**: Hand mesh parameters (HaMeR only)
- **Hand vertices**: 3D hand mesh vertices (HaMeR only)

Supported models:
- **MediaPipe Hands**: Fast, CPU-friendly, 2D/3D keypoints
- **HaMeR**: Accurate, GPU-required, full MANO parameters

## Model Comparison

| Feature | MediaPipe | HaMeR |
|---------|-----------|-------|
| **Speed** | Very Fast | Slower |
| **Device** | CPU/GPU | GPU only |
| **Output** | 2D/3D keypoints | MANO + 3D joints + mesh |
| **Accuracy** | Good | Excellent |
| **Installation** | Easy (`pip install mediapipe`) | Complex (dependencies) |
| **Use Case** | Quick analysis, prototyping | Research, detailed analysis |

## Basic Usage

### MediaPipe (Recommended for beginners)

```bash
python utils/extract_hand_pose.py \
    --video test_data/output_results/youtube/VIDEO_ID/segment_0/video.mp4 \
    --output_dir test_data/output_results/youtube/VIDEO_ID/segment_0 \
    --model mediapipe
```

### HaMeR (For detailed MANO parameters)

```bash
python utils/extract_hand_pose.py \
    --video test_data/output_results/youtube/VIDEO_ID/segment_0/video.mp4 \
    --output_dir test_data/output_results/youtube/VIDEO_ID/segment_0 \
    --model hamer \
    --device cuda
```

## Arguments

### Required Arguments

- `--video`: Path to input video file
- `--output_dir`: Output directory for hand pose data

### Optional Arguments

- `--model`: Model to use (choices: `mediapipe`, `hamer`)
  - Default: `mediapipe`
- `--device`: Device to run on (choices: `cuda`, `cpu`)
  - Default: `cuda`
  - Note: HaMeR requires CUDA
- `--sample_rate`: Process every N frames
  - Default: `1` (process every frame)

### Example with All Options

```bash
python utils/extract_hand_pose.py \
    --video test_data/output_results/youtube/3f7p5iDQit0/segment_0/video.mp4 \
    --output_dir test_data/output_results/youtube/3f7p5iDQit0/segment_0 \
    --model hamer \
    --device cuda \
    --sample_rate 2
```

## Output Structure

```
segment_0/
├── video.mp4
├── segment_info.json
├── masks/
└── hand_pose/
    ├── video_hand_pose.json       # All hand pose data (JSON)
    ├── video_hand_pose.npz        # MANO parameters (HaMeR only)
    └── visualizations/
        ├── frame_0000.jpg         # Visualized hand keypoints
        ├── frame_0001.jpg
        └── ...
```

## Output Files

### 1. hand_pose.json

Complete hand pose data for all frames.

**Structure (MediaPipe)**:
```json
[
  {
    "frame_idx": 0,
    "timestamp": 0.0,
    "hands": [
      {
        "hand_id": 0,
        "handedness": "Right",
        "confidence": 0.98,
        "keypoints_2d": [
          [x0, y0], [x1, y1], ..., [x20, y20]
        ],
        "keypoints_3d": [
          [x0, y0, z0], ..., [x20, y20, z20]
        ]
      }
    ]
  }
]
```

**Structure (HaMeR)**:
```json
[
  {
    "frame_idx": 0,
    "timestamp": 0.0,
    "hands": [
      {
        "hand_id": 0,
        "handedness": "Right",
        "confidence": 0.95,
        "bbox": [x1, y1, x2, y2],
        "keypoints_2d": [...],
        "joints_3d": [...],
        "mano_params": {
          "shape": [...],        # 10 shape parameters
          "pose": [...],         # 48 pose parameters
          "orient": [...]        # 3 global orientation
        },
        "vertices": [...]        # 778 mesh vertices
      }
    ]
  }
]
```

### 2. hand_pose.npz (HaMeR only)

NumPy archive containing MANO parameters and 3D data.

**Keys**:
- `frame_{idx}_hand_{id}_mano_shape`: Shape parameters (10,)
- `frame_{idx}_hand_{id}_mano_pose`: Pose parameters (48,)
- `frame_{idx}_hand_{id}_mano_orient`: Global orientation (3,)
- `frame_{idx}_hand_{id}_joints_3d`: 3D joint positions (21, 3)
- `frame_{idx}_hand_{id}_vertices`: Mesh vertices (778, 3)

**Usage**:
```python
import numpy as np

data = np.load('hand_pose.npz')
frame_0_hand_0_joints = data['frame_0_hand_0_joints_3d']
frame_0_hand_0_shape = data['frame_0_hand_0_mano_shape']
```

### 3. Visualizations

Images with hand keypoints overlaid on frames.

- **MediaPipe**: 21 landmarks with connections
- **HaMeR**: Detected hands with bounding boxes and keypoints

## Model Details

### MediaPipe Hands

**Installation**:
```bash
pip install mediapipe
```

**Features**:
- 21 hand landmarks (0-20)
- Both 2D and 3D coordinates
- Multi-hand detection (left/right classification)
- Runs on CPU efficiently

**Landmark indices**:
```
0: Wrist
1-4: Thumb (CMC, MCP, IP, Tip)
5-8: Index (MCP, PIP, DIP, Tip)
9-12: Middle (MCP, PIP, DIP, Tip)
13-16: Ring (MCP, PIP, DIP, Tip)
17-20: Pinky (MCP, PIP, DIP, Tip)
```

**When to use**:
- Quick prototyping
- CPU-only environments
- Real-time processing
- 2D/3D keypoints sufficient

### HaMeR (Hand Mesh Recovery)

**Installation**:
```bash
pip install git+https://github.com/geopavlakos/hamer.git
```

**Features**:
- Full MANO hand model parameters
- 3D hand mesh reconstruction (778 vertices)
- 21 3D joint positions
- Hand-object interaction modeling

**When to use**:
- Research requiring MANO parameters
- 3D hand shape and pose analysis
- Hand mesh reconstruction
- Detailed biomechanical analysis

**Requirements**:
- GPU with CUDA
- ~4GB VRAM per hand
- ViTPose for hand detection (auto-downloaded)

## Advanced Usage

### Batch Processing

Process all segments in a video:

```bash
# Find all segments
find test_data/output_results/youtube/VIDEO_ID -name "segment_*" -type d | while read seg; do
    video_file=$(find "$seg" -name "*.mp4" | head -1)
    if [ -n "$video_file" ]; then
        python utils/extract_hand_pose.py \
            --video "$video_file" \
            --output_dir "$seg" \
            --model mediapipe
    fi
done
```

### Sample Every N Frames

For faster processing:

```bash
# Process every 5 frames
python utils/extract_hand_pose.py \
    --video test_data/output_results/youtube/VIDEO_ID/segment_0/video.mp4 \
    --output_dir test_data/output_results/youtube/VIDEO_ID/segment_0 \
    --model mediapipe \
    --sample_rate 5
```

### Use MediaPipe on CPU

```bash
python utils/extract_hand_pose.py \
    --video test_data/output_results/youtube/VIDEO_ID/segment_0/video.mp4 \
    --output_dir test_data/output_results/youtube/VIDEO_ID/segment_0 \
    --model mediapipe \
    --device cpu
```

### Extract with HaMeR (Full Pipeline)

```bash
# Requires CUDA GPU
python utils/extract_hand_pose.py \
    --video test_data/output_results/youtube/VIDEO_ID/segment_0/video.mp4 \
    --output_dir test_data/output_results/youtube/VIDEO_ID/segment_0 \
    --model hamer \
    --device cuda \
    --sample_rate 1
```

## Performance Tips

### 1. Model Selection

| Scenario | Recommended Model |
|----------|-------------------|
| Quick preview | MediaPipe + `sample_rate=5` |
| CPU-only | MediaPipe |
| Research (MANO needed) | HaMeR + GPU |
| Real-time | MediaPipe + CPU |

### 2. Sampling Strategy

| Sample Rate | Use Case | Speed Gain |
|-------------|----------|------------|
| 1 | High precision | 1x (baseline) |
| 2 | Balanced | 2x faster |
| 5 | Quick analysis | 5x faster |
| 10 | Preview only | 10x faster |

### 3. Memory Management

**MediaPipe**:
- Very low memory usage (~100MB)
- Can run on CPU

**HaMeR**:
- ~4GB VRAM per detected hand
- Monitor with `nvidia-smi`
- Process shorter segments if OOM

### 4. Parallel Processing

Process multiple videos in parallel:

```bash
# GPU 0 - Video 1
CUDA_VISIBLE_DEVICES=0 python utils/extract_hand_pose.py \
    --video test_data/output_results/youtube/VIDEO_1/segment_0/video.mp4 \
    --output_dir test_data/output_results/youtube/VIDEO_1/segment_0 \
    --model hamer &

# GPU 1 - Video 2
CUDA_VISIBLE_DEVICES=1 python utils/extract_hand_pose.py \
    --video test_data/output_results/youtube/VIDEO_2/segment_0/video.mp4 \
    --output_dir test_data/output_results/youtube/VIDEO_2/segment_0 \
    --model hamer &
```

## Troubleshooting

### MediaPipe Issues

#### Import Error

**Problem**: `ModuleNotFoundError: No module named 'mediapipe'`

**Solution**:
```bash
pip install mediapipe
```

#### No Hands Detected

**Problem**: No hands found in video

**Solutions**:
1. Verify video contains visible hands
2. Check video quality (blur, lighting)
3. Try different frames (use `--sample_rate 1`)

### HaMeR Issues

#### Installation Failed

**Problem**: HaMeR installation fails

**Solution**:
```bash
# Install dependencies first
pip install torch torchvision
pip install numpy opencv-python pillow

# Then install HaMeR
pip install git+https://github.com/geopavlakos/hamer.git
```

#### Model Download Failed

**Problem**: ViTPose or HaMeR checkpoints fail to download

**Solution**:
Models are cached in `~/.cache/hamer/`. If download fails:
1. Check internet connection
2. Ensure sufficient disk space (~500MB)
3. Try manual download from [HaMeR repository](https://github.com/geopavlakos/hamer)

#### CUDA Error

**Problem**: `RuntimeError: CUDA error` or `CUDA out of memory`

**Solutions**:
1. Verify CUDA is available:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Use smaller `--sample_rate`:
   ```bash
   --sample_rate 5
   ```
3. Process shorter videos
4. Switch to MediaPipe:
   ```bash
   --model mediapipe
   ```

### General Issues

#### Video File Not Found

**Problem**: `FileNotFoundError: Video not found`

**Solution**:
```bash
# Verify video exists
ls -lh test_data/output_results/youtube/VIDEO_ID/segment_0/*.mp4

# Check path is correct
realpath test_data/output_results/youtube/VIDEO_ID/segment_0/video.mp4
```

#### Slow Processing

**Problem**: Processing takes too long

**Solutions**:
1. Use MediaPipe instead of HaMeR:
   ```bash
   --model mediapipe
   ```
2. Increase sample rate:
   ```bash
   --sample_rate 5
   ```
3. Use GPU (for MediaPipe):
   ```bash
   --device cuda
   ```
4. Process shorter segments

#### Poor Keypoint Quality

**Problem**: Inaccurate hand keypoints

**Solutions (MediaPipe)**:
- Use higher resolution video
- Ensure good lighting in video
- Process every frame (`--sample_rate 1`)

**Solutions (HaMeR)**:
- Ensure GPU is being used
- Check hand is clearly visible
- Try different confidence thresholds (requires code modification)

## Working with Output Data

### Load JSON Data

```python
import json

with open('hand_pose.json', 'r') as f:
    data = json.load(f)

# Access first frame
frame_0 = data[0]
frame_idx = frame_0['frame_idx']
timestamp = frame_0['timestamp']

# Access hand data
for hand in frame_0['hands']:
    handedness = hand['handedness']  # 'Left' or 'Right'
    keypoints_2d = hand['keypoints_2d']  # List of [x, y]
    keypoints_3d = hand['keypoints_3d']  # List of [x, y, z]
```

### Load NPZ Data (HaMeR)

```python
import numpy as np

data = np.load('hand_pose.npz')

# List all keys
print(data.files)

# Load specific frame/hand
joints_3d = data['frame_0_hand_0_joints_3d']  # Shape: (21, 3)
mano_shape = data['frame_0_hand_0_mano_shape']  # Shape: (10,)
mano_pose = data['frame_0_hand_0_mano_pose']  # Shape: (48,)
vertices = data['frame_0_hand_0_vertices']  # Shape: (778, 3)
```

### Visualize Results

```python
import cv2
import json

# Load hand pose data
with open('hand_pose.json', 'r') as f:
    hand_data = json.load(f)

# Load video
cap = cv2.VideoCapture('video.mp4')

for frame_data in hand_data:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Draw keypoints
    for hand in frame_data['hands']:
        for x, y in hand['keypoints_2d']:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    cv2.imshow('Hand Pose', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Next Steps

After extracting hand pose:
1. Combine with segmentation masks for hand-object interaction analysis
2. Use MANO parameters for biomechanical modeling
3. Track hand trajectories across frames
4. Analyze tool manipulation patterns

## References

- **MediaPipe Hands**: [Google MediaPipe](https://google.github.io/mediapipe/solutions/hands.html)
- **HaMeR**: [Hand Mesh Recovery](https://github.com/geopavlakos/hamer)
- **MANO**: [MANO Hand Model](https://mano.is.tue.mpg.de/)
