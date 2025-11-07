# Setup Guide

This guide covers the complete installation and setup process for EgoTools.

## Table of Contents

- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Model Checkpoints](#model-checkpoints)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Minimum 8GB VRAM for Qwen2-VL-2B
  - 16GB+ VRAM for Qwen2-VL-7B
  - 24GB+ VRAM for SAM2 + Grounding DINO
- **Storage**: At least 50GB free space for models and datasets
- **RAM**: 16GB+ recommended

### Software

- **OS**: Linux (tested on Ubuntu 20.04+)
- **CUDA**: 12.0+ (for GPU acceleration)
- **Python**: 3.10 (managed by conda)

## Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/EgoTools.git
cd EgoTools

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate egotools
```

### Option 2: Manual Installation

```bash
# Create conda environment
conda create -n egotools python=3.10 -y
conda activate egotools

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install pandas tqdm yt-dlp opencv-python pillow

# Install transformers and related packages
pip install transformers accelerate qwen-vl-utils

# Install SAM2 (for segmentation)
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Install Grounding DINO dependencies
pip install supervision groundingdino-py

# Optional: Install hand pose extraction dependencies
# For MediaPipe (easier, CPU-friendly)
pip install mediapipe

# For HaMeR (more accurate, requires GPU)
pip install git+https://github.com/geopavlakos/hamer.git
```

## Model Checkpoints

EgoTools uses several pre-trained models. Download the required checkpoints before running the pipeline.

### 1. SAM2 Checkpoint

**Model**: Segment Anything 2 (SAM2) - for precise segmentation

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download SAM2 Hiera-Large checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
    -P checkpoints/

# Alternative: SAM2 Hiera-Tiny (smaller, faster)
# wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -P checkpoints/

# Alternative: SAM2 Hiera-Small
# wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt -P checkpoints/
```

**Checkpoint sizes**:
- `sam2_hiera_tiny.pt`: ~154MB (fastest, lower accuracy)
- `sam2_hiera_small.pt`: ~184MB (balanced)
- `sam2_hiera_large.pt`: ~899MB (best accuracy, recommended)

### 2. Grounding DINO Checkpoint

**Model**: Grounding DINO - for object detection with text prompts

```bash
# Download Grounding DINO SwinT-OGC checkpoint
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    -P checkpoints/
```

**Checkpoint size**: ~694MB

### 3. Qwen2-VL Models

**Model**: Qwen2-VL - for video understanding and analysis

These models are automatically downloaded from HuggingFace when first used.

**Available models**:
- `Qwen/Qwen2-VL-2B-Instruct`: ~8GB VRAM, faster inference
- `Qwen/Qwen2-VL-7B-Instruct`: ~16GB VRAM, better accuracy (default)

**Manual download** (optional):

```bash
# Install huggingface-cli
pip install huggingface_hub

# Download Qwen2-VL-7B
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct

# Or download Qwen2-VL-2B for lower memory usage
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct
```

### 4. Hand Pose Models (Optional)

#### MediaPipe Hands (Recommended for CPU)

MediaPipe models are automatically downloaded when first used. No manual setup required.

#### HaMeR (Recommended for GPU)

HaMeR models are automatically downloaded from the HaMeR repository when first used.

**Pre-download** (optional):

```bash
# Models will be cached in ~/.cache/hamer/
# First run will automatically download:
# - ViTPose model for hand detection
# - HaMeR checkpoint for hand mesh recovery
```

## Verification

Verify your installation by checking each component:

### 1. Check Environment

```bash
# Activate environment
conda activate egotools

# Check Python version
python --version  # Should show Python 3.10.x

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Check Model Checkpoints

```bash
# List downloaded checkpoints
ls -lh checkpoints/

# Should show:
# groundingdino_swint_ogc.pth (~694MB)
# sam2_hiera_large.pt (~899MB)
```

### 3. Test Imports

```bash
python -c "import transformers; print('✓ Transformers')"
python -c "import cv2; print('✓ OpenCV')"
python -c "import pandas; print('✓ Pandas')"
python -c "from sam2.sam2_video_predictor import SAM2VideoPredictor; print('✓ SAM2')"
python -c "import supervision; print('✓ Supervision')"
python -c "import groundingdino; print('✓ Grounding DINO')"
```

### 4. GPU Test

```bash
# Check GPU availability
nvidia-smi

# Test CUDA with PyTorch
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

## Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues

**Problem**: `CUDA not available` or `GPU not found`

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 2. SAM2 Import Error

**Problem**: `ModuleNotFoundError: No module named 'sam2'`

**Solution**:
```bash
# Reinstall SAM2
pip uninstall segment-anything-2
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

#### 3. Grounding DINO Import Error

**Problem**: `ModuleNotFoundError: No module named 'groundingdino'`

**Solution**:
```bash
# Install Grounding DINO
pip install groundingdino-py supervision
```

#### 4. Model Download Fails

**Problem**: HuggingFace model download fails or times out

**Solution**:
```bash
# Set HuggingFace mirror (China users)
export HF_ENDPOINT=https://hf-mirror.com

# Or use manual download
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir ./models/qwen2-vl-7b
```

#### 5. Out of Memory (OOM)

**Problem**: `CUDA out of memory` error

**Solutions**:
- Use smaller models (Qwen2-VL-2B instead of 7B)
- Reduce batch size or sample interval
- Close other GPU applications
- Use CPU for some tasks (MediaPipe instead of HaMeR)

#### 6. Checkpoint Not Found

**Problem**: `FileNotFoundError: checkpoint not found`

**Solution**:
```bash
# Re-download checkpoints
cd checkpoints/

# SAM2
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# Grounding DINO
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Storage Requirements

Expected storage usage:

- **Conda environment**: ~10GB
- **Model checkpoints**: 
  - SAM2: ~900MB
  - Grounding DINO: ~700MB
  - Qwen2-VL-7B: ~14GB (cached in `~/.cache/huggingface/`)
  - HaMeR (optional): ~500MB (cached in `~/.cache/hamer/`)
- **Datasets** (depends on usage):
  - Video downloads: Variable (GB per video)
  - Segmentation masks: ~100MB per video
  - Hand pose data: ~10MB per video

**Total**: ~30GB minimum + dataset storage

## Next Steps

After completing setup:

1. Proceed to [Video Filtering Guide](VIDEO_FILTERING.md) to analyze videos
2. See [Segmentation Guide](SEGMENTATION.md) to segment hands and tools
3. See [Hand Pose Guide](HAND_POSE.md) to extract hand pose data

## Updates

To update the environment:

```bash
# Update conda environment
conda activate egotools
conda env update -f environment.yml

# Update specific packages
pip install --upgrade transformers
pip install --upgrade yt-dlp

# Update SAM2
pip install --upgrade git+https://github.com/facebookresearch/segment-anything-2.git
```
