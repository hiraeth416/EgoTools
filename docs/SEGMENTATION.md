# Segmentation Guide

This guide covers using Grounding DINO + SAM2 to segment hands and tools from video segments.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Basic Usage](#basic-usage)
- [Segmentation Modes](#segmentation-modes)
- [Arguments](#arguments)
- [Output Structure](#output-structure)
- [Advanced Usage](#advanced-usage)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Overview

The segmentation step uses:
- **Grounding DINO**: Detects objects (hands, tools) using text prompts
- **SAM2**: Generates precise segmentation masks from detected bounding boxes

This combination provides:
- Text-based object detection ("hand", "tool", specific tool names)
- Frame-by-frame segmentation tracking
- High-quality masks for hands and tools

## Prerequisites

### Model Checkpoints

Download required checkpoints (see [Setup Guide](SETUP.md)):

```bash
# SAM2 checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
    -P checkpoints/

# Grounding DINO checkpoint
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    -P checkpoints/
```

### Input Data

Segmentation requires video segments from the filtering step:

```
test_data/output_results/
└── youtube/
    └── VIDEO_ID/
        └── segment_0/
            ├── segment_info.json
            └── VIDEO_NAME_segment_0.mp4
```

## Basic Usage

### Process All Segments

```bash
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --mode grounding_dino
```

This will process all video segments found in the output directory.

### Process Single Segment

```bash
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --single-segment youtube/VIDEO_ID/segment_0 \
    --mode grounding_dino
```

## Segmentation Modes

### 1. Grounding DINO Mode (Recommended)

Uses text prompts to detect specific objects.

```bash
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --mode grounding_dino \
    --sample-interval 5
```

**Advantages**:
- Semantic understanding ("hand", "knife", "hammer")
- Detects specific tools by name
- Best accuracy for hand-tool separation

**Text prompts used**:
- "hand"
- Tools from segment metadata (knife, hammer, screwdriver, etc.)

### 2. Auto Mode

Automatically generates masks without text prompts.

```bash
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --mode auto \
    --max-masks 3
```

**Advantages**:
- No pre-defined object categories needed
- May detect unexpected objects

**Disadvantages**:
- Cannot distinguish between object types
- May include background objects

### 3. Manual Mode

Requires manual annotation (interactive).

```bash
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --mode manual
```

**Use case**: Fine-tuning specific segments with manual clicks.

## Arguments

### Required Arguments

- `--output-dir`: Directory containing segmented videos (from filter step)

### Optional Arguments

- `--mode`: Segmentation mode (choices: `grounding_dino`, `auto`, `manual`)
  - Default: `grounding_dino`
- `--model-name`: SAM2 model name
  - Default: `facebook/sam2-hiera-large`
  - Options: `facebook/sam2-hiera-tiny`, `facebook/sam2-hiera-small`
- `--checkpoint`: Local SAM2 checkpoint path
  - Default: `checkpoints/sam2_hiera_large.pt`
- `--sample-interval`: Sample every N frames
  - Default: `5` (process every 5th frame)
- `--max-masks`: Maximum masks per frame (auto mode only)
  - Default: `3`
- `--single-segment`: Process only one segment
  - Format: `platform/video_id/segment_N`

### Example with All Options

```bash
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --mode grounding_dino \
    --model-name facebook/sam2-hiera-large \
    --checkpoint checkpoints/sam2_hiera_large.pt \
    --sample-interval 5 \
    --single-segment youtube/3f7p5iDQit0/segment_0
```

## Output Structure

```
test_data/output_results/
└── youtube/
    └── VIDEO_ID/
        └── segment_0/
            ├── segment_info.json
            ├── VIDEO_NAME_segment_0.mp4
            └── masks/
                ├── segmentation_metadata.json    # Overall metadata
                ├── masks_details.json            # Per-frame mask info
                └── frame_0000/
                    ├── hand_0.png                # Hand mask
                    ├── hand_1.png                # Another hand mask
                    ├── tool_knife_0.png          # Tool mask
                    └── metadata.json             # Frame metadata
```

## Output Files

### 1. segmentation_metadata.json

Overall segmentation configuration and statistics.

**Structure**:
```json
{
  "video_path": "path/to/segment.mp4",
  "fps": 30.0,
  "total_frames": 150,
  "sample_interval": 5,
  "frames_processed": 30,
  "mode": "grounding_dino",
  "text_prompts": ["hand", "knife"],
  "timestamp": "2025-11-07T12:00:00"
}
```

### 2. masks_details.json

Per-frame mask information.

**Structure**:
```json
{
  "frames": [
    {
      "frame_idx": 0,
      "timestamp": 0.0,
      "masks": [
        {
          "mask_id": "hand_0",
          "category": "hand",
          "bbox": [100, 150, 200, 250],
          "area": 10000,
          "confidence": 0.95
        },
        {
          "mask_id": "tool_knife_0",
          "category": "knife",
          "bbox": [180, 200, 220, 280],
          "area": 3200,
          "confidence": 0.89
        }
      ]
    }
  ]
}
```

### 3. Frame Metadata (metadata.json)

Per-frame detailed information.

**Structure**:
```json
{
  "frame_idx": 0,
  "timestamp": 0.0,
  "detections": [
    {
      "category": "hand",
      "bbox": [100, 150, 200, 250],
      "confidence": 0.95,
      "mask_file": "hand_0.png"
    }
  ]
}
```

### 4. Mask Images

PNG images with:
- **White (255)**: Segmented object
- **Black (0)**: Background

**Naming convention**:
- `hand_0.png`, `hand_1.png`: Hand masks
- `tool_<name>_0.png`: Tool masks (e.g., `tool_knife_0.png`)

## Advanced Usage

### Use Different SAM2 Models

#### SAM2-Hiera-Tiny (Fastest)

```bash
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --mode grounding_dino \
    --model-name facebook/sam2-hiera-tiny \
    --checkpoint checkpoints/sam2_hiera_tiny.pt
```

**Best for**: Quick previews, limited GPU memory

#### SAM2-Hiera-Large (Best Quality)

```bash
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --mode grounding_dino \
    --model-name facebook/sam2-hiera-large \
    --checkpoint checkpoints/sam2_hiera_large.pt
```

**Best for**: Production, research

### Adjust Sampling Rate

```bash
# Process every frame (slow, high quality)
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --mode grounding_dino \
    --sample-interval 1

# Process every 10 frames (fast, lower quality)
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --mode grounding_dino \
    --sample-interval 10
```

### Process Specific Segments

```bash
# Process single segment
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --single-segment youtube/VIDEO_ID/segment_0 \
    --mode grounding_dino

# Process multiple specific segments (shell loop)
for seg in youtube/VIDEO_ID_1/segment_0 youtube/VIDEO_ID_2/segment_0; do
    python utils/segment_masks.py \
        --output-dir test_data/output_results \
        --single-segment $seg \
        --mode grounding_dino
done
```

### Batch Processing by Platform

```bash
# Process only YouTube segments
find test_data/output_results/youtube -name "segment_*" -type d | while read seg; do
    rel_path=$(realpath --relative-to=test_data/output_results "$seg")
    python utils/segment_masks.py \
        --output-dir test_data/output_results \
        --single-segment "$rel_path" \
        --mode grounding_dino
done
```

## Performance Tips

### 1. Model Selection

| Model | Speed | Quality | VRAM |
|-------|-------|---------|------|
| sam2-hiera-tiny | Fast | Good | ~4GB |
| sam2-hiera-small | Medium | Better | ~6GB |
| sam2-hiera-large | Slow | Best | ~8GB |

### 2. Sampling Strategy

| Interval | Use Case | Processing Time |
|----------|----------|-----------------|
| 1 | High precision, research | 100% (baseline) |
| 5 | Balanced (default) | ~20% |
| 10 | Quick preview | ~10% |
| 15 | Very fast | ~7% |

### 3. Memory Optimization

```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# If memory issues:
# 1. Use smaller model
--model-name facebook/sam2-hiera-tiny

# 2. Increase sample interval
--sample-interval 10

# 3. Process segments one by one
--single-segment youtube/VIDEO_ID/segment_0
```

### 4. Parallel Processing

Process multiple segments in parallel (if you have multiple GPUs):

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --single-segment youtube/VIDEO_1/segment_0 \
    --mode grounding_dino &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --single-segment youtube/VIDEO_2/segment_0 \
    --mode grounding_dino &
```

## Troubleshooting

### Checkpoint Not Found

**Problem**: `FileNotFoundError: Checkpoint not found`

**Solution**:
```bash
# Download SAM2 checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
    -P checkpoints/

# Download Grounding DINO checkpoint
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    -P checkpoints/
```

### Grounding DINO Import Error

**Problem**: `ModuleNotFoundError: No module named 'groundingdino'`

**Solution**:
```bash
pip install groundingdino-py supervision
```

### SAM2 Import Error

**Problem**: `ModuleNotFoundError: No module named 'sam2'`

**Solution**:
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Out of Memory (OOM)

**Problem**: `CUDA out of memory`

**Solutions**:
1. Use smaller model:
   ```bash
   --model-name facebook/sam2-hiera-tiny
   ```
2. Increase sample interval:
   ```bash
   --sample-interval 10
   ```
3. Process one segment at a time:
   ```bash
   --single-segment youtube/VIDEO_ID/segment_0
   ```

### No Masks Generated

**Problem**: No masks saved in output directory

**Possible causes**:
1. Grounding DINO checkpoint not found
2. No objects detected (low confidence threshold)
3. Video file corrupted or empty

**Solutions**:
1. Verify checkpoint exists:
   ```bash
   ls -lh checkpoints/groundingdino_swint_ogc.pth
   ```
2. Check video file:
   ```bash
   ffprobe test_data/output_results/youtube/VIDEO_ID/segment_0/*.mp4
   ```
3. Try auto mode to verify SAM2 works:
   ```bash
   --mode auto --max-masks 3
   ```

### Slow Processing

**Problem**: Segmentation takes too long

**Solutions**:
1. Increase `--sample-interval`:
   ```bash
   --sample-interval 10  # Process every 10th frame
   ```
2. Use smaller model:
   ```bash
   --model-name facebook/sam2-hiera-tiny
   ```
3. Use GPU (verify):
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Poor Mask Quality

**Problem**: Masks are inaccurate or incomplete

**Solutions**:
1. Use larger model:
   ```bash
   --model-name facebook/sam2-hiera-large
   ```
2. Decrease sample interval:
   ```bash
   --sample-interval 1  # Process every frame
   ```
3. Verify text prompts match objects in video (check segment_info.json)
4. Try manual mode for specific frames

## Understanding Mask Quality

### Good Segmentation Indicators

- Masks align with object boundaries
- Consistent across consecutive frames
- Clear separation between hands and tools
- Minimal background noise

### Common Issues

1. **Overlapping masks**: Hands holding tools may merge
2. **Missing detections**: Objects outside camera view or occluded
3. **Background objects**: Non-relevant objects detected (use grounding_dino mode)
4. **Temporal inconsistency**: Masks vary significantly between frames (lower sample_interval)

## Next Steps

After segmentation:
1. Review mask quality in `test_data/output_results/*/masks/`
2. Use masks for downstream tasks (hand-object interaction analysis)
3. Extract hand pose using [Hand Pose Guide](HAND_POSE.md)
4. Combine masks with hand pose for complete analysis
