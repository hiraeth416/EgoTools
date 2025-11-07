# Video Filtering Guide

This guide covers using Qwen2-VL vision-language model to analyze videos and identify first-person perspective videos with tool usage.

## Table of Contents

- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [Arguments](#arguments)
- [Output Structure](#output-structure)
- [Model Selection](#model-selection)
- [Advanced Usage](#advanced-usage)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Overview

The video filtering step uses Qwen2-VL to:
- Identify first-person (egocentric) perspective videos
- Detect tools being used in videos
- Identify presence of human hands
- Recognize activities and actions
- Extract relevant video segments showing hand-tool interactions

## Basic Usage

### Quick Start

```bash
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --save_clips
```

### Process Specific Platform

```bash
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --platforms youtube \
    --save_clips
```

## Arguments

### Required Arguments

- `--csv_path`: Path to metadata CSV file from video search
- `--video_dir`: Directory containing downloaded videos
- `--output_dir`: Output directory for analysis results

### Optional Arguments

- `--model_name`: Qwen model name or path (default: `Qwen/Qwen2-VL-7B-Instruct`)
- `--platforms`: Comma-separated list of platforms to process (e.g., `youtube,bilibili`)
- `--save_clips`: Flag to save video segments where hands and tools appear together
- `--sample_interval`: Sample one frame every N frames (default: 30)
- `--start_index`: Start processing from video index (default: 0)
- `--max_videos`: Maximum number of videos to process

### Example with All Options

```bash
python utils/filter_videos.py \
    --csv_path datasets/20251107_120000/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --platforms youtube \
    --sample_interval 15 \
    --save_clips \
    --start_index 0 \
    --max_videos 50
```

## Output Structure

```
test_data/output_results/
├── analysis_summary.csv                      # Summary of all videos
├── filtered_first_person_tool_videos.csv     # Qualified videos only
└── youtube/
    └── VIDEO_ID/
        ├── analysis.json                     # Detailed analysis
        ├── segment_0/
        │   ├── segment_info.json             # Segment metadata
        │   └── VIDEO_NAME_segment_0.mp4      # Extracted segment
        └── segment_1/
            ├── segment_info.json
            └── VIDEO_NAME_segment_1.mp4
```

## Output Files

### 1. analysis_summary.csv

Summary statistics for all processed videos.

**Columns**:
- `video_id`: Unique video identifier
- `platform`: Video platform (youtube, bilibili, tiktok)
- `title`: Video title
- `is_first_person`: Boolean indicating first-person perspective
- `has_hand`: Boolean indicating hand detection
- `has_tool_usage`: Boolean indicating tool usage
- `tools`: Comma-separated list of detected tools
- `activities`: Comma-separated list of detected activities
- `num_segments`: Number of valid segments extracted
- `total_segment_duration`: Total duration of extracted segments (seconds)
- `first_person_ratio`: Ratio of frames detected as first-person
- `hand_ratio`: Ratio of frames with hands detected
- `tool_ratio`: Ratio of frames with tools detected

### 2. filtered_first_person_tool_videos.csv

Contains only videos that meet criteria:
- Is first-person perspective
- Contains tool usage
- Contains human hands

Same columns as `analysis_summary.csv`.

### 3. analysis.json

Detailed per-video analysis.

**Structure**:
```json
{
  "video_id": "VIDEO_ID",
  "platform": "youtube",
  "title": "Video Title",
  "video_path": "path/to/video.mp4",
  "fps": 30.0,
  "duration": 120.5,
  "total_frames": 3615,
  "is_first_person": true,
  "has_tool_usage": true,
  "has_hand": true,
  "tools": ["knife", "cutting board"],
  "activities": ["chopping", "slicing vegetables"],
  "description": "First-person view of cooking...",
  "valid_segments": [
    {
      "segment_id": 0,
      "start_frame": 0,
      "end_frame": 150,
      "start_time": 0.0,
      "end_time": 5.0,
      "duration": 5.0,
      "description": "Using knife on cutting board..."
    }
  ]
}
```

### 4. segment_info.json

Metadata for each extracted segment.

**Structure**:
```json
{
  "segment_id": 0,
  "start_frame": 0,
  "end_frame": 150,
  "start_time": 0.0,
  "end_time": 5.0,
  "duration": 5.0,
  "fps": 30.0,
  "frame_analyses": [
    {
      "frame_idx": 0,
      "timestamp": 0.0,
      "is_first_person": true,
      "has_hand": true,
      "has_tool": true,
      "tools": ["knife"],
      "description": "Hand holding knife..."
    }
  ]
}
```

## Model Selection

### Available Models

#### Qwen2-VL-7B-Instruct (Default)

**Best for**: High accuracy, detailed analysis

```bash
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --save_clips
```

**Requirements**:
- GPU: 16GB+ VRAM
- Speed: ~30 seconds per video (720p, 30fps)

#### Qwen2-VL-2B-Instruct

**Best for**: Faster processing, lower memory usage

```bash
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --save_clips
```

**Requirements**:
- GPU: 8GB+ VRAM
- Speed: ~15 seconds per video (720p, 30fps)

### Using Local Model

If you've downloaded a model locally:

```bash
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --model_name /path/to/local/qwen2-vl-7b \
    --save_clips
```

## Advanced Usage

### Batch Processing

Process videos in batches to manage memory:

```bash
# Process first 10 videos
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --start_index 0 \
    --max_videos 10 \
    --save_clips

# Process next 10 videos
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --start_index 10 \
    --max_videos 10 \
    --save_clips
```

### Adjust Sampling Rate

Control frame sampling for speed vs. accuracy:

```bash
# Sample every 60 frames (~2 seconds at 30fps) - faster
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --sample_interval 60 \
    --save_clips

# Sample every 15 frames (~0.5 seconds at 30fps) - more accurate
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --sample_interval 15 \
    --save_clips
```

### Process Specific Platform Only

```bash
# Process only YouTube videos
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --platforms youtube \
    --save_clips

# Process YouTube and Bilibili
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --platforms youtube,bilibili \
    --save_clips
```

### Resume Interrupted Processing

The script automatically skips already processed videos:

```bash
# If processing was interrupted, just run the same command again
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --save_clips
```

## Performance Tips

### 1. Optimize Sampling Rate

- **Fast preview** (sample_interval=60): ~10 sec/video, may miss details
- **Balanced** (sample_interval=30, default): ~20 sec/video, good accuracy
- **High accuracy** (sample_interval=15): ~40 sec/video, best results

### 2. Model Selection Strategy

- **Development/testing**: Use Qwen2-VL-2B-Instruct
- **Production/research**: Use Qwen2-VL-7B-Instruct
- **Limited GPU**: Use Qwen2-VL-2B-Instruct with higher sample_interval

### 3. Memory Management

```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# If OOM errors occur:
# 1. Use smaller model (2B instead of 7B)
# 2. Increase sample_interval (60 instead of 30)
# 3. Process fewer videos at once (use --max_videos)
# 4. Close other GPU applications
```

### 4. Parallel Processing

To process multiple videos in parallel (if you have multiple GPUs):

```bash
# GPU 0 - first half
CUDA_VISIBLE_DEVICES=0 python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --start_index 0 \
    --max_videos 50 \
    --save_clips &

# GPU 1 - second half
CUDA_VISIBLE_DEVICES=1 python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --start_index 50 \
    --max_videos 50 \
    --save_clips &
```

## Troubleshooting

### Out of Memory (OOM)

**Problem**: `CUDA out of memory` error

**Solutions**:
1. Use smaller model:
   ```bash
   --model_name Qwen/Qwen2-VL-2B-Instruct
   ```
2. Increase sampling interval:
   ```bash
   --sample_interval 60
   ```
3. Process fewer videos:
   ```bash
   --max_videos 10
   ```

### Model Download Fails

**Problem**: HuggingFace download timeout or connection error

**Solutions**:
1. Check internet connection
2. Set HuggingFace mirror (for China users):
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. Download manually:
   ```bash
   huggingface-cli download Qwen/Qwen2-VL-7B-Instruct
   ```

### Video Not Found

**Problem**: `FileNotFoundError: Video not found`

**Solutions**:
1. Check video directory structure:
   ```
   test_data/download/
   └── youtube/
       └── VIDEO_ID/
           └── VIDEO_TITLE-VIDEO_ID.mp4
   ```
2. Verify video was downloaded successfully
3. Check platform name matches (youtube, bilibili, tiktok)

### Slow Processing

**Problem**: Processing takes too long

**Solutions**:
1. Increase `--sample_interval` to 60 or higher
2. Use smaller model (Qwen2-VL-2B)
3. Reduce video resolution during download:
   ```bash
   --max-height 720  # in download_videos.py
   ```

### No Segments Extracted

**Problem**: Videos analyzed but no segments saved

**Solutions**:
1. Ensure `--save_clips` flag is set
2. Check if videos actually contain hand-tool interactions
3. Lower the detection threshold (requires code modification)
4. Reduce `--sample_interval` for more detailed analysis

## Analysis Quality

### What Makes a "Good" First-Person Tool Video?

The model looks for:
- **First-person view**: Camera appears to be from user's perspective
- **Visible hands**: User's hands are visible in frame
- **Tool usage**: Clear interaction with tools
- **Sustained activity**: Hand-tool interaction lasts multiple frames

### Common False Positives

- Third-person POV mistaken as first-person
- Background hands/tools without actual usage
- Brief tool appearances in transitions

### Improving Accuracy

1. Use lower `sample_interval` (15-30 frames)
2. Use larger model (Qwen2-VL-7B)
3. Filter results by `num_segments` > 0 and `total_segment_duration` > threshold
4. Manually review `filtered_first_person_tool_videos.csv`

## Next Steps

After filtering videos:
1. Review `filtered_first_person_tool_videos.csv` for qualified videos
2. Proceed to [Segmentation Guide](SEGMENTATION.md) to segment hands and tools
3. Extract hand pose using [Hand Pose Guide](HAND_POSE.md)
