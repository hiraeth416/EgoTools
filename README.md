# EgoTools

A comprehensive toolkit for collecting, downloading, and analyzing first-person (egocentric) tool usage video datasets.

## Overview

EgoTools provides a complete pipeline for processing first-person tool usage videos:
1. **Search** videos on multiple platforms (YouTube, Bilibili, TikTok)
2. **Download** videos with metadata
3. **Filter** videos using Qwen2-VL to identify first-person tool usage
4. **Segment** hands and tools using SAM2 + Grounding DINO
5. **Extract** hand pose (MANO parameters and joints)

## Quick Start

### Installation

See [Setup Guide](docs/SETUP.md) for detailed installation instructions.

```bash
# Create conda environment
conda env create -f environment.yml
conda activate egotools

# Download model checkpoints (SAM2, Grounding DINO)
# See docs/SETUP.md for details
```

### Basic Workflow

```bash
# 1. Search for videos
python utils/search_videos.py \
    --output datasets \
    --max_results 100 \
    --platforms youtube

# 2. Download videos
python utils/download_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --output test_data/download \
    --platforms youtube \
    --max-videos 10

# 3. Filter and analyze videos
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --save_clips

# 4. Segment hands and tools
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --mode grounding_dino

# 5. Extract hand pose
python utils/extract_hand_pose.py \
    --video test_data/output_results/youtube/VIDEO_ID/segment_0/video.mp4 \
    --output_dir test_data/output_results/youtube/VIDEO_ID/segment_0 \
    --model mediapipe
```

## Pipeline Steps

### 1. Video Search

Search and collect video metadata from multiple platforms.

```bash
python utils/search_videos.py \
    --output datasets \
    --max_results 1000 \
    --platforms youtube,bilibili
```

**Output**: Creates timestamped directory with `metadata.csv` containing video information.

---

### 2. Video Download

Download videos based on the metadata CSV.

```bash
python utils/download_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --output test_data/download \
    --platforms youtube \
    --max-videos 50
```

**Output**: Videos organized by platform and ID in `test_data/download/`.

---

### 3. Video Filtering and Analysis

Analyze videos using Qwen2-VL to identify first-person tool usage and extract relevant segments.

```bash
python utils/filter_videos.py \
    --csv_path datasets/TIMESTAMP/metadata.csv \
    --video_dir test_data/download \
    --output_dir test_data/output_results \
    --save_clips
```

**Output**: Filtered segments in `test_data/output_results/` with analysis results.

📖 **See [Video Filtering Guide](docs/VIDEO_FILTERING.md) for detailed usage.**

---

### 4. Hand and Tool Segmentation

Segment hands and tools using SAM2 + Grounding DINO.

```bash
python utils/segment_masks.py \
    --output-dir test_data/output_results \
    --mode grounding_dino \
    --sample-interval 5
```

**Output**: Frame-by-frame segmentation masks in `test_data/output_results/*/masks/`.

📖 **See [Segmentation Guide](docs/SEGMENTATION.md) for model setup and detailed usage.**

---

### 5. Hand Pose Extraction

Extract hand pose (MANO parameters and joints) from segmented videos.

```bash
python utils/extract_hand_pose.py \
    --video test_data/output_results/youtube/VIDEO_ID/segment_0/video.mp4 \
    --output_dir test_data/output_results/youtube/VIDEO_ID/segment_0 \
    --model mediapipe
```

**Output**: Hand pose data (JSON and NPZ) in the segment directory.

📖 **See [Hand Pose Extraction Guide](docs/HAND_POSE.md) for model options and detailed usage.**

## Documentation

- 📦 [**Setup Guide**](docs/SETUP.md) - Installation and model checkpoint downloads
- 🎥 [**Video Filtering**](docs/VIDEO_FILTERING.md) - Qwen2-VL analysis details
- ✂️ [**Segmentation**](docs/SEGMENTATION.md) - SAM2 + Grounding DINO usage
- 🖐️ [**Hand Pose Extraction**](docs/HAND_POSE.md) - HaMeR and MediaPipe usage

## Project Structure

```
EgoTools/
├── utils/
│   ├── search_videos.py       # Video search
│   ├── download_videos.py     # Video download
│   ├── filter_videos.py       # Video filtering (Qwen2-VL)
│   ├── segment_masks.py       # Segmentation (SAM2 + Grounding DINO)
│   └── extract_hand_pose.py   # Hand pose extraction (HaMeR/MediaPipe)
├── docs/                      # Detailed documentation
├── checkpoints/               # Model checkpoints
├── datasets/                  # Search results
└── test_data/                 # Downloaded videos and results
```

## License

TBD

## Citation

If you use EgoTools in your research, please cite:

```
TBD
```

