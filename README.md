# EgoTools

A comprehensive toolkit for collecting, downloading, and analyzing first-person (egocentric) tool usage video datasets.

## Overview

EgoTools provides a complete pipeline for:
1. **Searching** videos on multiple platforms (YouTube, Bilibili, TikTok)
2. **Downloading** videos with metadata
3. **Filtering and analyzing** videos using Qwen2-VL vision-language model to identify first-person perspective videos with tool usage

## Features

### Video Search and Collection
- Multi-platform support (YouTube, Bilibili, TikTok)
- Automatic deduplication
- Real-time progress tracking
- Incremental data saving
- Detailed duration statistics

### Video Download
- Parallel downloading with configurable workers
- Platform-specific directory organization
- Video quality control (configurable max height)
- Automatic metadata and thumbnail extraction
- Each video saved in separate folder by ID

### Intelligent Video Analysis (Qwen3-VL)
- **First-Person Detection**: Identifies egocentric perspective videos
- **Tool Detection**: Detects tools being used in videos  
- **Hand Detection**: Identifies presence of human hands
- **Activity Recognition**: Describes activities and actions being performed
- **Tool Identification**: Lists specific tools detected in the video
- **Segment Extraction**: Extracts clips where hands and tools appear together
- **Whole Video Analysis**: Analyzes entire videos using advanced vision-language model

## Installation

### Requirements

```bash
# Core dependencies
pip install pandas tqdm yt-dlp

# For video filtering and analysis (Qwen3-VL)
pip install torch transformers opencv-python pillow
```

**Note**: Video analysis requires a GPU with sufficient VRAM:
- Qwen2-VL-2B: ~8GB VRAM (recommended for testing)
- Qwen2-VL-7B: ~16GB VRAM (good balance)
- Qwen3-VL models: Check model card for requirements

## Quick Start

### Complete Workflow

```bash
# 1. Search for videos
python search_videos.py \
    --output datasets \
    --max_results 100 \
    --platforms youtube

# 2. Download videos
python download_videos.py \
    --csv_path datasets/20251105_094846/metadata.csv \
    --output download \
    --platforms youtube \
    --max-videos 10

# 3. Filter and analyze videos
python filter_videos.py \
    --csv_path datasets/20251105_094846/metadata.csv \
    --video_dir download \
    --output_dir filtered_videos \
    --save_clips
```

## Detailed Usage

### 1. Video Search (`search_videos.py`)

Search and collect video metadata from multiple platforms.

```bash
python search_videos.py \
    --output datasets \
    --max_results 1000 \
    --platforms youtube,bilibili,tiktok
```

**Arguments:**
- `--output`: Output directory (default: `datasets`)
- `--max_results`: Maximum results per keyword (default: 1000)
- `--platforms`: Platforms to search, comma-separated (default: `youtube`)

**Output:**
Creates a timestamped directory containing:
- `metadata.csv`: Video metadata including ID, title, description, duration, platform, search keyword

**Search Keywords:**
- POV tool usage
- egocentric tool usage
- first person view DIY
- POV crafting tutorial
- first person woodworking
- POV cooking techniques
- And more...

### 2. Video Download (`download_videos.py`)

Download videos based on the metadata CSV.

```bash
python download_videos.py \
    --csv_path datasets/20251105_094846/metadata.csv \
    --output download \
    --max-workers 4 \
    --max-height 1080 \
    --platforms youtube \
    --max-videos 50
```

**Arguments:**
- `--csv_path`: Path to metadata.csv file (required)
- `--output`: Video save directory (default: `downloads`)
- `--max-workers`: Number of parallel download threads (default: 4)
- `--max-height`: Maximum video height (default: 1080)
- `--platforms`: Platforms to download, comma-separated (e.g., `youtube,bilibili`)
- `--start-index`: Starting video index (default: 0)
- `--max-videos`: Maximum number of videos to download

**Output Structure:**
```
download/
└── youtube/
    └── VIDEO_ID/
        ├── VIDEO_TITLE-VIDEO_ID.mp4
        ├── VIDEO_TITLE-VIDEO_ID.info.json
        └── VIDEO_TITLE-VIDEO_ID.webp (thumbnail)
```

### 3. Video Filtering and Analysis (`filter_videos.py`)

Analyze videos using Qwen3-VL model to identify first-person tool usage and extract relevant segments.

```bash
python filter_videos.py \
    --csv_path datasets/20251105_094846/metadata.csv \
    --video_dir download \
    --output_dir filtered_videos \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --save_clips
```

**Arguments:**
- `--csv_path`: Path to metadata.csv file (required)
- `--video_dir`: Directory containing downloaded videos (required)
- `--output_dir`: Output directory for analysis results (required)
- `--model_name`: Qwen model name or path (default: `Qwen/Qwen2-VL-7B-Instruct`)
- `--save_clips`: Save video segments where hands and tools appear together

**Key Improvements with Qwen3-VL:**
- Analyzes entire videos instead of sampling frames
- Better understanding of temporal context
- More accurate tool and activity recognition
- Natural language descriptions of what's happening

**Output Structure:**
```
filtered_videos/
├── analysis_summary.csv                    # Summary of all videos
├── filtered_first_person_tool_videos.csv   # First-person videos with tools
└── youtube/
    └── VIDEO_ID/
        ├── analysis_result.json            # Detailed analysis
        ├── VIDEO_segment_0_0.0s-15.5s.mp4  # Extracted segment
        └── VIDEO_segment_0_0.0s-15.5s_metadata.json
```

**Analysis Output Files:**

1. **analysis_summary.csv**: Summary statistics for all videos
   - `video_id`, `platform`, `title`
   - `is_first_person`: Boolean for first-person detection
   - `has_hand`: Boolean for hand detection
   - `has_tool_usage`: Boolean for tool detection
   - `tools`: Comma-separated list of detected tools
   - `activities`: Comma-separated list of detected activities
   - `num_segments`: Number of valid segments
   - `total_segment_duration`: Total duration of segments

2. **analysis_result.json**: Detailed per-video analysis
   - Video metadata (fps, duration, etc.)
   - Overall classification (is_first_person, has_tool_usage, has_hand)
   - Tools detected: List of specific tools identified
   - Activities: List of actions being performed
   - Description: Natural language summary of video content
   - Valid segments with timestamps and descriptions

3. **Segment metadata JSON**: For each extracted segment
   - Start/end frame and timestamp
   - Duration
   - Frame analyses (tool names, actions, descriptions)

## Advanced Usage

### Custom Model Selection

### Custom Model Selection

Use a smaller model for faster processing:

```bash
python filter_videos.py \
    --csv_path metadata.csv \
    --video_dir download \
    --output_dir filtered_videos \
    --model_name Qwen/Qwen2-VL-2B-Instruct
```

### Batch Processing with Partial Downloads

Download and process videos in batches:

```bash
# Download first 10 videos
python download_videos.py \
    --csv_path metadata.csv \
    --output download \
    --start-index 0 \
    --max-videos 10

# Download next 10 videos
python download_videos.py \
    --csv_path metadata.csv \
    --output download \
    --start-index 10 \
    --max-videos 10
```

## Performance Tips

1. **Search**: Use specific keywords to reduce noise
2. **Download**: 
   - Adjust `--max-workers` based on network bandwidth
   - Use `--max-height 720` for faster downloads
3. **Analysis**:
   - Use smaller models (`Qwen2-VL-2B-Instruct`) for faster processing
   - Monitor GPU memory usage
   - Process videos in batches to avoid memory issues

## Troubleshooting

### Common Issues

**Search/Download:**
- **No results found**: Check internet connection and platform accessibility
- **Download fails**: Verify `yt-dlp` is up to date: `pip install -U yt-dlp`
- **Cookie errors**: Update browser cookies or use `--cookies` flag

**Video Analysis:**
- **Out of Memory**: Use a smaller model (e.g., `Qwen2-VL-2B-Instruct`)
- **Slow processing**: Use more powerful GPU or smaller model
- **Model download fails**: Ensure Hugging Face access and sufficient disk space
- **Video not found**: Check directory structure: `video_dir/platform/video_id/video.mp4`
- **Video format issues**: Ensure videos are in supported formats (.mp4, .webm, .mkv)

## Project Structure

```
EgoTools/
├── search_videos.py      # Video metadata collection
├── download_videos.py    # Video downloader
├── filter_videos.py      # Video analysis with Qwen3-VL
├── test_qwen3.py        # Test script for Qwen3-VL
├── README.md            # This file
├── datasets/            # Search results (timestamped)
├── download/            # Downloaded videos
└── filtered_videos/     # Analysis results and segments
```

## License

TBD

## Contributing

TBD

## Citation

If you use EgoTools in your research, please cite:

```
TBD
```

