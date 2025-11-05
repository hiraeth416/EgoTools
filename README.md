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

### Intelligent Video Analysis (Qwen2-VL)
- **First-Person Detection**: Identifies egocentric perspective videos
- **Tool Detection**: Detects tools being used in videos
- **Hand Detection**: Identifies presence of human hands
- **Segment Extraction**: Extracts clips where hands and tools appear together
- **Detailed Analysis**: Frame-by-frame analysis with tool names and descriptions

## Installation

### Requirements

```bash
# Core dependencies
pip install pandas tqdm yt-dlp

# For video filtering and analysis
pip install torch transformers opencv-python pillow qwen-vl-utils
```

**Note**: Video analysis requires a GPU with sufficient VRAM (recommended: 16GB+ for the 7B model).

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

Analyze videos using Qwen2-VL model to identify first-person tool usage and extract relevant segments.

```bash
python filter_videos.py \
    --csv_path datasets/20251105_094846/metadata.csv \
    --video_dir download \
    --output_dir filtered_videos \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --sample_interval 30 \
    --save_clips
```

**Arguments:**
- `--csv_path`: Path to metadata.csv file (required)
- `--video_dir`: Directory containing downloaded videos (required)
- `--output_dir`: Output directory for analysis results (required)
- `--model_name`: Qwen2-VL model name or path (default: `Qwen/Qwen2-VL-7B-Instruct`)
- `--sample_interval`: Sample one frame every N frames (default: 30)
- `--save_clips`: Save video segments where hands and tools appear together

**Output Structure:**
```
filtered_videos/
├── analysis_summary.csv                    # Summary of all videos
├── filtered_first_person_tool_videos.csv   # First-person videos with tools
└── youtube/
    └── VIDEO_ID/
        ├── analysis_result.json            # Detailed analysis
        ├── VIDEO_segment_0_2.5s-5.3s.mp4  # Extracted segment
        └── VIDEO_segment_0_2.5s-5.3s_metadata.json
```

**Analysis Output Files:**

1. **analysis_summary.csv**: Summary statistics for all videos
   - `video_id`, `platform`, `title`
   - `is_first_person`: Boolean for first-person detection
   - `has_tool_usage`: Boolean for tool detection
   - `first_person_ratio`, `hand_ratio`, `tool_ratio`, `hand_and_tool_ratio`
   - `num_segments`: Number of valid segments
   - `total_segment_duration`: Total duration of segments

2. **analysis_result.json**: Detailed per-video analysis
   - Video metadata (fps, duration, resolution)
   - Frame-by-frame analysis results
   - Valid segments with timestamps
   - Tool names and descriptions

3. **Segment metadata JSON**: For each extracted segment
   - Start/end frame and timestamp
   - Duration
   - Frame analyses (tool names, actions, descriptions)

## Advanced Usage

### Custom Model Selection

Use a smaller model for faster processing:

```bash
python filter_videos.py \
    --csv_path metadata.csv \
    --video_dir download \
    --output_dir filtered_videos \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --sample_interval 60
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
   - Increase `--sample-interval` (e.g., 60) for faster processing
   - Use smaller models (`Qwen2-VL-2B-Instruct`) for speed
   - Monitor GPU memory usage

## Troubleshooting

### Common Issues

**Search/Download:**
- **No results found**: Check internet connection and platform accessibility
- **Download fails**: Verify `yt-dlp` is up to date: `pip install -U yt-dlp`
- **Cookie errors**: Update browser cookies or use `--cookies` flag

**Video Analysis:**
- **Out of Memory**: Use smaller model or increase `--sample-interval`
- **Slow processing**: Increase sampling interval or use more powerful GPU
- **Model download fails**: Ensure Hugging Face access and sufficient disk space
- **Video not found**: Check directory structure: `video_dir/platform/video_id/video.mp4`

## Project Structure

```
EgoTools/
├── search_videos.py      # Video metadata collection
├── download_videos.py    # Video downloader
├── filter_videos.py      # Video analysis with Qwen2-VL
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

