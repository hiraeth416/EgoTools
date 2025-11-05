# Video Filtering with Qwen2-VL

This tool uses the Qwen2-VL vision-language model to analyze videos and identify first-person perspective videos with tool usage, detecting segments where hands and tools appear together.

## Features

- **First-Person Detection**: Identifies if a video is filmed from an egocentric (first-person) perspective
- **Tool Detection**: Detects if tools are being used in the video
- **Hand Detection**: Identifies the presence of human hands
- **Segment Extraction**: Extracts video segments where both hands and tools appear together
- **Detailed Analysis**: Provides frame-by-frame analysis with tool names and descriptions

## Requirements

Install the required dependencies:

```bash
pip install torch transformers opencv-python pandas pillow qwen-vl-utils
```

You'll also need a GPU with sufficient VRAM to run the Qwen2-VL model (recommended: 16GB+ for the 7B model).

## Usage

### Basic Usage

```bash
python filter_videos.py \
    --csv_path test/20251105_094846/metadata.csv \
    --video_dir download/youtube \
    --output_dir filtered_videos \
    --save_clips
```

### Arguments

- `--csv_path`: Path to the metadata CSV file containing video information
- `--video_dir`: Directory containing the downloaded videos (organized by platform/video_id/)
- `--output_dir`: Output directory for analysis results and filtered segments
- `--model_name`: Qwen2-VL model to use (default: `Qwen/Qwen2-VL-7B-Instruct`)
- `--sample_interval`: Sample one frame every N frames (default: 30, i.e., 1 frame per second at 30fps)
- `--save_clips`: Flag to save video segments where hands and tools appear together

### Advanced Usage

Use a different model or sampling rate:

```bash
python filter_videos.py \
    --csv_path metadata.csv \
    --video_dir download/youtube \
    --output_dir filtered_videos \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --sample_interval 15 \
    --save_clips
```

## Output Structure

The tool generates the following outputs:

```
filtered_videos/
├── analysis_summary.csv                    # Summary of all videos analyzed
├── filtered_first_person_tool_videos.csv   # Videos that are first-person with tool usage
└── youtube/                                # Results organized by platform
    └── VIDEO_ID/
        ├── analysis_result.json            # Detailed analysis results
        ├── VIDEO_NAME_segment_0_2.5s-5.3s.mp4         # Extracted segment
        ├── VIDEO_NAME_segment_0_2.5s-5.3s_metadata.json  # Segment metadata
        └── ...
```

### Output Files

#### analysis_summary.csv
Contains summary statistics for all processed videos:
- `video_id`: Unique video identifier
- `platform`: Video platform (youtube, bilibili, etc.)
- `title`: Video title
- `is_first_person`: Boolean indicating if video is first-person
- `has_tool_usage`: Boolean indicating if tools are used
- `duration`: Total video duration in seconds
- `first_person_ratio`: Ratio of frames detected as first-person
- `hand_ratio`: Ratio of frames with hands detected
- `tool_ratio`: Ratio of frames with tools detected
- `hand_and_tool_ratio`: Ratio of frames with both hands and tools
- `num_segments`: Number of valid segments extracted
- `total_segment_duration`: Total duration of extracted segments

#### analysis_result.json
Detailed per-video analysis including:
- Video metadata (path, fps, duration, etc.)
- Overall classification (is_first_person, has_tool_usage)
- Frame-by-frame analysis results
- Valid segments with timestamps and descriptions

#### Segment Metadata JSON
For each extracted segment:
- Start/end frame and timestamp
- Segment duration
- Frame analyses within the segment (tool names, descriptions)

## Example Workflow

1. **Search and download videos**:
```bash
# Search for videos
python search_videos.py --output test --max_results 50

# Download videos
python download_videos.py \
    --csv_path test/20251105_094846/metadata.csv \
    --output download \
    --platforms youtube
```

2. **Filter and analyze videos**:
```bash
python filter_videos.py \
    --csv_path test/20251105_094846/metadata.csv \
    --video_dir download \
    --output_dir filtered_videos \
    --save_clips
```

3. **Review results**:
- Check `filtered_videos/analysis_summary.csv` for overview
- Check `filtered_videos/filtered_first_person_tool_videos.csv` for qualified videos
- Review individual `analysis_result.json` files for detailed information
- Watch extracted segments showing hand-tool interactions

## Performance Tips

1. **Sampling Rate**: Increase `--sample_interval` to process faster (e.g., 60 = sample every 2 seconds)
2. **Model Selection**: Use smaller models like `Qwen2-VL-2B-Instruct` for faster processing
3. **GPU Memory**: Monitor GPU memory usage; reduce batch processing if OOM errors occur
4. **Disk Space**: Ensure sufficient disk space for extracted segments if using `--save_clips`

## Troubleshooting

- **Out of Memory**: Use a smaller model or increase `--sample_interval`
- **Slow Processing**: Increase sampling interval or use a more powerful GPU
- **Model Download Issues**: Ensure you have access to Hugging Face and the model repository
- **Video Not Found**: Check that video files are organized as `video_dir/platform/video_id/video_file.mp4`

## Notes

- The model analyzes videos frame-by-frame at the specified sampling interval
- First-person classification is based on majority voting across sampled frames
- Segments are created when consecutive frames show both hands and tools
- JSON output includes detailed descriptions that can be used for further analysis
