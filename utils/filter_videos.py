"""
Use Qwen3-VL model to analyze whether videos are first-person perspective tool usage videos
and identify segments where hands and tools appear together in the frame
"""
import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import json
import os
from datetime import datetime
from pathlib import Path
import argparse
import subprocess
import shutil


class QwenVideoAnalyzer:
    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", use_flash_attention=False):
        """Initialize Qwen3-VL model"""
        print(f"Loading model: {model_name}")
        
        if use_flash_attention:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, 
                dtype="auto", 
                device_map="auto"
            )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded successfully")
    
    def scan_download_directory(self, download_dir):
        """
        Scan download directory and find all videos
        Returns list of video info dicts with video_id, platform, and video_path
        """
        download_path = Path(download_dir)
        if not download_path.exists():
            raise FileNotFoundError(f"Download directory not found: {download_dir}")
        
        videos = []
        video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov']
        
        # Scan platform directories (youtube, bilibili, tiktok, etc.)
        for platform_dir in download_path.iterdir():
            if not platform_dir.is_dir():
                continue
            
            platform = platform_dir.name
            
            # Scan video ID directories
            for video_dir in platform_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                
                video_id = video_dir.name
                
                # Find video file
                for ext in video_extensions:
                    video_files = list(video_dir.glob(f"*{ext}"))
                    if video_files:
                        video_path = str(video_files[0])
                        # Try to extract title from filename
                        title = video_files[0].stem.replace(f"-{video_id}", "")
                        
                        videos.append({
                            'video_id': video_id,
                            'platform': platform,
                            'video_path': video_path,
                            'title': title
                        })
                        break
        
        return videos
    
    def analyze_video(self, video_path, analysis_type="full"):
        """
        Analyze video
        
        Args:
            video_path: Video file path (local path or URL)
            analysis_type: Analysis type ("full", "pov_check", "hand_tool_segments")
        
        Returns:
            Analysis result
        """
        if analysis_type == "full":
            return self._full_analysis(video_path)
        elif analysis_type == "pov_check":
            return self._check_first_person_pov(video_path)
        elif analysis_type == "hand_tool_segments":
            return self._detect_hand_tool_segments(video_path)
    
    def _full_analysis(self, video_path):
        """Full analysis: check both first-person perspective and hand-tool segments"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {
                        "type": "text", 
                        "text": """Please analyze this video and answer the following questions:
1. Is this video shot from a first-person perspective (POV/First-person perspective)?
2. Is there someone using tools in the video?
3. If tools are being used, please describe in detail the time periods (in seconds) where both hands and tools appear in the frame.

Please return the result in JSON format with the following fields:
{
  "is_first_person": true/false,
  "has_tool_usage": true/false,
  "confidence": "high/medium/low",
  "tool_description": "Description of the tools used",
  "hand_tool_segments": [
    {"start_time": 0, "end_time": 10, "description": "Description of actions in this segment"},
    ...
  ],
  "reasoning": "Reasoning for the judgment"
}"""
                    },
                ],
            }
        ]
        
        return self._generate_response(messages)
    
    def _check_first_person_pov(self, video_path):
        """Check if it's first-person perspective"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {
                        "type": "text", 
                        "text": """Please determine if this video is shot from a first-person perspective (First-person perspective/POV).
Characteristics of first-person perspective include:
- Viewing from the photographer's eye perspective
- Usually showing the photographer's own hands or body parts
- Camera moves with the photographer's head/body movement

Please answer:
1. Is it first-person perspective? (Yes/No)
2. Confidence level (High/Medium/Low)
3. Reasoning for judgment

Please return in JSON format: {"is_first_person": true/false, "confidence": "high/medium/low", "reasoning": "reason"}"""
                    },
                ],
            }
        ]
        
        return self._generate_response(messages)
    
    def _detect_hand_tool_segments(self, video_path):
        """Detect segments where hands and tools appear together"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {
                        "type": "text", 
                        "text": """Please carefully observe this video and identify all time periods where human hands and tools appear together in the frame.
                                For each such segment, please provide:
                                1. Start time (seconds)
                                2. End time (seconds)
                                3. Brief description (tools used and actions performed)

                                Please return in JSON format:
                                {
                                  "segments": [
                                    {"start_time": 0, "end_time": 5, "tool": "tool name", "action": "action description"},
                                    ...
                                  ],
                                  "total_segments": count
                                }"""
                    },
                ],
            }
        ]
        
        return self._generate_response(messages)
    
    def _generate_response(self, messages, max_new_tokens=512):
        """Generate model response"""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # Generate output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0] if output_text else ""


def cut_video_segment(input_video, output_video, start_time, end_time):
    """
    Cut a segment from video using ffmpeg
    
    Args:
        input_video: Input video path
        output_video: Output video path
        start_time: Start time in seconds
        end_time: End time in seconds
    
    Returns:
        True if successful, False otherwise
    """
    try:
        duration = end_time - start_time
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if exists
            '-i', input_video,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',  # Copy codec (faster, no re-encoding)
            output_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error cutting video: {e}")
        return False


def organize_and_cut_videos(results, output_dir):
    """
    Organize results and cut video segments based on analysis
    
    Creates structure:
    output_results/
        platform/
            video_id/
                analysis.json
                segment_0/
                    segment.mp4
                    segment_info.json
                segment_1/
                    segment.mp4
                    segment_info.json
                ...
    
    Args:
        results: List of analysis results
        output_dir: Base output directory
    """
    output_path = Path(output_dir)
    
    for result in results:
        if 'error' in result:
            print(f"Skipping {result['video_id']} due to error")
            continue
        
        video_id = result['video_id']
        platform = result['platform']
        video_path = result['video_path']
        
        # Create directory structure: output_results/platform/video_id/
        video_output_dir = output_path / platform / video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save analysis.json in video directory
        analysis_file = video_output_dir / "analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\nProcessing video: {platform}/{video_id}")
        
        # Get hand_tool_segments
        analysis = result.get('analysis', {})
        segments = analysis.get('hand_tool_segments', [])
        
        if not segments:
            print(f"No hand-tool segments found for {video_id}")
            continue
        
        print(f"Found {len(segments)} segments to cut")
        
        # Cut each segment
        for idx, segment in enumerate(segments):
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', 0)
            description = segment.get('description', '')
            
            # Create segment directory
            segment_dir = video_output_dir / f"segment_{idx}"
            segment_dir.mkdir(exist_ok=True)
            
            # Output video file
            segment_video = segment_dir / f"segment_{idx}.mp4"
            
            print(f"  Cutting segment {idx}: {start_time}s - {end_time}s")
            
            # Cut video segment
            success = cut_video_segment(video_path, str(segment_video), start_time, end_time)
            
            if success:
                # Save segment info
                segment_info = {
                    'segment_id': idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'description': description,
                    'tool': segment.get('tool', ''),
                    'action': segment.get('action', description),
                    'source_video': video_path,
                    'source_video_id': video_id,
                    'platform': platform
                }
                
                segment_info_file = segment_dir / "segment_info.json"
                with open(segment_info_file, 'w', encoding='utf-8') as f:
                    json.dump(segment_info, f, ensure_ascii=False, indent=2)
                
                print(f"    ✓ Saved to {segment_dir}")
            else:
                print(f"    ✗ Failed to cut segment {idx}")


def process_videos(download_dir="download", output_dir=None, use_flash_attention=False, cut_videos=True):
    """
    Process all videos in download directory
    
    Args:
        download_dir: Local video download directory (default: "download/")
        output_dir: Output directory
        use_flash_attention: Whether to use flash attention
        cut_videos: Whether to cut video segments (default: True)
    """
    # Initialize analyzer
    analyzer = QwenVideoAnalyzer(use_flash_attention=use_flash_attention)
    
    # Scan download directory for videos
    print(f"Scanning directory: {download_dir}")
    videos = analyzer.scan_download_directory(download_dir)
    print(f"Found {len(videos)} videos to analyze")
    
    if not videos:
        print("No videos found in download directory")
        return
    
    # Show video statistics
    platforms = {}
    for video in videos:
        platform = video['platform']
        platforms[platform] = platforms.get(platform, 0) + 1
    
    print("\nVideos by platform:")
    for platform, count in platforms.items():
        print(f"  {platform}: {count} videos")
    
    # Prepare output directory
    if output_dir is None:
        output_dir = Path(download_dir).parent / f"qwen_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store results
    results = []
    
    # Process each video
    for idx, video_info in enumerate(videos):
        video_id = video_info['video_id']
        platform = video_info['platform']
        title = video_info['title']
        video_path = video_info['video_path']
        
        print(f"\n[{idx+1}/{len(videos)}] Analyzing video: {video_id}")
        print(f"Platform: {platform}")
        print(f"Title: {title}")
        print(f"Path: {video_path}")
        
        try:
            # Execute full analysis
            print("Analyzing...")
            analysis_result = analyzer.analyze_video(video_path, analysis_type="full")
            
            print(f"Analysis result:\n{analysis_result}")
            
            # Try to parse JSON result
            try:
                # Try to extract JSON from output
                if "{" in analysis_result and "}" in analysis_result:
                    json_start = analysis_result.find("{")
                    json_end = analysis_result.rfind("}") + 1
                    json_str = analysis_result[json_start:json_end]
                    parsed_result = json.loads(json_str)
                else:
                    parsed_result = {"raw_response": analysis_result}
            except json.JSONDecodeError:
                parsed_result = {"raw_response": analysis_result}
            
            # Save result
            result = {
                'video_id': video_id,
                'platform': platform,
                'title': title,
                'video_path': video_path,
                'analysis': parsed_result,
                'raw_output': analysis_result,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error: {e}")
            result = {
                'video_id': video_id,
                'platform': platform,
                'title': title,
                'video_path': video_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
    
    # Save comprehensive results to JSON
    summary_file = output_dir / "analysis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved in: {output_dir}")
    print(f"- Comprehensive results: analysis_summary.json")
    print(f"  Total videos analyzed: {len(results)}")
    
    # Cut videos based on analysis results
    if cut_videos:
        print(f"\n{'='*60}")
        print("Starting video segmentation...")
        print(f"{'='*60}")
        organize_and_cut_videos(results, output_dir)
        
        print(f"\n{'='*60}")
        print("All processing complete!")
        print(f"{'='*60}")
    else:
        print("\nSkipping video segmentation (--no-cut flag set)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze videos using Qwen3-VL')
    parser.add_argument('--download-dir', type=str, default='download', help='Local video download directory (default: download/)')
    parser.add_argument('--output-dir', type=str, default='output_results', help='Output directory')
    parser.add_argument('--flash-attention', action='store_true', help='Use flash attention 2')
    parser.add_argument('--single-video', type=str, default=None, help='Analyze only a single video ID (format: platform/video_id)')
    parser.add_argument('--no-cut', action='store_true', help='Skip video cutting, only perform analysis')
    
    args = parser.parse_args()
    
    if args.single_video:
        # Single video analysis mode
        analyzer = QwenVideoAnalyzer(use_flash_attention=args.flash_attention)
        
        # Parse platform/video_id format
        if '/' in args.single_video:
            platform, video_id = args.single_video.split('/', 1)
        else:
            # Default to youtube if no platform specified
            platform = 'youtube'
            video_id = args.single_video
        
        # Find video file
        video_dir = Path(args.download_dir) / platform / video_id
        if not video_dir.exists():
            print(f"Error: Video directory not found: {video_dir}")
            return
        
        video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov']
        video_path = None
        for ext in video_extensions:
            video_files = list(video_dir.glob(f"*{ext}"))
            if video_files:
                video_path = str(video_files[0])
                break
        
        if not video_path:
            print(f"Error: No video file found in {video_dir}")
            return
        
        print(f"Analyzing video: {video_id}")
        print(f"Platform: {platform}")
        print(f"Path: {video_path}")
        
        result = analyzer.analyze_video(video_path, analysis_type="full")
        print(f"\nAnalysis result:\n{result}")
    else:
        # Batch analysis mode
        process_videos(args.download_dir, args.output_dir, args.flash_attention, cut_videos=not args.no_cut)


if __name__ == "__main__":
    main()
