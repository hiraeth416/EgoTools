import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import argparse
import base64
import json
from typing import List, Tuple, Optional
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import io

class VideoFilter:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"):
        """
        Initialize video filter with Qwen3-VL model
        Args:
            model_name: Name or path of the Qwen3-VL model
        """
        print(f"Loading Qwen3-VL model: {model_name}")

        # Load model and processor
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        print("Model loaded successfully")

    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 string"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def analyze_frame(self, frame) -> dict:
        """
        Analyze a single frame using Qwen3-VL model
        Returns:
            dict with keys: is_first_person, has_hand, has_tool, description
        """
        # Convert frame to base64
        image_base64 = self.frame_to_base64(frame)
        
        # Prepare prompt
        prompt = """Analyze this image and answer the following questions:
1. Is this a first-person view (egocentric perspective)? 
2. Are there human hands visible in the image?
3. Are there any tools being used or held?
4. If yes to questions 2 and 3, describe what tool is being used and what action is being performed.

Please respond in JSON format:
{
    "is_first_person": true/false,
    "has_hand": true/false,
    "has_tool": true/false,
    "tool_name": "name of tool or null",
    "description": "brief description of the scene"
}"""
        
        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_base64},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        # Parse JSON response
        try:
            # Try to extract JSON from the response
            start_idx = output_text.find('{')
            end_idx = output_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = output_text[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                # Fallback: create default response
                result = {
                    "is_first_person": False,
                    "has_hand": False,
                    "has_tool": False,
                    "tool_name": None,
                    "description": output_text
                }
        except json.JSONDecodeError:
            # If JSON parsing fails, create a default response
            result = {
                "is_first_person": False,
                "has_hand": False,
                "has_tool": False,
                "tool_name": None,
                "description": output_text
            }
        
        return result

    def process_video(self, video_path, output_path=None, save_clips=True, sample_interval=30):
        """
        Process video and extract valid segments where hands and tools appear together
        Args:
            video_path: Path to the video file
            output_path: Directory to save video segments
            save_clips: Whether to save the extracted segments
            sample_interval: Sample one frame every N frames (default: 30, i.e., 1 per second at 30fps)
        Returns:
            dict with video analysis results and valid segments
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}, Duration: {total_frames/fps:.2f}s")
        
        # Store analysis results
        frame_analyses = []
        valid_segments = []
        segment_start = None
        segment_analyses = []
        
        # Video-level statistics
        first_person_count = 0
        hand_count = 0
        tool_count = 0
        hand_and_tool_count = 0
        
        frame_count = 0
        sampled_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample frames at specified interval
            if frame_count % sample_interval == 0:
                sampled_count += 1
                print(f"Analyzing frame {frame_count}/{total_frames} ({sampled_count} sampled frames)...")
                
                # Analyze frame with Qwen3-VL
                analysis = self.analyze_frame(frame)
                analysis['frame_number'] = frame_count
                analysis['timestamp'] = frame_count / fps
                frame_analyses.append(analysis)
                
                # Update statistics
                if analysis.get('is_first_person', False):
                    first_person_count += 1
                if analysis.get('has_hand', False):
                    hand_count += 1
                if analysis.get('has_tool', False):
                    tool_count += 1
                if analysis.get('has_hand', False) and analysis.get('has_tool', False):
                    hand_and_tool_count += 1
                    
                    # Start a new segment or continue current one
                    if segment_start is None:
                        segment_start = frame_count
                        segment_analyses = [analysis]
                    else:
                        segment_analyses.append(analysis)
                else:
                    # End current segment if it exists
                    if segment_start is not None:
                        valid_segments.append({
                            'start_frame': segment_start,
                            'end_frame': frame_count,
                            'start_time': segment_start / fps,
                            'end_time': frame_count / fps,
                            'duration': (frame_count - segment_start) / fps,
                            'analyses': segment_analyses
                        })
                        segment_start = None
                        segment_analyses = []
            
            frame_count += 1
        
        # Process the last segment
        if segment_start is not None:
            valid_segments.append({
                'start_frame': segment_start,
                'end_frame': frame_count,
                'start_time': segment_start / fps,
                'end_time': frame_count / fps,
                'duration': (frame_count - segment_start) / fps,
                'analyses': segment_analyses
            })
        
        cap.release()
        
        # Calculate video-level metrics
        is_first_person_video = (first_person_count / max(sampled_count, 1)) > 0.5
        has_tool_usage = tool_count > 0
        
        results = {
            'video_path': video_path,
            'is_first_person': is_first_person_video,
            'has_tool_usage': has_tool_usage,
            'total_frames': total_frames,
            'sampled_frames': sampled_count,
            'fps': fps,
            'duration': total_frames / fps,
            'first_person_ratio': first_person_count / max(sampled_count, 1),
            'hand_ratio': hand_count / max(sampled_count, 1),
            'tool_ratio': tool_count / max(sampled_count, 1),
            'hand_and_tool_ratio': hand_and_tool_count / max(sampled_count, 1),
            'valid_segments': valid_segments,
            'frame_analyses': frame_analyses
        }
        
        # Save segments
        if save_clips and output_path and valid_segments:
            self.save_video_segments(video_path, valid_segments, output_path)
        
        return results

    def save_video_segments(self, video_path, segments, output_dir):
        """
        Save video segments to specified directory
        Args:
            video_path: Path to source video
            segments: List of segment dicts with start_frame, end_frame, etc.
            output_dir: Directory to save segments
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        for i, segment in enumerate(segments):
            start_frame = segment['start_frame']
            end_frame = segment['end_frame']
            
            output_path = os.path.join(
                output_dir, 
                f"{Path(video_path).stem}_segment_{i}_{segment['start_time']:.1f}s-{segment['end_time']:.1f}s.mp4"
            )
            
            print(f"Saving segment {i+1}/{len(segments)}: {output_path}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            out.release()
            
            # Save segment metadata
            metadata_path = output_path.replace('.mp4', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'duration': segment['duration'],
                    'analyses': segment['analyses']
                }, f, indent=2)
        
        cap.release()

def main():
    parser = argparse.ArgumentParser(description='Filter videos based on hand and tool detection using Qwen3-VL')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to metadata.csv file')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing downloaded videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for filtered video segments')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-7B-Instruct', 
                        help='Qwen3-VL model name or path')
    parser.add_argument('--sample_interval', type=int, default=30, 
                        help='Sample one frame every N frames (default: 30)')
    parser.add_argument('--save_clips', action='store_true', 
                        help='Save video segments where hands and tools appear together')
    args = parser.parse_args()

    # Load metadata
    print(f"Loading metadata from {args.csv_path}")
    metadata_df = pd.read_csv(args.csv_path)
    print(f"Found {len(metadata_df)} videos in metadata")
    
    # Initialize video filter with Qwen3-VL
    video_filter = VideoFilter(model_name=args.model_name)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each video
    all_results = []
    
    for idx, row in metadata_df.iterrows():
        video_id = row['id']
        platform = row.get('platform', 'youtube')
        
        # Construct video path
        video_path = os.path.join(args.video_dir, platform, video_id)
        
        # Find the video file in the directory
        if os.path.isdir(video_path):
            video_files = [f for f in os.listdir(video_path) if f.endswith(('.mp4', '.webm', '.mkv'))]
            if not video_files:
                print(f"No video file found in {video_path}")
                continue
            video_file = os.path.join(video_path, video_files[0])
        else:
            print(f"Video directory not found: {video_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing [{idx+1}/{len(metadata_df)}]: {row.get('title', video_id)}")
        print(f"Video ID: {video_id}")
        print(f"{'='*80}")
        
        # Create output folder for this video
        output_folder = os.path.join(args.output_dir, platform, video_id)
        os.makedirs(output_folder, exist_ok=True)
        
        # Process video
        result = video_filter.process_video(
            video_file,
            output_folder,
            save_clips=args.save_clips,
            sample_interval=args.sample_interval
        )
        
        if result:
            # Add metadata to result
            result['video_id'] = video_id
            result['platform'] = platform
            result['title'] = row.get('title', '')
            
            all_results.append({
                'video_id': video_id,
                'platform': platform,
                'title': row.get('title', ''),
                'is_first_person': result['is_first_person'],
                'has_tool_usage': result['has_tool_usage'],
                'duration': result['duration'],
                'first_person_ratio': result['first_person_ratio'],
                'hand_ratio': result['hand_ratio'],
                'tool_ratio': result['tool_ratio'],
                'hand_and_tool_ratio': result['hand_and_tool_ratio'],
                'num_segments': len(result['valid_segments']),
                'total_segment_duration': sum(seg['duration'] for seg in result['valid_segments'])
            })
            
            # Save detailed results for this video
            result_file = os.path.join(output_folder, 'analysis_result.json')
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\nResults:")
            print(f"  First-person video: {result['is_first_person']}")
            print(f"  Has tool usage: {result['has_tool_usage']}")
            print(f"  Valid segments with hands + tools: {len(result['valid_segments'])}")
            if result['valid_segments']:
                total_duration = sum(seg['duration'] for seg in result['valid_segments'])
                print(f"  Total segment duration: {total_duration:.2f}s")
    
    # Save summary results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save filtered videos (first-person with tool usage)
        filtered_df = results_df[results_df['is_first_person'] & results_df['has_tool_usage']]
        
        summary_path = os.path.join(args.output_dir, 'analysis_summary.csv')
        results_df.to_csv(summary_path, index=False)
        print(f"\n{'='*80}")
        print(f"Analysis complete!")
        print(f"Total videos processed: {len(results_df)}")
        print(f"First-person videos: {results_df['is_first_person'].sum()}")
        print(f"Videos with tool usage: {results_df['has_tool_usage'].sum()}")
        print(f"First-person videos with tool usage: {len(filtered_df)}")
        print(f"Summary saved to: {summary_path}")
        
        # Save filtered list
        if len(filtered_df) > 0:
            filtered_path = os.path.join(args.output_dir, 'filtered_first_person_tool_videos.csv')
            filtered_df.to_csv(filtered_path, index=False)
            print(f"Filtered list saved to: {filtered_path}")

if __name__ == "__main__":
    main()