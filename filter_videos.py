import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import mediapipe as mp
import argparse

class VideoFilter:
    def __init__(self):
        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    def detect_hand_and_tool(self, frame):
        """检测画面中是否同时存在手和工具"""
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 手部检测
        results = self.hands.process(rgb_frame)
        hand_detected = results.multi_hand_landmarks is not None
        
        # 工具检测 (这里使用简单的边缘检测作为示例，可以根据需求使用更复杂的工具检测方法)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        tool_detected = np.sum(edges) > 10000  # 阈值可以根据实际情况调整
        
        return hand_detected and tool_detected

    def process_video(self, video_path, output_path=None, save_clips=True):
        """处理视频并提取有效片段"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 用于存储有效片段的时间戳
        valid_segments = []
        segment_start = None
        frames_without_detection = 0
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 30 == 0:  # 每秒检查一次（假设30fps）
                has_hand_and_tool = self.detect_hand_and_tool(frame)
                
                if has_hand_and_tool:
                    frames_without_detection = 0
                    if segment_start is None:
                        segment_start = frame_count
                else:
                    frames_without_detection += 1
                    if frames_without_detection >= fps * 2 and segment_start is not None:  # 如果2秒都没有检测到
                        valid_segments.append((segment_start, frame_count))
                        segment_start = None
            
            frame_count += 1
        
        # 处理最后一个片段
        if segment_start is not None:
            valid_segments.append((segment_start, frame_count))
        
        cap.release()
        
        # 保存片段
        if save_clips and output_path and valid_segments:
            self.save_video_segments(video_path, valid_segments, output_path)
        
        return valid_segments

    def save_video_segments(self, video_path, segments, output_dir):
        """将视频片段保存到指定目录"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        for i, (start_frame, end_frame) in enumerate(segments):
            output_path = os.path.join(
                output_dir, 
                f"{Path(video_path).stem}_segment_{i}.mp4"
            )
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            out.release()
        
        cap.release()

def process_dataset_folder(dataset_path, output_base_dir):
    """处理数据集文件夹中的所有子文件夹"""
    results = []
    for folder in Path(dataset_path).glob('*'):
        if folder.is_dir():
            metadata_file = folder / 'metadata.csv'
            if metadata_file.exists():
                print(f"Processing folder: {folder.name}")
                df = pd.read_csv(metadata_file)
                results.append((folder.name, df))
    return results

def main():
    parser = argparse.ArgumentParser(description='Filter videos based on hand and tool detection')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Input dataset directory containing metadata.csv files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for filtered video segments')
    args = parser.parse_args()

    video_filter = VideoFilter()
    
    # 处理数据集中的所有文件夹
    dataset_results = process_dataset_folder(args.dataset_dir, args.output_dir)
    
    # 创建一个新的DataFrame来存储筛选结果
    filtered_results = []
    
    for folder_name, metadata_df in dataset_results:
        print(f"\nProcessing videos from {folder_name}...")
        
        for _, row in metadata_df.iterrows():
            video_id = row['id']
            video_title = row['title']
            video_path = f"videos/{video_id}.mp4"  # 假设视频存储在videos子文件夹中
            
            if not os.path.exists(video_path):
                print(f"Video file not found for ID: {video_id}")
                continue
                
            print(f"Processing video: {video_title} ({video_id})")
            output_folder = os.path.join(args.output_dir, folder_name, video_id)
            
            segments = video_filter.process_video(
                video_path,
                output_folder,
                save_clips=True
            )
            
            if segments:
                print(f"Found {len(segments)} valid segments")
                # 记录筛选结果
                for i, (start_frame, end_frame) in enumerate(segments):
                    filtered_results.append({
                        'folder': folder_name,
                        'video_id': video_id,
                        'video_title': video_title,
                        'segment_id': i,
                        'start_frame': start_frame,
                        'end_frame': end_frame
                    })
            else:
                print(f"No valid segments found")
    
    # 保存筛选结果
    if filtered_results:
        results_df = pd.DataFrame(filtered_results)
        results_path = os.path.join(args.output_dir, 'filtered_segments.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()