"""
Extract hand MANO parameters and hand joints from videos using HaMeR (Hand Mesh Recovery)
"""

import os
import cv2
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse


class HandPoseExtractor:
    def __init__(self, model_type='hamer', device='cuda'):
        """
        Initialize hand pose extractor
        
        Args:
            model_type: 'hamer' or 'mediapipe'
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model_type = model_type
        
        if model_type == 'hamer':
            self._init_hamer()
        elif model_type == 'mediapipe':
            self._init_mediapipe()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _init_hamer(self):
        """Initialize HaMeR model"""
        try:
            from hamer.configs import CACHE_DIR_HAMER
            from hamer.models import HAMER, download_models
            from hamer.utils import recursive_to
            from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
            from vitpose_model import ViTPoseModel
            
            # Download and load models
            download_models(CACHE_DIR_HAMER)
            
            # Load HaMeR model
            model_cfg = str(Path(CACHE_DIR_HAMER) / 'hamer' / 'checkpoints' / 'config.yaml')
            model_ckpt = str(Path(CACHE_DIR_HAMER) / 'hamer' / 'checkpoints' / 'checkpoint.pth')
            
            self.model = HAMER.from_pretrained(model_cfg, model_ckpt).to(self.device)
            self.model.eval()
            
            # Load detector
            self.detector = ViTPoseModel(self.device)
            
            print(f"HaMeR model loaded successfully on {self.device}")
            
        except ImportError:
            print("HaMeR not installed. Installing...")
            print("Please run: pip install git+https://github.com/geopavlakos/hamer.git")
            raise
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Hands model"""
        try:
            import mediapipe as mp
            
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            print("MediaPipe Hands initialized successfully")
            
        except ImportError:
            print("MediaPipe not installed. Installing...")
            print("Please run: pip install mediapipe")
            raise
    
    def extract_from_frame_hamer(self, frame):
        """
        Extract hand pose from a single frame using HaMeR
        
        Returns:
            dict with keys:
                - 'mano_params': MANO parameters (pose, shape, cam)
                - 'joints_3d': 3D hand joints
                - 'vertices': Hand mesh vertices
                - 'bbox': Hand bounding box
        """
        from hamer.utils import recursive_to
        from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
        
        # Detect hands in frame
        detections = self.detector.detect(frame)
        
        results = []
        
        for detection in detections:
            # Crop and preprocess hand region
            bbox = detection['bbox']
            
            # Prepare input
            dataset = ViTDetDataset(self.model.cfg, frame, [bbox])
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            
            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                
                with torch.no_grad():
                    out = self.model(batch)
                
                # Extract results
                pred_cam = out['pred_cam'].cpu().numpy()[0]
                pred_mano_params = {
                    'global_orient': out['pred_mano_params']['global_orient'].cpu().numpy()[0],
                    'hand_pose': out['pred_mano_params']['hand_pose'].cpu().numpy()[0],
                    'betas': out['pred_mano_params']['betas'].cpu().numpy()[0],
                }
                pred_vertices = out['pred_vertices'].cpu().numpy()[0]
                pred_joints_3d = out['pred_keypoints_3d'].cpu().numpy()[0]
                
                results.append({
                    'bbox': bbox,
                    'mano_params': pred_mano_params,
                    'cam': pred_cam,
                    'joints_3d': pred_joints_3d,
                    'vertices': pred_vertices
                })
        
        return results
    
    def extract_from_frame_mediapipe(self, frame):
        """
        Extract hand joints from a single frame using MediaPipe
        
        Returns:
            dict with keys:
                - 'joints_2d': 2D hand landmarks (21 keypoints per hand)
                - 'handedness': Left or Right hand
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(frame_rgb)
        
        hands_data = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Extract 21 landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                
                handedness = results.multi_handedness[idx].classification[0].label
                
                hands_data.append({
                    'joints_2d': np.array(landmarks),
                    'handedness': handedness,
                    'score': results.multi_handedness[idx].classification[0].score
                })
        
        return hands_data
    
    def extract_from_video(self, video_path, output_dir, sample_rate=1):
        """
        Extract hand pose from video
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save results
            sample_rate: Process every N frames (default: 1, process all frames)
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Processing video: {video_path.name}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        all_results = []
        frame_idx = 0
        
        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_idx % sample_rate == 0:
                    if self.model_type == 'hamer':
                        frame_results = self.extract_from_frame_hamer(frame)
                    else:
                        frame_results = self.extract_from_frame_mediapipe(frame)
                    
                    all_results.append({
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / fps,
                        'hands': frame_results
                    })
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # Save results
        self._save_results(all_results, output_dir, video_path.stem)
        
        print(f"Results saved to {output_dir}")
        
        return all_results
    
    def _save_results(self, results, output_dir, video_name):
        """Save extraction results"""
        output_dir = Path(output_dir)
        
        # Save as JSON (for metadata and 2D data)
        json_path = output_dir / f"{video_name}_hand_pose.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for frame_data in results:
            frame_dict = {
                'frame_idx': frame_data['frame_idx'],
                'timestamp': frame_data['timestamp'],
                'hands': []
            }
            
            for hand_data in frame_data['hands']:
                hand_dict = {}
                for key, value in hand_data.items():
                    if isinstance(value, np.ndarray):
                        hand_dict[key] = value.tolist()
                    elif isinstance(value, dict):
                        hand_dict[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                         for k, v in value.items()}
                    else:
                        hand_dict[key] = value
                frame_dict['hands'].append(hand_dict)
            
            json_results.append(frame_dict)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Saved JSON results to {json_path}")
        
        # Save as NPZ (for MANO parameters and 3D data)
        if self.model_type == 'hamer' and len(results) > 0 and len(results[0]['hands']) > 0:
            npz_data = {}
            
            # Collect all data
            for frame_data in results:
                frame_idx = frame_data['frame_idx']
                for hand_idx, hand_data in enumerate(frame_data['hands']):
                    prefix = f"frame_{frame_idx}_hand_{hand_idx}"
                    
                    if 'mano_params' in hand_data:
                        for param_name, param_value in hand_data['mano_params'].items():
                            npz_data[f"{prefix}_mano_{param_name}"] = param_value
                    
                    if 'joints_3d' in hand_data:
                        npz_data[f"{prefix}_joints_3d"] = hand_data['joints_3d']
                    
                    if 'vertices' in hand_data:
                        npz_data[f"{prefix}_vertices"] = hand_data['vertices']
            
            npz_path = output_dir / f"{video_name}_hand_pose.npz"
            np.savez(npz_path, **npz_data)
            print(f"Saved NPZ results to {npz_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract hand pose from videos')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model', type=str, default='mediapipe', 
                       choices=['hamer', 'mediapipe'],
                       help='Model to use (default: mediapipe, easier to install)')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--sample_rate', type=int, default=1,
                       help='Process every N frames (default: 1)')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Initialize extractor
    extractor = HandPoseExtractor(model_type=args.model, device=args.device)
    
    # Extract hand pose
    results = extractor.extract_from_video(
        video_path=args.video,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate
    )
    
    print(f"Processed {len(results)} frames")


if __name__ == '__main__':
    main()
