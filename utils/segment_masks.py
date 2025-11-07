"""
Use Grounding DINO + SAM2 to segment hands and tools from video segments
Grounding DINO: Detect objects based on text prompts (e.g., "hand", "knife")
SAM2: Precise segmentation using detected bounding boxes as prompts
"""
import torch
import numpy as np
from pathlib import Path
import json
import cv2
import argparse
from datetime import datetime


class HandToolSegmenter:
    def __init__(self, model_name="facebook/sam2-hiera-large", checkpoint_path=None, use_grounding_dino=True):
        """
        Initialize SAM2 and optionally Grounding DINO models
        
        Args:
            model_name: HuggingFace model name for SAM2
            checkpoint_path: Optional local checkpoint path for SAM2
            use_grounding_dino: Whether to use Grounding DINO for detection (recommended)
        """
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Please install with: "
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print(f"Loading SAM2 model: {model_name}")
        
        try:
            # Load video predictor
            if checkpoint_path and Path(checkpoint_path).exists():
                print(f"Loading from local checkpoint: {checkpoint_path}")
                # For local checkpoint, need to use build_sam approach
                from sam2.build_sam import build_sam2_video_predictor
                
                # Map checkpoint filename to config
                checkpoint_name = Path(checkpoint_path).stem
                config_map = {
                    "sam2_hiera_tiny": "sam2_hiera_t",
                    "sam2_hiera_small": "sam2_hiera_s",
                    "sam2_hiera_base_plus": "sam2_hiera_b+",
                    "sam2_hiera_large": "sam2_hiera_l",
                }
                model_cfg = config_map.get(checkpoint_name, "sam2_hiera_l")
                
                self.video_predictor = build_sam2_video_predictor(model_cfg, checkpoint_path)
                self.image_predictor = SAM2ImagePredictor.from_pretrained(model_name)
            else:
                # Download from HuggingFace
                print(f"Loading from HuggingFace: {model_name}")
                self.video_predictor = SAM2VideoPredictor.from_pretrained(model_name)
                self.image_predictor = SAM2ImagePredictor.from_pretrained(model_name)
            
            print("SAM2 model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nIf using local checkpoint, make sure it exists:")
            print("  checkpoints/sam2_hiera_large.pt")
            print("\nOr the model will be downloaded from HuggingFace automatically.")
            raise
        
        # Load Grounding DINO for object detection
        self.use_grounding_dino = use_grounding_dino
        if use_grounding_dino:
            try:
                from groundingdino.util.inference import load_model, predict
                from groundingdino.util.utils import clean_state_dict
                import groundingdino.datasets.transforms as T
                
                print("Loading Grounding DINO model...")
                # Using GroundingDINO config and checkpoint
                grounding_dino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
                grounding_dino_checkpoint = "checkpoints/groundingdino_swint_ogc.pth"
                
                if not Path(grounding_dino_checkpoint).exists():
                    print(f"Warning: Grounding DINO checkpoint not found at {grounding_dino_checkpoint}")
                    print("Please download it with:")
                    print("wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P checkpoints/")
                    print("\nFalling back to SAM2 auto-segmentation mode")
                    self.use_grounding_dino = False
                    self.grounding_dino_model = None
                else:
                    self.grounding_dino_model = load_model(grounding_dino_config, grounding_dino_checkpoint)
                    self.grounding_dino_predict = predict
                    
                    # Transform for Grounding DINO
                    self.transform = T.Compose([
                        T.RandomResize([800], max_size=1333),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
                    print("Grounding DINO model loaded successfully")
                    
            except ImportError:
                print("Warning: Grounding DINO not installed. Using SAM2 auto-segmentation mode.")
                print("To install Grounding DINO:")
                print("  pip install groundingdino-py")
                self.use_grounding_dino = False
                self.grounding_dino_model = None
            except Exception as e:
                print(f"Warning: Error loading Grounding DINO: {e}")
                print("Falling back to SAM2 auto-segmentation mode")
                self.use_grounding_dino = False
                self.grounding_dino_model = None
import torch
import numpy as np
from pathlib import Path
import json
import cv2
import argparse
from datetime import datetime


class HandToolSegmenter:
    def __init__(self, model_name="facebook/sam2-hiera-large", checkpoint_path=None):
        """
        Initialize SAM2 model
        
        Args:
            model_name: Model name for from_pretrained (e.g., "facebook/sam2-hiera-large")
            checkpoint_path: Optional local checkpoint path (if None, will download from HuggingFace)
        """
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Please install with: "
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print(f"Loading SAM2 model: {model_name}")
        
        try:
            # Load video predictor
            if checkpoint_path and Path(checkpoint_path).exists():
                print(f"Loading from local checkpoint: {checkpoint_path}")
                # For local checkpoint, need to use build_sam approach
                from sam2.build_sam import build_sam2_video_predictor
                
                # Map checkpoint filename to config
                checkpoint_name = Path(checkpoint_path).stem
                config_map = {
                    "sam2_hiera_tiny": "sam2_hiera_t",
                    "sam2_hiera_small": "sam2_hiera_s",
                    "sam2_hiera_base_plus": "sam2_hiera_b+",
                    "sam2_hiera_large": "sam2_hiera_l",
                }
                model_cfg = config_map.get(checkpoint_name, "sam2_hiera_l")
                
                self.video_predictor = build_sam2_video_predictor(model_cfg, checkpoint_path)
                self.image_predictor = SAM2ImagePredictor.from_pretrained(model_name)
            else:
                # Download from HuggingFace
                print(f"Loading from HuggingFace: {model_name}")
                self.video_predictor = SAM2VideoPredictor.from_pretrained(model_name)
                self.image_predictor = SAM2ImagePredictor.from_pretrained(model_name)
            
            print("SAM2 model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nIf using local checkpoint, make sure it exists:")
            print("  checkpoints/sam2_hiera_large.pt")
            print("\nOr the model will be downloaded from HuggingFace automatically.")
            raise
    
    def _load_tool_names_from_analysis(self, segment_dir):
        """
        Load tool names from analysis.json in parent directory
        
        Args:
            segment_dir: Path to segment directory (e.g., .../youtube/video_id/segment_0)
        
        Returns:
            List of tool names mentioned in this segment
        """
        segment_path = Path(segment_dir)
        
        # Try to load segment_info.json first
        segment_info_path = segment_path / "segment_info.json"
        if segment_info_path.exists():
            try:
                with open(segment_info_path, 'r') as f:
                    segment_info = json.load(f)
                    tool = segment_info.get('tool', '')
                    if tool:
                        return [tool]
            except Exception as e:
                print(f"Warning: Could not read segment_info.json: {e}")
        
        # Fallback to analysis.json in parent directory
        analysis_path = segment_path.parent / "analysis.json"
        if analysis_path.exists():
            try:
                with open(analysis_path, 'r') as f:
                    analysis = json.load(f)
                    
                # Get segment index from directory name
                segment_name = segment_path.name  # e.g., "segment_0"
                segment_idx = int(segment_name.split('_')[1])
                
                # Find matching segment in hand_tool_segments
                hand_tool_segments = analysis.get('analysis', {}).get('hand_tool_segments', [])
                if segment_idx < len(hand_tool_segments):
                    tool = hand_tool_segments[segment_idx].get('tool', '')
                    if tool:
                        return [tool]
                
                # Fallback: get all unique tools mentioned
                tools = set()
                for seg in hand_tool_segments:
                    tool = seg.get('tool', '')
                    if tool:
                        tools.add(tool)
                return list(tools)
                
            except Exception as e:
                print(f"Warning: Could not read analysis.json: {e}")
        
        # Default fallback
        return ['tool', 'knife', 'scissors', 'bottle']  # Common tools
    
    def detect_objects_with_grounding_dino(self, image, text_prompt, box_threshold=0.25, text_threshold=0.2):
        """
        Detect objects in image using Grounding DINO
        
        Args:
            image: Image array (RGB)
            text_prompt: Text description (e.g., "hand . knife . scissors")
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text
        
        Returns:
            boxes: Detected bounding boxes in xyxy format
            logits: Confidence scores
            phrases: Detected phrases
        """
        if not self.use_grounding_dino or self.grounding_dino_model is None:
            return None, None, None
        
        # Convert to PIL Image
        from PIL import Image as PILImage
        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image
        
        # Transform image
        image_transformed, _ = self.transform(pil_image, None)
        
        # Predict
        boxes, logits, phrases = self.grounding_dino_predict(
            model=self.grounding_dino_model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        # Convert boxes to image coordinates
        h, w = image.shape[:2] if isinstance(image, np.ndarray) else (pil_image.height, pil_image.width)
        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        
        return boxes.cpu().numpy(), logits.cpu().numpy(), phrases
    
    def segment_video_with_prompts(self, video_path, output_dir, prompt_frame=0, hand_points=None, tool_points=None):
        """
        Segment hands and tools from video using point prompts
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for masks
            prompt_frame: Frame index to provide point prompts (0-indexed)
            hand_points: List of (x, y) points on hands in prompt frame
            tool_points: List of (x, y) points on tools in prompt frame
        
        Returns:
            Dictionary with segmentation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSegmenting video: {video_path}")
        print(f"Output directory: {output_dir}")
        
        # Load video frames
        frames = self._load_video_frames(video_path)
        print(f"Loaded {len(frames)} frames")
        
        if not hand_points and not tool_points:
            print("Warning: No prompts provided, will use automatic segmentation")
            return self.auto_segment_video(video_path, output_dir)
        
        # Initialize video predictor with inference mode
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.video_predictor.init_state(video_path=str(video_path))
            
            # Add prompts for hands
            if hand_points:
                print(f"Adding {len(hand_points)} hand prompts at frame {prompt_frame}")
                for i, point in enumerate(hand_points):
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt_frame,
                        obj_id=i + 1,  # Hand object IDs start from 1
                        points=np.array([point], dtype=np.float32),
                        labels=np.array([1], np.int32),  # 1 for foreground
                    )
            
            # Add prompts for tools
            tool_start_id = len(hand_points) + 1 if hand_points else 1
            if tool_points:
                print(f"Adding {len(tool_points)} tool prompts at frame {prompt_frame}")
                for i, point in enumerate(tool_points):
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt_frame,
                        obj_id=tool_start_id + i,  # Tool object IDs
                        points=np.array([point], dtype=np.float32),
                        labels=np.array([1], np.int32),
                    )
            
            # Propagate masks through video
            print("Propagating masks through video...")
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
        
        # Save masks
        print("Saving masks...")
        results = self._save_video_masks(video_segments, frames, output_path, 
                                         num_hands=len(hand_points) if hand_points else 0,
                                         num_tools=len(tool_points) if tool_points else 0)
        
        print(f"✓ Segmentation complete! Saved {results['total_masks']} masks")
        return results
    
    def segment_video_with_grounding_dino(self, video_path, output_dir, tool_names=None, 
                                         segment_dir=None, sample_interval=5):
        """
        Segment video using Grounding DINO for detection + SAM2 for segmentation
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for masks
            tool_names: List of tool names to detect (e.g., ["knife", "scissors"])
            segment_dir: Path to segment directory (to load tool names from analysis)
            sample_interval: Sample every N frames (default: 5)
        
        Returns:
            Dictionary with segmentation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSegmenting video with Grounding DINO + SAM2: {video_path}")
        print(f"Output directory: {output_dir}")
        
        # Load tool names from analysis if not provided
        if tool_names is None and segment_dir is not None:
            tool_names = self._load_tool_names_from_analysis(segment_dir)
            print(f"Loaded tool names from analysis: {tool_names}")
        elif tool_names is None:
            tool_names = ['tool']
        
        # Build text prompt for Grounding DINO
        # Format: "hand . knife . scissors"
        text_prompt = "hand . " + " . ".join(tool_names)
        print(f"Text prompt: '{text_prompt}'")
        
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_to_process = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_interval == 0:
                frames_to_process.append((frame_idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            
            frame_idx += 1
        
        cap.release()
        print(f"Processing {len(frames_to_process)} sampled frames (total: {total_frames})")
        
        all_masks = []
        total_masks = 0
        
        for frame_idx, frame in frames_to_process:
            print(f"Processing frame {frame_idx}...", end='\r')
            
            # Detect objects with Grounding DINO
            boxes, logits, phrases = self.detect_objects_with_grounding_dino(
                frame, text_prompt, box_threshold=0.25, text_threshold=0.2
            )
            
            if boxes is None or len(boxes) == 0:
                continue
            
            # Create frame directory
            frame_dir = output_path / f"frame_{frame_idx:04d}"
            frame_dir.mkdir(exist_ok=True)
            
            # Save original frame
            frame_path = frame_dir / "frame.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Segment each detected object with SAM2
            self.image_predictor.set_image(frame)
            
            frame_masks = []
            for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                # Convert box to SAM2 format (xyxy)
                box_xyxy = box  # Already in xyxy format
                
                # Segment with SAM2
                masks, scores, _ = self.image_predictor.predict(
                    box=box_xyxy[None, :],
                    multimask_output=False,
                )
                
                mask = masks[0]  # Take the first mask
                
                # Determine object type from phrase
                if 'hand' in phrase.lower():
                    obj_type = f"hand_{i}"
                else:
                    obj_type = f"tool_{i}"
                
                # Save mask
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_path = frame_dir / f"{obj_type}_mask.png"
                cv2.imwrite(str(mask_path), mask_uint8)
                
                # Save visualization
                vis = self._visualize_mask(frame, mask, i + 1)
                vis_path = frame_dir / f"{obj_type}_vis.png"
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                
                frame_masks.append({
                    'object_id': i,
                    'object_type': obj_type,
                    'phrase': phrase,
                    'confidence': float(logit),
                    'bbox': box.tolist(),
                    'mask_path': str(mask_path.relative_to(output_path)),
                    'vis_path': str(vis_path.relative_to(output_path))
                })
                total_masks += 1
            
            if frame_masks:
                all_masks.append({
                    'frame_idx': frame_idx,
                    'masks': frame_masks
                })
        
        print(f"\nProcessed {len(all_masks)} frames")
        
        # Save metadata
        metadata = {
            'video_path': str(video_path),
            'total_frames': total_frames,
            'fps': fps,
            'frames_sampled': len(frames_to_process),
            'frames_with_masks': len(all_masks),
            'sample_interval': sample_interval,
            'tool_names': tool_names,
            'text_prompt': text_prompt,
            'total_masks': total_masks,
            'timestamp': datetime.now().isoformat(),
            'method': 'grounding_dino_sam2'
        }
        
        with open(output_path / 'segmentation_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save detailed results
        with open(output_path / 'masks_details.json', 'w') as f:
            json.dump(all_masks, f, indent=2)
        
        print(f"✓ Segmentation complete! Saved {total_masks} masks")
        return {
            'total_masks': total_masks,
            'frames_processed': len(all_masks),
            'metadata': metadata
        }
    
    def auto_segment_video(self, video_path, output_dir, sample_interval=5, 
                          min_mask_area=1000, max_mask_area=100000, max_masks_per_frame=5):
        """
        Automatic segmentation without manual prompts
        Uses automatic mask generation on sampled frames
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for masks
            sample_interval: Sample every N frames (default: 5)
            min_mask_area: Minimum mask area in pixels (default: 1000)
            max_mask_area: Maximum mask area in pixels (default: 100000)
            max_masks_per_frame: Maximum masks to keep per frame (default: 5)
        
        Returns:
            Dictionary with segmentation results
        """
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nAuto-segmenting video: {video_path}")
        print(f"Sampling every {sample_interval} frames")
        
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_to_process = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_interval == 0:
                frames_to_process.append((frame_idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            
            frame_idx += 1
        
        cap.release()
        print(f"Processing {len(frames_to_process)} sampled frames (total: {total_frames})")
        
        # Use automatic mask generator with image predictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        mask_generator = SAM2AutomaticMaskGenerator(
            self.image_predictor.model,
            points_per_side=32,
            pred_iou_thresh=0.75,  # Higher threshold for better quality
            stability_score_thresh=0.90,  # Higher threshold for better quality
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )
        
        all_masks = []
        for frame_idx, frame in frames_to_process:
            print(f"Processing frame {frame_idx}...", end='\r')
            masks = mask_generator.generate(frame)
            
            # Filter masks - keep only hands and tools
            filtered_masks = self._filter_hand_tool_masks(
                masks, 
                min_mask_area, 
                max_mask_area,
                max_masks=max_masks_per_frame
            )
            
            if filtered_masks:
                all_masks.append({
                    'frame_idx': frame_idx,
                    'frame': frame,
                    'masks': filtered_masks
                })
        
        print(f"\nFound {len(all_masks)} frames with potential hand/tool masks")
        
        # Save results
        results = self._save_auto_masks(all_masks, output_path)
        
        # Save metadata
        metadata = {
            'video_path': str(video_path),
            'total_frames': total_frames,
            'fps': fps,
            'frames_sampled': len(frames_to_process),
            'frames_with_masks': len(all_masks),
            'sample_interval': sample_interval,
            'min_mask_area': min_mask_area,
            'max_mask_area': max_mask_area,
            'total_masks': results['total_masks'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path / 'segmentation_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Auto-segmentation complete! Saved {results['total_masks']} masks")
        return results
    
    def segment_frame(self, image, points, output_path=None):
        """
        Segment a single image/frame with point prompts
        
        Args:
            image: Image array (RGB)
            points: List of (x, y) points
            output_path: Optional path to save mask
        
        Returns:
            Binary mask array
        """
        self.image_predictor.set_image(image)
        
        masks, scores, logits = self.image_predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array([1] * len(points)),
            multimask_output=True,
        )
        
        # Use mask with highest score
        best_mask = masks[np.argmax(scores)]
        
        if output_path:
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            cv2.imwrite(str(output_path), mask_uint8)
        
        return best_mask
    
    def _load_video_frames(self, video_path):
        """Load all frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return frames
    
    def _filter_hand_tool_masks(self, masks, min_area=1000, max_area=100000, max_masks=5):
        """
        Filter masks to keep only likely hands and tools
        
        Args:
            masks: List of mask dictionaries from SAM2
            min_area: Minimum mask area (hands/tools are typically > 1000 pixels)
            max_area: Maximum mask area (hands/tools are typically < 100000 pixels)
            max_masks: Maximum number of masks to keep per frame (keep top N by quality)
        
        Returns:
            Filtered list of masks
        """
        filtered = []
        for mask in masks:
            area = mask['area']
            # Filter by size
            if min_area < area < max_area:
                # Filter by quality scores
                iou = mask.get('predicted_iou', 0)
                stability = mask.get('stability_score', 0)
                
                # Keep masks with good quality scores
                if iou > 0.7 and stability > 0.85:
                    filtered.append(mask)
        
        # Sort by area and quality, keep top N
        filtered.sort(key=lambda x: (x.get('predicted_iou', 0) * x['area']), reverse=True)
        return filtered[:max_masks]
    
    def _save_video_masks(self, video_segments, frames, output_dir, num_hands=0, num_tools=0):
        """Save segmentation masks from video propagation, organized by frame"""
        total_masks = 0
        
        for frame_idx, segments in video_segments.items():
            # Create frame directory
            frame_dir = output_dir / f"frame_{frame_idx:04d}"
            frame_dir.mkdir(exist_ok=True)
            
            for obj_id, mask in segments.items():
                # Determine object type
                if obj_id <= num_hands:
                    obj_type = f"hand_{obj_id}"
                else:
                    obj_type = f"tool_{obj_id - num_hands}"
                
                # Save as binary mask
                mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
                mask_path = frame_dir / f"{obj_type}_mask.png"
                cv2.imwrite(str(mask_path), mask_uint8)
                total_masks += 1
                
                # Also save visualization
                if frame_idx < len(frames):
                    vis = self._visualize_mask(frames[frame_idx], mask.squeeze(), obj_id)
                    vis_path = frame_dir / f"{obj_type}_vis.png"
                    cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        return {
            'total_masks': total_masks,
            'frames_processed': len(video_segments)
        }
    
    def _save_auto_masks(self, all_masks, output_dir):
        """Save automatically generated masks, organized by frame"""
        results = []
        total_masks = 0
        
        for frame_data in all_masks:
            frame_idx = frame_data['frame_idx']
            frame = frame_data['frame']
            masks = frame_data['masks']
            
            if not masks:
                continue
            
            # Create frame directory
            frame_dir = output_dir / f"frame_{frame_idx:04d}"
            frame_dir.mkdir(exist_ok=True)
            
            # Label masks as hand or tool based on characteristics
            for i, mask_data in enumerate(masks):
                mask = mask_data['segmentation']
                mask_uint8 = (mask * 255).astype(np.uint8)
                
                # Simple heuristic: larger masks are more likely to be hands
                # smaller ones are tools (can be improved with a classifier)
                area = mask_data['area']
                if area > 5000:  # Likely hand
                    obj_type = f"hand_{i}"
                else:  # Likely tool
                    obj_type = f"tool_{i}"
                
                # Save mask
                mask_path = frame_dir / f"{obj_type}_mask.png"
                cv2.imwrite(str(mask_path), mask_uint8)
                
                # Save visualization
                vis = self._visualize_mask(frame, mask, i + 1)
                vis_path = frame_dir / f"{obj_type}_vis.png"
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                
                # Save original frame for reference
                if i == 0:  # Only save once per frame
                    frame_path = frame_dir / "frame.jpg"
                    cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                results.append({
                    'frame_idx': frame_idx,
                    'object_id': i,
                    'object_type': obj_type,
                    'area': int(mask_data['area']),
                    'bbox': [int(x) for x in mask_data['bbox']],
                    'predicted_iou': float(mask_data.get('predicted_iou', 0)),
                    'stability_score': float(mask_data.get('stability_score', 0)),
                    'mask_path': str(mask_path.relative_to(output_dir)),
                    'vis_path': str(vis_path.relative_to(output_dir))
                })
                total_masks += 1
        
        # Save detailed results
        with open(output_dir / 'masks_details.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return {
            'total_masks': total_masks,
            'frames_processed': len(all_masks),
            'masks_per_frame': [len(m['masks']) for m in all_masks]
        }
    
    def _visualize_mask(self, image, mask, obj_id, alpha=0.5):
        """Overlay mask on image with color"""
        vis = image.copy()
        
        # Different colors for different objects
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
        ]
        
        color = np.array(colors[(obj_id - 1) % len(colors)])
        
        vis[mask > 0] = vis[mask > 0] * (1 - alpha) + color * alpha
        return vis.astype(np.uint8)


def process_all_segments(output_results_dir, model_name="facebook/sam2-hiera-large", 
                        checkpoint_path=None, mode='grounding_dino', sample_interval=5, max_masks_per_frame=3):
    """
    Process all video segments and generate masks
    
    Args:
        output_results_dir: Directory containing segmented videos (e.g., test_data/output_results)
        model_name: HuggingFace model name (default: "facebook/sam2-hiera-large")
        checkpoint_path: Optional local checkpoint path
        mode: 'grounding_dino' (recommended), 'auto', or 'manual'
        sample_interval: Sample every N frames (default: 5)
        max_masks_per_frame: Maximum masks to keep per frame for auto mode (default: 3)
    """
    use_grounding_dino = (mode == 'grounding_dino')
    segmenter = HandToolSegmenter(
        model_name=model_name, 
        checkpoint_path=checkpoint_path,
        use_grounding_dino=use_grounding_dino
    )
    
    results_path = Path(output_results_dir)
    
    if not results_path.exists():
        print(f"Error: Directory not found: {output_results_dir}")
        return
    
    total_processed = 0
    
    # Find all segment directories
    for platform_dir in results_path.iterdir():
        if not platform_dir.is_dir():
            continue
        
        for video_dir in platform_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            # Process each segment
            for seg_dir in sorted(video_dir.glob("segment_*")):
                # Find video file
                video_files = list(seg_dir.glob("*.mp4"))
                if not video_files:
                    continue
                
                video_file = video_files[0]
                
                print(f"\n{'='*60}")
                print(f"Processing: {platform_dir.name}/{video_dir.name}/{seg_dir.name}")
                print(f"Video: {video_file.name}")
                
                # Create masks directory
                masks_dir = seg_dir / "masks"
                masks_dir.mkdir(exist_ok=True)
                
                try:
                    if mode == 'grounding_dino':
                        # Use Grounding DINO + SAM2 (recommended)
                        segmenter.segment_video_with_grounding_dino(
                            str(video_file),
                            str(masks_dir),
                            segment_dir=str(seg_dir),
                            sample_interval=sample_interval
                        )
                    elif mode == 'auto':
                        # Auto segment (no manual prompts needed)
                        segmenter.auto_segment_video(
                            str(video_file), 
                            str(masks_dir),
                            sample_interval=sample_interval,
                            max_masks_per_frame=max_masks_per_frame
                        )
                    else:
                        # Manual mode - would need user to provide points
                        print("Manual mode not implemented in batch processing")
                        print("Use auto mode or process individual videos")
                        continue
                    
                    total_processed += 1
                    print(f"✓ Masks saved to: {masks_dir}")
                    
                except Exception as e:
                    print(f"✗ Error processing {seg_dir.name}: {e}")
                    continue
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Total segments processed: {total_processed}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Segment hands and tools from video using Grounding DINO + SAM2')
    parser.add_argument('--output-dir', type=str, default='test_data/output_results',
                       help='Directory containing segmented videos')
    parser.add_argument('--model-name', type=str, default='facebook/sam2-hiera-large',
                       help='HuggingFace model name (default: facebook/sam2-hiera-large)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sam2_hiera_large.pt',
                       help='Local checkpoint path (optional, will use HuggingFace if not found)')
    parser.add_argument('--mode', type=str, choices=['grounding_dino', 'auto', 'manual'], default='grounding_dino',
                       help='Segmentation mode: grounding_dino (recommended), auto, or manual')
    parser.add_argument('--sample-interval', type=int, default=5,
                       help='Sample every N frames (default: 5)')
    parser.add_argument('--max-masks', type=int, default=3,
                       help='Maximum masks per frame for auto mode (default: 3)')
    parser.add_argument('--single-segment', type=str, default=None,
                       help='Process single segment (format: platform/video_id/segment_N)')
    
    args = parser.parse_args()
    
    if args.single_segment:
        # Process single segment
        use_grounding_dino = (args.mode == 'grounding_dino')
        segmenter = HandToolSegmenter(
            model_name=args.model_name, 
            checkpoint_path=args.checkpoint,
            use_grounding_dino=use_grounding_dino
        )
        
        segment_path = Path(args.output_dir) / args.single_segment
        if not segment_path.exists():
            print(f"Error: Segment not found: {segment_path}")
            return
        
        video_files = list(segment_path.glob("*.mp4"))
        if not video_files:
            print(f"Error: No video file found in {segment_path}")
            return
        
        video_file = video_files[0]
        masks_dir = segment_path / "masks"
        
        print(f"Processing single segment: {args.single_segment}")
        
        if args.mode == 'grounding_dino':
            segmenter.segment_video_with_grounding_dino(
                str(video_file),
                str(masks_dir),
                segment_dir=str(segment_path),
                sample_interval=args.sample_interval
            )
        else:
            segmenter.auto_segment_video(
                str(video_file),
                str(masks_dir),
                sample_interval=args.sample_interval,
                max_masks_per_frame=args.max_masks
            )
    else:
        # Batch process all segments
        process_all_segments(
            args.output_dir,
            model_name=args.model_name,
            checkpoint_path=args.checkpoint,
            mode=args.mode,
            sample_interval=args.sample_interval,
            max_masks_per_frame=args.max_masks
        )


if __name__ == "__main__":
    main()
