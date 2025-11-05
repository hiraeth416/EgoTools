import os
import pandas as pd
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from typing import List, Optional

class VideoDownloader:
    """Video downloader class"""
    def __init__(self, output_dir: str, max_height: int = 1080):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Basic download options
        self.max_height = max_height  # Store maximum height value
        self.ydl_opts = {
            'writeinfojson': True,
            'writethumbnail': True,
            'ignoreerrors': True,
            'no_warnings': True,
            'quiet': False  # Quiet mode is disabled by default
        }

    def _build_ydl_args(self, url: str, platform: str) -> List[str]:
        """Build yt-dlp command line arguments"""
        args = []
        # Create platform-specific output directory
        platform_dir = os.path.join(self.output_dir, platform)
        os.makedirs(platform_dir, exist_ok=True)
        
        # Set output template - each video in a separate folder named by ID
        output_template = os.path.join(platform_dir, '%(id)s', '%(title)s-%(id)s.%(ext)s')
        
        # Basic arguments
        args.extend([
            '--format', f'bestvideo[height<={self.max_height}]+bestaudio/best[height<={self.max_height}]',
            '-o', output_template,
            '--cookies-from-browser', 'chrome',
            '--cookies', 'cookies.txt',
        ])
        
        # Add other options
        if self.ydl_opts.get('writeinfojson', False):
            args.append('--write-info-json')
        if self.ydl_opts.get('writethumbnail', False):
            args.append('--write-thumbnail')
        if self.ydl_opts.get('ignoreerrors', False):
            args.append('--ignore-errors')
        if self.ydl_opts.get('no_warnings', False):
            args.append('--no-warnings')
        if self.ydl_opts.get('quiet', False):
            args.append('--quiet')
        if self.ydl_opts['no_warnings']:
            args.append('--no-warnings')
        
        # Add URL
        args.append(url)
        return args

    def download_video(self, video_info: dict) -> bool:
        """
        Download a single video
        Args:
            video_info: Dictionary containing video information, must include 'id' and 'platform' fields
        """
        try:
            platform = str(video_info.get('platform', '')).lower()
            video_id = str(video_info.get('id', ''))
            
            if not platform or not video_id:
                print(f"Invalid video info: missing platform or ID information")
                return False
            
            # Validate if platform is supported
            if platform not in ['youtube', 'bilibili', 'tiktok']:
                print(f"Unsupported platform: {platform}")
                return False
        except Exception as e:
            print(f"Error loading info {video_info}: {str(e)}")
            return False
        
        # Build video URL based on platform
        if platform == 'youtube':
            url = f"https://www.youtube.com/watch?v={video_id}"
        elif platform == 'bilibili':
            if video_id.startswith('BV') or video_id.startswith('bv'):
                url = f"https://www.bilibili.com/video/{video_id}"
            else:
                url = f"https://www.bilibili.com/video/av{video_id}"
        elif platform == 'tiktok':
            url = video_id  # TikTok's ID is the complete URL
        else:
            print(f"Unsupported platform: {platform}")
            return False

        try:
            command = ['yt-dlp'] + self._build_ydl_args(url, platform)
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            else:
                print(f"Download failed {url}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Download error {url}: {str(e)}")
            return False

def download_from_metadata(
    metadata_path: str,
    output_dir: str,
    max_workers: int = 4,
    max_height: int = 1080,
    platforms: Optional[List[str]] = None,
    start_index: int = 0,
    max_videos: Optional[int] = None
) -> None:
    """
    Download videos from metadata.csv file
    Args:
        metadata_path: Path to metadata.csv
        output_dir: Video save directory
        max_workers: Number of parallel download threads
        max_height: Maximum video height
        platforms: List of platforms to download, None downloads all platforms
        start_index: Starting video index for download
        max_videos: Maximum number of videos to download, None downloads all videos
    """
    # Read metadata
    df = pd.read_csv(metadata_path)
    print(f"Read {len(df)} video records")
    
    # Filter platforms
    if platforms:
        df = df[df['platform'].isin(platforms)]
        print(f"Remaining {len(df)} records after filtering")
    
    # Process download range
    if max_videos:
        df = df.iloc[start_index:start_index + max_videos]
    else:
        df = df.iloc[start_index:]
    
    if df.empty:
        print("No videos to download")
        return
    
    print(f"\nWill download {len(df)} videos:")
    for platform in df['platform'].unique():
        count = len(df[df['platform'] == platform])
        print(f"- {platform}: {count} videos")
    
    # Initialize downloader
    downloader = VideoDownloader(output_dir, max_height)
    
    # Ensure required columns exist
    required_columns = ['id', 'platform']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: metadata.csv missing required columns: {', '.join(missing_columns)}")
        return
    
    # Validate platform information
    valid_platforms = ['youtube', 'bilibili', 'tiktok']
    df['platform'] = df['platform'].str.lower()
    invalid_platforms = df[~df['platform'].isin(valid_platforms)]
    if not invalid_platforms.empty:
        print(f"Warning: Found unsupported platforms, these records will be skipped:")
        print(invalid_platforms['platform'].unique())
        df = df[df['platform'].isin(valid_platforms)]
        if df.empty:
            print("No valid video records available for download")
            return
    
    # Create video info list
    video_infos = df.to_dict('records')
    
    print(f"\nActual downloadable videos: {len(df)}")
    print("Video count by platform:")
    for platform in valid_platforms:
        count = len(df[df['platform'] == platform])
        if count > 0:
            print(f"- {platform}: {count} videos")
    
    # Start downloading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(downloader.download_video, video_infos),
            total=len(video_infos),
            desc="Download progress"
        ))
    
    # Statistics
    success_count = sum(results)
    print(f"\nDownload complete: {success_count}/{len(video_infos)} videos successfully downloaded")
    
    # Statistics by platform
    for platform in df['platform'].unique():
        platform_videos = df[df['platform'] == platform]
        platform_results = results[:len(platform_videos)]
        results = results[len(platform_videos):]
        success = sum(platform_results)
        print(f"{platform}: {success}/{len(platform_videos)} videos successfully downloaded")

def main():
    parser = argparse.ArgumentParser(description='Download videos from metadata.csv')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to metadata.csv file')
    parser.add_argument('--output', type=str, default='downloads',
                        help='Video save directory')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Number of parallel download threads')
    parser.add_argument('--max-height', type=int, default=1080,
                        help='Maximum video height')
    parser.add_argument('--platforms', type=str,
                        help='Platforms to download, comma-separated, e.g.: youtube,bilibili')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting video index for download (0 to start from beginning)')
    parser.add_argument('--max-videos', type=int,
                        help='Maximum number of videos to download')
    
    args = parser.parse_args()
    
    # Process platform arguments
    platforms = None
    if args.platforms:
        platforms = [p.strip().lower() for p in args.platforms.split(',')]
    
    # Start downloading
    download_from_metadata(
        metadata_path=args.csv_path,
        output_dir=args.output,
        max_workers=args.max_workers,
        max_height=args.max_height,
        platforms=platforms,
        start_index=args.start_index,
        max_videos=args.max_videos
    )

if __name__ == "__main__":
    main()