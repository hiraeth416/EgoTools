import os
import pandas as pd
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from typing import List, Optional

class VideoDownloader:
    """视频下载器类"""
    def __init__(self, output_dir: str, max_height: int = 1080):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 基础下载选项
        self.max_height = max_height  # 存储最大高度值
        self.ydl_opts = {
            'writeinfojson': True,
            'writethumbnail': True,
            'ignoreerrors': True,
            'no_warnings': True,
            'quiet': False  # 默认不启用安静模式
        }

    def _build_ydl_args(self, url: str, platform: str) -> List[str]:
        """构建yt-dlp命令行参数"""
        args = []
        # 创建平台特定的输出目录
        platform_dir = os.path.join(self.output_dir, platform)
        os.makedirs(platform_dir, exist_ok=True)
        
        # 设置输出模板
        output_template = os.path.join(platform_dir, '%(title)s-%(id)s.%(ext)s')
        
        # 基础参数
        args.extend([
            '--format', f'bestvideo[height<={self.max_height}]+bestaudio/best[height<={self.max_height}]',
            '-o', output_template,
        ])
        
        # 添加其他选项
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
        
        # 添加URL
        args.append(url)
        return args

    def download_video(self, video_info: dict) -> bool:
        """
        下载单个视频
        Args:
            video_info: 包含视频信息的字典，必须包含 'id', 'platform' 字段
        """
        try:
            platform = str(video_info.get('platform', '')).lower()
            video_id = str(video_info.get('id', ''))
            
            if not platform or not video_id:
                print(f"无效的视频信息: 缺少平台或ID信息")
                return False
            
            # 验证平台是否支持
            if platform not in ['youtube', 'bilibili', 'tiktok']:
                print(f"不支持的平台: {platform}")
                return False
        except Exception as e:
            print(f"加载信息出错 {video_info}: {str(e)}")
            return False
        
        # 根据平台构建视频URL
        if platform == 'youtube':
            url = f"https://www.youtube.com/watch?v={video_id}"
        elif platform == 'bilibili':
            if video_id.startswith('BV') or video_id.startswith('bv'):
                url = f"https://www.bilibili.com/video/{video_id}"
            else:
                url = f"https://www.bilibili.com/video/av{video_id}"
        elif platform == 'tiktok':
            url = video_id  # TikTok的ID就是完整URL
        else:
            print(f"不支持的平台: {platform}")
            return False

        try:
            command = ['yt-dlp'] + self._build_ydl_args(url, platform)
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            else:
                print(f"下载失败 {url}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"下载出错 {url}: {str(e)}")
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
    从metadata.csv文件中下载视频
    Args:
        metadata_path: metadata.csv的路径
        output_dir: 视频保存目录
        max_workers: 并行下载的线程数
        max_height: 视频最大高度
        platforms: 要下载的平台列表，为None则下载所有平台
        start_index: 从第几个视频开始下载
        max_videos: 最多下载多少个视频，为None则下载所有视频
    """
    # 读取metadata
    df = pd.read_csv(metadata_path)
    print(f"读取到 {len(df)} 条视频记录")
    
    # 筛选平台
    if platforms:
        df = df[df['platform'].isin(platforms)]
        print(f"筛选后剩余 {len(df)} 条记录")
    
    # 处理下载范围
    if max_videos:
        df = df.iloc[start_index:start_index + max_videos]
    else:
        df = df.iloc[start_index:]
    
    if df.empty:
        print("没有需要下载的视频")
        return
    
    print(f"\n将下载 {len(df)} 个视频:")
    for platform in df['platform'].unique():
        count = len(df[df['platform'] == platform])
        print(f"- {platform}: {count} 个视频")
    
    # 初始化下载器
    downloader = VideoDownloader(output_dir, max_height)
    
    # 确保所需的列都存在
    required_columns = ['id', 'platform']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误：metadata.csv 缺少必要的列: {', '.join(missing_columns)}")
        return
    
    # 验证平台信息
    valid_platforms = ['youtube', 'bilibili', 'tiktok']
    df['platform'] = df['platform'].str.lower()
    invalid_platforms = df[~df['platform'].isin(valid_platforms)]
    if not invalid_platforms.empty:
        print(f"警告：发现不支持的平台，这些记录将被跳过：")
        print(invalid_platforms['platform'].unique())
        df = df[df['platform'].isin(valid_platforms)]
        if df.empty:
            print("没有有效的视频记录可供下载")
            return
    
    # 创建视频信息列表
    video_infos = df.to_dict('records')
    
    print(f"\n实际可下载视频数：{len(df)}")
    print("各平台视频数量：")
    for platform in valid_platforms:
        count = len(df[df['platform'] == platform])
        if count > 0:
            print(f"- {platform}: {count} 个视频")
    
    # 开始下载
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(downloader.download_video, video_infos),
            total=len(video_infos),
            desc="下载进度"
        ))
    
    # 统计结果
    success_count = sum(results)
    print(f"\n下载完成: {success_count}/{len(video_infos)} 个视频成功下载")
    
    # 按平台统计
    for platform in df['platform'].unique():
        platform_videos = df[df['platform'] == platform]
        platform_results = results[:len(platform_videos)]
        results = results[len(platform_videos):]
        success = sum(platform_results)
        print(f"{platform}: {success}/{len(platform_videos)} 个视频成功下载")

def main():
    parser = argparse.ArgumentParser(description='从metadata.csv下载视频')
    parser.add_argument('--csv_path', type=str, required=True, help='metadata.csv文件路径')
    parser.add_argument('--output', type=str, default='downloads',
                        help='视频保存目录')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='并行下载的线程数')
    parser.add_argument('--max-height', type=int, default=1080,
                        help='视频最大高度')
    parser.add_argument('--platforms', type=str,
                        help='要下载的平台，用逗号分隔，例如：youtube,bilibili')
    parser.add_argument('--start-index', type=int, default=0,
                        help='从第几个视频开始下载（0表示从头开始）')
    parser.add_argument('--max-videos', type=int,
                        help='最多下载多少个视频')
    
    args = parser.parse_args()
    
    # 处理平台参数
    platforms = None
    if args.platforms:
        platforms = [p.strip().lower() for p in args.platforms.split(',')]
    
    # 开始下载
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