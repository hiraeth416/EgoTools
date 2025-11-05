import json
import subprocess
import pandas as pd
import time
from tqdm import tqdm
import os
from datetime import datetime
import argparse
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

def search_bilibili(search_query: str, max_results: int = 500) -> pd.DataFrame:
    """
    在Bilibili上搜索视频
    """
    try:
        # 使用yt-dlp搜索Bilibili
        command = [
            'yt-dlp',
            '--skip-download',
            '--flat-playlist',
            '--dump-json',
            '--playlist-items', f'1-{max_results}',
            f'https://search.bilibili.com/all?keyword={search_query}'
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
        
        if result.stdout.strip():
            raw_data = [
                json.loads(line)
                for line in result.stdout.strip().split('\n')
                if line.strip()
            ]
            df = pd.DataFrame(raw_data)
            df['platform'] = 'bilibili'
            return df
    except Exception as e:
        print(f"Bilibili搜索失败: {str(e)}")
    return pd.DataFrame()

def search_tiktok(search_query: str, max_results: int = 500) -> pd.DataFrame:
    """
    在TikTok上搜索视频
    """
    try:
        # 将搜索词转换为TikTok标签格式
        search_terms = search_query.replace('#', '').split()
        # 将多个词组合成一个标签
        tag = ''.join(word.capitalize() for word in search_terms)
        
        all_results = []
        command = [
            'yt-dlp',
            '--skip-download',
            '--flat-playlist',
            '--dump-json',
            '--playlist-items', f'1-{max_results}',
            f'https://www.tiktok.com/tag/{tag}'
        ]
            
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
            if result.stdout.strip():
                raw_data = [
                    json.loads(line)
                    for line in result.stdout.strip().split('\n')
                    if line.strip()
                ]
                all_results.extend(raw_data)
        except Exception as e:
            print(f"TikTok标签 '{tag}' 搜索失败: {str(e)}")
        
        if all_results:
            df = pd.DataFrame(all_results)
            df['platform'] = 'tiktok'
            return df
    except Exception as e:
        print(f"TikTok搜索失败: {str(e)}")
    return pd.DataFrame()

class Duration:
    """视频时长的数据类"""
    def __init__(self, hours=0, minutes=0, seconds=0):
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
    
    def __add__(self, other):
        total_seconds = (self.hours * 3600 + self.minutes * 60 + self.seconds +
                        other.hours * 3600 + other.minutes * 60 + other.seconds)
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return Duration(h, m, s)
    
    def __str__(self):
        if self.hours > 0:
            return f"{self.hours}小时{self.minutes}分钟{self.seconds}秒"
        elif self.minutes > 0:
            return f"{self.minutes}分钟{self.seconds}秒"
        else:
            return f"{self.seconds}秒"

def parse_duration(duration_string):
    """
    解析时长为Duration对象
    支持以下格式:
    - "1:23:45" -> Duration(hours=1, minutes=23, seconds=45)
    - "5:30" -> Duration(minutes=5, seconds=30)
    - "42" -> Duration(seconds=42)
    - 123.45 (浮点数秒) -> Duration(minutes=2, seconds=3)
    """
    if duration_string is None or pd.isna(duration_string):
        return Duration()

    try:
        # 如果是数字类型（整数或浮点数），直接转换为秒
        if isinstance(duration_string, (int, float)):
            total_seconds = int(duration_string)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return Duration(hours=hours, minutes=minutes, seconds=seconds)
        
        # 如果是字符串，尝试按时:分:秒格式解析
        if isinstance(duration_string, str):
            parts = duration_string.split(':')
            if len(parts) == 3:  # HH:MM:SS
                h, m, s = map(int, parts)
                return Duration(hours=h, minutes=m, seconds=s)
            elif len(parts) == 2:  # MM:SS
                m, s = map(int, parts)
                return Duration(minutes=m, seconds=s)
            elif len(parts) == 1:  # SS or float
                try:
                    seconds = int(float(parts[0]))
                    return Duration(seconds=seconds)
                except ValueError:
                    print(f"警告: 无法解析时长格式 '{duration_string}'")
                    return Duration()
    except (ValueError, TypeError) as e:
        print(f"警告: 无法解析时长 '{duration_string}': {str(e)}")
        return Duration()
    
    return Duration()

def search_youtube(search_query: str, max_results: int = 500) -> pd.DataFrame:
    """
    在YouTube上搜索视频
    """
    print(f"Searching YouTube for: '{search_query}'")
    
    # 处理包含hashtag的关键词
    if search_query.startswith('#'):
        search_query = search_query[1:]
    
    query = f"ytsearch{max_results}:{search_query}"
    
    all_results = []
    command = [
        'yt-dlp',
        '--skip-download',
        '--flat-playlist',
        '--dump-json',
        '--socket-timeout', '60',
        query
    ]
        
    for attempt in range(3):
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True, 
                timeout=120  # 增加整体超时时间到120秒
            )
            
            if result.stdout.strip():
                raw_data = [
                    json.loads(line) 
                    for line in result.stdout.strip().split('\n') 
                    if line.strip()
                ]
                all_results.extend(raw_data)
                print(f"Found {len(raw_data)} entries from YouTube search")
            break  # 成功则退出重试循环
            
        except subprocess.TimeoutExpired:
            print(f"Attempt {attempt+1} timed out")
            if attempt < 2:  # 如果不是最后一次尝试
                print("Waiting before retry...")
                time.sleep(5 + (5 * attempt))
            continue
            
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt+1} failed. Error: {e.stderr.strip()}")
            if attempt < 2:
                print("Waiting before retry...")
                time.sleep(5 + (5 * attempt))
            continue
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {str(e)}")
            continue
            
        except Exception as e:
            print(f"未预期的错误: {str(e)}")
            break
    
    if all_results:
        df = pd.DataFrame(all_results)
        df['platform'] = 'youtube'  # 添加平台信息
        print(f"Successfully collected {len(df)} total entries")
        return df
    return pd.DataFrame()

def run_platform_search(search_query: str, platform: str, max_results: int = 500) -> pd.DataFrame:
    """
    在指定平台上搜索视频
    """
    if platform == 'youtube':
        return search_youtube(search_query, max_results)
    elif platform == 'bilibili':
        return search_bilibili(search_query, max_results)
    elif platform == 'tiktok':
        return search_tiktok(search_query, max_results)
    return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='收集第一人称工具使用视频数据集的元数据')
    parser.add_argument('--output', type=str, default='datasets',
                        help='输出目录')
    parser.add_argument('--max_results', type=int, default=1000,
                        help='每个关键词搜索的最大结果数')
    parser.add_argument('--platforms', type=str, default='youtube',
                        help='要搜索的平台，用逗号分隔。支持：youtube,bilibili,tiktok')
    args = parser.parse_args()

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # 搜索关键词列表
    KEYWORDS = {
        "POV tool usage",
        #"egocentric tool usage",
        #"first person view DIY",
        #"POV crafting tutorial",
        #"first person woodworking",
        #"POV cooking techniques",
        #"first person tool demonstration",
        #"egocentric view workshop",
        #"POV handcraft tutorial",
        #"hands on tutorial tools",
        #"POV maker tutorial",
        #"first person crafting",
        #"egocentric DIY",
        #"POV tools hands",
        #"first person making"
    }

    # 处理平台参数
    platforms = [p.strip().lower() for p in args.platforms.split(',')]
    valid_platforms = ['youtube', 'bilibili', 'tiktok']
    platforms = [p for p in platforms if p in valid_platforms]
    
    if not platforms:
        print("没有指定有效的平台，默认使用YouTube")
        platforms = ['youtube']
    
    print(f"将在以下平台搜索: {', '.join(platforms)}")
    
    # 创建metadata文件
    metadata_path = os.path.join(output_dir, "metadata.csv")
    total_duration = Duration()  # 初始化为Duration对象
    total_videos = 0
    platform_stats = {p: {'videos': 0, 'duration': Duration()} for p in platforms}
    
    # 创建一个空的DataFrame来存储所有元数据
    columns = ['id', 'title', 'description', 'duration_string', 'search_keyword', 'platform']
    final_df = pd.DataFrame(columns=columns)
    final_df.to_csv(metadata_path, index=False)
    
    for kw in tqdm(KEYWORDS, desc="搜索关键词"):
        for platform in platforms:
            print(f"\n在 {platform} 上搜索: {kw}")
            df = run_platform_search(kw, platform, max_results=args.max_results)
            if not df.empty:
                # 筛选并处理数据
                # 首先确保所有必要的列都存在
                df_filtered = df.copy()
                
                # 添加或保留必要的列
                df_filtered['platform'] = platform  # 使用当前平台
                df_filtered['search_keyword'] = kw  # 使用当前搜索关键词
                
                # 选择最终需要的列，按照指定的顺序
                df_filtered = df_filtered[['id', 'title', 'description', 'duration_string', 'search_keyword', 'platform']]
                
                # 计算新增视频的时长
                new_videos = df_filtered[~df_filtered['id'].isin(final_df['id'])]
                
                if not new_videos.empty:
                    new_duration = Duration()
                    for d in new_videos['duration_string']:
                        new_duration = new_duration + parse_duration(d)
                
                # 更新总计数据
                total_duration = total_duration + new_duration
                total_videos += len(new_videos)
                
                # 更新平台统计
                curr_platform = new_videos['platform'].iloc[0]  # 获取当前平台
                platform_stats[curr_platform]['videos'] += len(new_videos)
                platform_stats[curr_platform]['duration'] = platform_stats[curr_platform]['duration'] + new_duration
                
                # 将新数据追加到CSV文件
                new_videos.to_csv(metadata_path, mode='a', header=False, index=False)
                
                # 更新内存中的DataFrame
                final_df = pd.concat([final_df, new_videos], ignore_index=True)
                
                print(f"\n平台 '{curr_platform}' 关键词 '{kw}' 新增 {len(new_videos)} 个视频，总时长 {new_duration}")
                print(f"当前平台共计 {platform_stats[curr_platform]['videos']} 个视频，"
                      f"总时长 {platform_stats[curr_platform]['duration']}")
                print(f"所有平台共计 {total_videos} 个视频，总时长 {total_duration}")
    
    print("\n=== 最终统计 ===")
    print(f"总视频数: {total_videos}")
    print(f"总时长: {total_duration}")
    print("\n各平台统计:")
    for platform, stats in platform_stats.items():
        print(f"{platform}: {stats['videos']} 个视频，总时长 {stats['duration']}")

if __name__ == "__main__":
    main()