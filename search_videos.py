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
    Search videos on Bilibili
    """
    try:
        # Use yt-dlp to search Bilibili
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
        print(f"Bilibili search failed: {str(e)}")
    return pd.DataFrame()

def search_tiktok(search_query: str, max_results: int = 500) -> pd.DataFrame:
    """
    Search videos on TikTok
    """
    try:
        # Convert search terms to TikTok tag format
        search_terms = search_query.replace('#', '').split()
        # Combine multiple words into a single tag
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
            print(f"TikTok tag '{tag}' search failed: {str(e)}")
        
        if all_results:
            df = pd.DataFrame(all_results)
            df['platform'] = 'tiktok'
            return df
    except Exception as e:
        print(f"TikTok search failed: {str(e)}")
    return pd.DataFrame()

class Duration:
    """Video duration data class"""
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
            return f"{self.hours}hours{self.minutes}minutes{self.seconds}seconds"
        elif self.minutes > 0:
            return f"{self.minutes}minutes{self.seconds}seconds"
        else:
            return f"{self.seconds}seconds"

def parse_duration(duration_string):
    """
    Parse duration string to Duration object
    Supports the following formats:
    - "1:23:45" -> Duration(hours=1, minutes=23, seconds=45)
    - "5:30" -> Duration(minutes=5, seconds=30)
    - "42" -> Duration(seconds=42)
    - 123.45 (float seconds) -> Duration(minutes=2, seconds=3)
    """
    if duration_string is None or pd.isna(duration_string):
        return Duration()

    try:
        # If it's a numeric type (int or float), convert directly to seconds
        if isinstance(duration_string, (int, float)):
            total_seconds = int(duration_string)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return Duration(hours=hours, minutes=minutes, seconds=seconds)
        
        # If it's a string, try to parse as HH:MM:SS format
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
                    print(f"Warning: Unable to parse duration format '{duration_string}'")
                    return Duration()
    except (ValueError, TypeError) as e:
        print(f"Warning: Unable to parse duration '{duration_string}': {str(e)}")
        return Duration()
    
    return Duration()

def search_youtube(search_query: str, max_results: int = 500) -> pd.DataFrame:
    """
    Search videos on YouTube
    """
    print(f"Searching YouTube for: '{search_query}'")
    
    # Handle keywords with hashtags
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
                timeout=120  # Increase overall timeout to 120 seconds
            )
            
            if result.stdout.strip():
                raw_data = [
                    json.loads(line) 
                    for line in result.stdout.strip().split('\n') 
                    if line.strip()
                ]
                all_results.extend(raw_data)
                print(f"Found {len(raw_data)} entries from YouTube search")
            break  # Exit retry loop on success
            
        except subprocess.TimeoutExpired:
            print(f"Attempt {attempt+1} timed out")
            if attempt < 2:  # If not the last attempt
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
            print(f"JSON parsing error: {str(e)}")
            continue
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            break
    
    if all_results:
        df = pd.DataFrame(all_results)
        df['platform'] = 'youtube'  # Add platform information
        print(f"Successfully collected {len(df)} total entries")
        return df
    return pd.DataFrame()

def run_platform_search(search_query: str, platform: str, max_results: int = 500) -> pd.DataFrame:
    """
    Search videos on specified platform
    """
    if platform == 'youtube':
        return search_youtube(search_query, max_results)
    elif platform == 'bilibili':
        return search_bilibili(search_query, max_results)
    elif platform == 'tiktok':
        return search_tiktok(search_query, max_results)
    return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='Collect metadata for first-person tool usage video dataset')
    parser.add_argument('--output', type=str, default='datasets',
                        help='Output directory')
    parser.add_argument('--max_results', type=int, default=1000,
                        help='Maximum number of results per keyword search')
    parser.add_argument('--platforms', type=str, default='youtube',
                        help='Platforms to search, comma-separated. Supported: youtube,bilibili,tiktok')
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Search keywords list
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

    # Process platform arguments
    platforms = [p.strip().lower() for p in args.platforms.split(',')]
    valid_platforms = ['youtube', 'bilibili', 'tiktok']
    platforms = [p for p in platforms if p in valid_platforms]
    
    if not platforms:
        print("No valid platforms specified, defaulting to YouTube")
        platforms = ['youtube']
    
    print(f"Will search on the following platforms: {', '.join(platforms)}")
    
    # Create metadata file
    metadata_path = os.path.join(output_dir, "metadata.csv")
    total_duration = Duration()  # Initialize as Duration object
    total_videos = 0
    platform_stats = {p: {'videos': 0, 'duration': Duration()} for p in platforms}
    
    # Create an empty DataFrame to store all metadata
    columns = ['id', 'title', 'description', 'duration_string', 'search_keyword', 'platform']
    final_df = pd.DataFrame(columns=columns)
    final_df.to_csv(metadata_path, index=False)
    
    for kw in tqdm(KEYWORDS, desc="Searching keywords"):
        for platform in platforms:
            print(f"\nSearching on {platform} for: {kw}")
            df = run_platform_search(kw, platform, max_results=args.max_results)
            if not df.empty:
                # Filter and process data
                # First ensure all necessary columns exist
                df_filtered = df.copy()
                
                                # Filter and process data
                # First ensure all necessary columns exist
                df_filtered = df.copy()
                
                # Add or retain necessary columns
                df_filtered['platform'] = platform  # Use current platform
                df_filtered['search_keyword'] = kw  # Use current search keyword
                
                # Select final required columns in specified order
                df_filtered = df_filtered[['id', 'title', 'description', 'duration_string', 'search_keyword', 'platform']]
                
                # Calculate duration of new videos
                new_videos = df_filtered[~df_filtered['id'].isin(final_df['id'])]
                
                if not new_videos.empty:
                    new_duration = Duration()
                    for d in new_videos['duration_string']:
                        new_duration = new_duration + parse_duration(d)
                
                # Update total statistics
                total_duration = total_duration + new_duration
                total_videos += len(new_videos)
                
                # Update platform statistics
                curr_platform = new_videos['platform'].iloc[0]  # Get current platform
                platform_stats[curr_platform]['videos'] += len(new_videos)
                platform_stats[curr_platform]['duration'] = platform_stats[curr_platform]['duration'] + new_duration
                
                # Append new data to CSV file
                new_videos.to_csv(metadata_path, mode='a', header=False, index=False)
                
                # Update DataFrame in memory
                final_df = pd.concat([final_df, new_videos], ignore_index=True)
                
                print(f"\nPlatform '{curr_platform}' keyword '{kw}' added {len(new_videos)} videos, total duration {new_duration}")
                print(f"Current platform total {platform_stats[curr_platform]['videos']} videos, "
                      f"total duration {platform_stats[curr_platform]['duration']}")
                print(f"All platforms total {total_videos} videos, total duration {total_duration}")
    
    print("\n=== Final Statistics ===")
    print(f"Total videos: {total_videos}")
    print(f"Total duration: {total_duration}")
    print("\nStatistics by platform:")
    for platform, stats in platform_stats.items():
        print(f"{platform}: {stats['videos']} videos, total duration {stats['duration']}")

if __name__ == "__main__":
    main()