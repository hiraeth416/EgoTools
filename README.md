# EgoTools

用于收集第一人称视角的工具使用视频数据集的工具。

## 功能特点

- 自动搜索并收集YouTube上的第一人称工具使用视频
- 支持多个相关关键词搜索
- 自动去重并保存元数据
- 实时显示搜索进度和统计信息
- 支持视频下载功能
- 增量保存数据，防止中途中断导致数据丢失
- 详细的时长统计（总时长、每个关键词的时长等）

## 依赖安装

TBD

## 使用方法

### 基础使用

只收集视频元数据：

```bash
python get_data.py --output dataset --max_results 100 --platforms youtube,bilibili,tiktok
```

### 参数说明

- `--output`: 指定输出目录，默认为 "datasets"
- `--max_results`: 每个关键词搜索的最大结果数，默认为 1000
- `--platforms`: 搜索视频的平台，默认为youtube

## 输出说明

程序会在指定的输出目录下创建一个以时间戳命名的子目录，包含：

1. `metadata.csv`: 包含所有收集到的视频元数据
   - 视频ID
   - 标题
   - 描述
   - 时长
   - 对应的搜索关键词


## 搜索关键词

当前包含以下搜索关键词：
- egocentric tool usage
- first person view DIY
- POV crafting tutorial
- first person woodworking
- POV cooking techniques
- first person tool demonstration
- egocentric view workshop
- POV handcraft tutorial
- hands on tutorial tools
- POV maker tutorial
- first person crafting
- egocentric DIY
- POV tools hands
- first person making

