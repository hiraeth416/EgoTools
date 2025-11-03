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

确保已安装以下依赖：

```bash
pip install yt-dlp pandas tqdm
```

## 使用方法

### 基础使用

只收集视频元数据（不下载视频）：

```bash
python get_data.py --output dataset
```

### 收集元数据并下载视频

```bash
python get_data.py --output dataset --download
```

### 高级选项

可以通过命令行参数自定义行为：

```bash
python get_data.py --output dataset \  # 指定输出目录
                   --download \        # 是否下载视频
                   --max_results 500 \ # 每个关键词的最大结果数
                   --max_workers 8     # 下载视频时的并行线程数
```

### 参数说明

- `--output`: 指定输出目录，默认为 "downloads"
- `--max_results`: 每个关键词搜索的最大结果数，默认为 1000
- `--download`: 是否下载视频文件，默认不下载
- `--max_workers`: 下载视频时的并行线程数，默认为 4

## 输出说明

程序会在指定的输出目录下创建一个以时间戳命名的子目录，包含：

1. `metadata.csv`: 包含所有收集到的视频元数据
   - 视频ID
   - 标题
   - 描述
   - 时长
   - 对应的搜索关键词

2. 如果启用了下载选项，视频文件会保存在同一目录下

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

## 注意事项

1. 确保有足够的磁盘空间存储下载的视频
2. 如果网络不稳定，建议使用代理
3. 程序支持中断后继续，已下载的数据不会丢失
4. 视频元数据会实时保存到CSV文件中
5. 下载大量视频时建议使用较小的 `max_results` 值进行测试

## 使用代理

如果需要使用代理，可以设置环境变量：

```bash
export HTTP_PROXY="http://your-proxy:port"
export HTTPS_PROXY="http://your-proxy:port"
```

然后再运行脚本。
