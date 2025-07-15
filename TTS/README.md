# 李白语音转写服务

基于OpenAI Whisper模型的语音识别服务，支持多种语音转文本方式，可作为独立服务或与OpenWebUI集成。

## 项目结构

```
李白大模型/
├── cache/              # 缓存目录，存储临时音频文件
│   └── audio/          # 音频缓存目录
├── models/             # 模型存储目录
├── speech_service.py   # 语音服务核心模块，提供多种语音识别方式
├── whisper_api.py      # FastAPI实现的Whisper API服务
├── requirements.txt    # 项目依赖
└── README.md           # 项目文档
```

## 特性

- 支持多种语音转写引擎:
  - 本地Whisper模型 (基于faster-whisper)
  - 远程Whisper API服务
  - OpenAI Whisper API
  - Deepgram API
- 模块化设计，便于切换和扩展不同的语音识别服务
- 支持多种模型规格 (tiny, base, small, medium, large等)
- 自动检测语音语言
- 支持CUDA加速
- 兼容OpenWebUI接口

## 安装

1. 克隆仓库
```bash
git clone <仓库地址>
cd 李白大模型
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. (可选) 如果使用CUDA加速，请确保已安装CUDA和cuDNN

## 配置

项目通过环境变量进行配置，可以设置以下环境变量:

```bash
# 基本配置
WHISPER_MODEL=large-v3      # 模型规格 (tiny, base, small, medium, large, large-v2, large-v3)
DEVICE_TYPE=cuda            # 设备类型 (cuda, cpu)
WHISPER_MODEL_DIR=./models  # 模型下载和缓存目录
CACHE_DIR=./cache           # 缓存目录
WHISPER_LANGUAGE=zh         # 默认语言 (不设置则自动检测)
WHISPER_VAD_FILTER=True     # 是否使用语音活动检测过滤

# 外部服务配置 (可选)
EXTERNAL_WHISPER_URL=http://localhost:8000  # 外部Whisper服务URL
OPENAI_API_KEY=your-api-key                 # OpenAI API密钥
OPENAI_API_BASE_URL=https://api.openai.com  # OpenAI API基础URL
DEEPGRAM_API_KEY=your-api-key               # Deepgram API密钥
```

你也可以创建`.env`文件，项目会自动加载其中的配置。

## 使用方法

### 作为独立服务运行

```bash
# 启动Whisper API服务
python whisper_api.py
```

服务启动后，可通过以下端点访问:

- `GET /`: API欢迎页
- `GET /health`: 健康检查
- `GET /services`: 列出所有可用语音服务
- `POST /transcribe`: 转写音频文件
- `POST /change-service`: 切换当前使用的语音服务

### API使用示例

转写音频:
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "language=zh"
```

### 与OpenWebUI集成

1. 确保服务已启动
2. 在OpenWebUI的管理设置中配置:
   - STT引擎: `external_whisper`
   - 外部Whisper服务URL: `http://localhost:8000`

## 作为模块使用

```python
from speech_service import create_speech_service_manager

# 创建服务管理器
config = {
    "enable_local_whisper": True,
    "local_whisper_model": "large-v3",
    "device_type": "cuda",
    "whisper_model_dir": "./models",
    "external_whisper_url": "http://localhost:8000",
    "default_service": "local_whisper"
}

manager = create_speech_service_manager(config)

# 列出可用服务
services = manager.list_services()
print(f"可用服务: {services}")

# 转写音频
result = manager.transcribe("sample.wav", service_id="local_whisper")
print(f"转写结果: {result}")
```

## 切换模型

### 方法1: 通过环境变量

修改`WHISPER_MODEL`环境变量，然后重启服务。

### 方法2: 通过API

```bash
curl -X POST "http://localhost:8000/change-service" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"service_id": "remote_whisper"}'
```

## 性能优化

- 对于更快的转写速度但精度较低，使用较小的模型 (tiny, base)
- 对于更高的精度但速度较慢，使用较大的模型 (medium, large)
- 使用CUDA可以显著提高转写速度

## 故障排除

常见问题:

1. **模型下载失败**
   - 检查网络连接
   - 设置代理
   - 手动下载模型并放入models目录

2. **CUDA错误**
   - 确认CUDA和PyTorch版本匹配
   - 尝试使用CPU (`DEVICE_TYPE=cpu`)

3. **音频格式不支持**
   - 确保ffmpeg已正确安装
   - 转换音频为常见格式 (wav, mp3)

## 许可证

[MIT License](LICENSE) 