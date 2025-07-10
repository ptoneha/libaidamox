# 高精度Whisper语音识别服务

这个项目提供了一个高精度的语音识别服务，使用OpenAI的Whisper模型，可以集成到Open WebUI系统中，提供比原有系统更高精度的语音转文字功能。

## 项目结构

- `whisper_service.py`: 独立的高精度Whisper语音识别服务
- `whisper_integration.py`: 集成工具，用于将Whisper服务与Open WebUI系统集成
- `external_whisper.py`: 生成的代理端点文件，用于添加到Open WebUI系统中
- `whisper_integration_instructions.md`: 详细的集成说明

## 特点

- 使用最新的Whisper large-v3模型，提供最高精度的语音识别
- 支持多种音频格式和自动格式转换
- 支持长音频文件的分块处理
- 与Open WebUI系统无缝集成
- 可作为独立服务运行，也可以集成到其他系统中

## 安装依赖

```bash
pip install fastapi uvicorn openai-whisper pydub python-multipart python-dotenv requests
```

## 使用方法

### 1. 启动独立的Whisper服务

```bash
python whisper_service.py
```

服务将在 http://localhost:8000 上运行。

### 2. 使用集成工具

```bash
python whisper_integration.py
```

集成工具将生成必要的文件和说明，帮助你将Whisper服务集成到Open WebUI系统中。

#### 命令行参数

- `--whisper-url`: Whisper服务URL (默认: http://localhost:8000)
- `--openwebui-url`: Open WebUI URL (默认: http://localhost:8080)
- `--admin-email`: 管理员邮箱 (默认: admin@example.com)
- `--admin-password`: 管理员密码
- `--test-only`: 仅测试连接
- `--generate-only`: 仅生成集成文件

### 3. 集成到Open WebUI

按照生成的 `whisper_integration_instructions.md` 文件中的说明进行操作，将高精度语音识别服务集成到Open WebUI系统中。

## API接口

### Whisper服务API

- `GET /`: 服务信息
- `GET /health`: 健康检查
- `POST /transcribe`: 转录音频文件
- `GET /models`: 列出可用的Whisper模型

### 示例：转录音频文件

```python
import requests

# 转录音频文件
with open('audio.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/transcribe',
        files={'file': f},
        data={'language': 'zh'}  # 可选，指定语言
    )

result = response.json()
print(result['text'])
```

## 配置

可以通过环境变量或.env文件配置以下参数：

- `WHISPER_SERVICE_URL`: Whisper服务URL
- `OPENWEBUI_URL`: Open WebUI URL
- `ADMIN_EMAIL`: 管理员邮箱
- `ADMIN_PASSWORD`: 管理员密码

## 注意事项

- 运行Whisper模型需要足够的计算资源，特别是large-v3模型
- 首次运行时会下载模型文件，可能需要一些时间
- 对于GPU加速，需要安装适当的CUDA和PyTorch版本

## 性能优化

- 使用GPU可以显著提高转录速度
- 可以通过修改`whisper_service.py`中的模型名称来平衡精度和速度
  - tiny/base: 速度快，精度低
  - small/medium: 平衡速度和精度
  - large/large-v2/large-v3: 精度高，速度慢

## 常见问题

### 1. 模型加载失败

确保你有足够的磁盘空间和内存。large-v3模型大小约为3GB。

### 2. 音频转录失败

检查音频文件格式是否支持，服务支持大多数常见音频格式，如mp3、wav、flac等。

### 3. 集成后无法使用

确保Whisper服务正在运行，并且Open WebUI系统可以访问该服务。检查网络连接和防火墙设置。

## 许可证

MIT 