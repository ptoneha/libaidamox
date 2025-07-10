import os
import tempfile
import logging
import uuid
from typing import Optional, List
from pathlib import Path
from pydub import AudioSegment

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("whisper-service")

# 创建FastAPI应用
app = FastAPI(
    title="高精度Whisper语音识别服务",
    description="提供高精度语音识别功能的API服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_CACHE_DIR = CACHE_DIR / "audio"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE_MB = 25  # 最大文件大小(MB)
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # 转换为字节

# 全局Whisper模型
whisper_model = None


def load_whisper_model(model_name: str = "large-v3"):
    """
    加载Whisper模型
    """
    global whisper_model
    
    try:
        import whisper
        logger.info(f"加载Whisper模型: {model_name}")
        whisper_model = whisper.load_model(model_name)
        logger.info(f"Whisper模型 {model_name} 加载成功")
        return True
    except Exception as e:
        logger.error(f"加载Whisper模型失败: {e}")
        return False


def is_audio_conversion_required(file_path):
    """
    检查音频文件是否需要转换为mp3格式
    """
    SUPPORTED_FORMATS = {"flac", "m4a", "mp3", "mp4", "mpeg", "wav", "webm"}

    if not os.path.isfile(file_path):
        logger.error(f"文件不存在: {file_path}")
        return False

    try:
        from pydub.utils import mediainfo
        info = mediainfo(file_path)
        codec_name = info.get("codec_name", "").lower()
        codec_type = info.get("codec_type", "").lower()
        codec_tag_string = info.get("codec_tag_string", "").lower()

        if codec_name == "aac" and codec_type == "audio" and codec_tag_string == "mp4a":
            # 文件是AAC/mp4a音频，建议转换为mp3
            return True

        # 如果编解码器名称在支持的格式中
        if codec_name in SUPPORTED_FORMATS:
            return False

        return True
    except Exception as e:
        logger.error(f"获取音频格式时出错: {e}")
        return False


def convert_audio_to_mp3(file_path):
    """
    将音频文件转换为mp3格式
    """
    try:
        output_path = os.path.splitext(file_path)[0] + ".mp3"
        audio = AudioSegment.from_file(file_path)
        audio.export(output_path, format="mp3")
        logger.info(f"已将 {file_path} 转换为 {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"转换音频文件时出错: {e}")
        return None


def compress_audio(file_path):
    """
    压缩音频文件
    """
    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        id = os.path.splitext(os.path.basename(file_path))[0]
        file_dir = os.path.dirname(file_path)

        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # 压缩音频

        compressed_path = os.path.join(file_dir, f"{id}_compressed.mp3")
        audio.export(compressed_path, format="mp3", bitrate="32k")
        logger.info(f"已压缩音频至 {compressed_path}")

        return compressed_path
    else:
        return file_path


def split_audio(file_path, max_bytes, format="mp3", bitrate="32k"):
    """
    将音频分割为不超过max_bytes的块
    返回块文件路径列表。如果音频适合，则返回包含原始路径的列表。
    """
    file_size = os.path.getsize(file_path)
    if file_size <= max_bytes:
        return [file_path]  # 无需分割

    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    orig_size = file_size

    approx_chunk_ms = max(int(duration_ms * (max_bytes / orig_size)) - 1000, 1000)
    chunks = []
    start = 0
    i = 0

    base, _ = os.path.splitext(file_path)

    while start < duration_ms:
        end = min(start + approx_chunk_ms, duration_ms)
        chunk = audio[start:end]
        chunk_path = f"{base}_chunk_{i}.{format}"
        chunk.export(chunk_path, format=format, bitrate=bitrate)

        # 如果仍然太大，则减少块持续时间
        while os.path.getsize(chunk_path) > max_bytes and (end - start) > 5000:
            end = start + ((end - start) // 2)
            chunk = audio[start:end]
            chunk.export(chunk_path, format=format, bitrate=bitrate)

        if os.path.getsize(chunk_path) > max_bytes:
            os.remove(chunk_path)
            raise Exception("音频块无法减小到最大文件大小以下。")

        chunks.append(chunk_path)
        start = end
        i += 1

    return chunks


def transcribe_with_whisper(file_path: str, language: Optional[str] = None):
    """
    使用Whisper模型转录音频文件
    """
    global whisper_model
    
    # 如果模型未加载，则加载模型
    if whisper_model is None:
        if not load_whisper_model():
            raise HTTPException(status_code=500, detail="无法加载Whisper模型")
    
    try:
        # 音频预处理
        if is_audio_conversion_required(file_path):
            file_path = convert_audio_to_mp3(file_path)
            if not file_path:
                raise HTTPException(status_code=400, detail="音频转换失败")
        
        try:
            file_path = compress_audio(file_path)
        except Exception as e:
            logger.exception(e)
        
        # 分割音频（如果需要）
        chunk_paths = split_audio(file_path, MAX_FILE_SIZE)
        logger.info(f"音频分块路径: {chunk_paths}")
        
        results = []
        try:
            for chunk_path in chunk_paths:
                # 使用Whisper模型进行转录
                transcribe_options = {
                    "fp16": False,  # 使用FP32以获得更高精度
                    "beam_size": 5,  # 增加beam search大小
                }
                
                if language:
                    transcribe_options["language"] = language
                
                result = whisper_model.transcribe(
                    chunk_path, 
                    **transcribe_options
                )
                
                results.append(result["text"])
        finally:
            # 清理临时块，但不删除原始文件
            for chunk_path in chunk_paths:
                if chunk_path != file_path and os.path.isfile(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except Exception:
                        pass
        
        return " ".join(results).strip()
    except Exception as e:
        logger.exception(f"转录时出错: {e}")
        raise HTTPException(status_code=500, detail=f"转录失败: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """
    应用启动时执行的事件
    """
    # 预加载Whisper模型
    load_whisper_model()


@app.get("/")
async def root():
    """
    根路由，返回服务信息
    """
    return {"message": "高精度Whisper语音识别服务已运行"}


@app.get("/health")
async def health_check():
    """
    健康检查端点
    """
    return {"status": "ok", "model_loaded": whisper_model is not None}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """
    转录上传的音频文件
    """
    # 检查文件类型
    content_type = file.content_type
    if not content_type or not content_type.startswith(("audio/", "video/")):
        raise HTTPException(
            status_code=400, 
            detail="不支持的文件类型。请上传音频文件。"
        )
    
    # 保存上传的文件
    try:
        # 创建临时文件
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            # 读取上传的文件内容
            content = await file.read()
            # 写入临时文件
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # 使用Whisper进行转录
        transcript = transcribe_with_whisper(temp_file_path, language)
        
        # 返回转录结果
        return JSONResponse(
            content={
                "text": transcript,
                "filename": file.filename
            }
        )
    except Exception as e:
        logger.exception(f"处理音频文件时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理音频文件时出错: {str(e)}")
    finally:
        # 清理临时文件
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.get("/models")
async def list_available_models():
    """
    列出可用的Whisper模型
    """
    available_models = [
        {"id": "tiny", "name": "Whisper Tiny", "description": "最小的模型，速度最快但精度最低"},
        {"id": "base", "name": "Whisper Base", "description": "基础模型，平衡速度和精度"},
        {"id": "small", "name": "Whisper Small", "description": "小型模型，精度较高"},
        {"id": "medium", "name": "Whisper Medium", "description": "中型模型，高精度"},
        {"id": "large", "name": "Whisper Large", "description": "大型模型，最高精度"},
        {"id": "large-v2", "name": "Whisper Large V2", "description": "大型模型V2版本，改进的精度"},
        {"id": "large-v3", "name": "Whisper Large V3", "description": "大型模型V3版本，最新最高精度"}
    ]
    
    return {"models": available_models}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 