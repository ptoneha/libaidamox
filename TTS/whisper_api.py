#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import torch
import uuid
import re

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 导入语音服务模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speech_service import (
    create_speech_service_manager,
    LocalWhisperService,
    RemoteWhisperService,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("whisper_api.log")
    ]
)
log = logging.getLogger("whisper_api")

# 环境变量配置
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
DEVICE_TYPE = os.getenv("DEVICE_TYPE", "cuda" if torch.cuda.is_available() else "cpu")
WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", "./models")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "zh")
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "True").lower() in ("true", "1", "t")

# 创建临时文件目录
os.makedirs(f"{CACHE_DIR}/audio/transcriptions", exist_ok=True)

# 创建FastAPI应用
app = FastAPI(
    title="Whisper API",
    description="语音识别API服务，基于OpenAI的Whisper模型",
    version="1.0.0",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建语音服务管理器
config = {
    "enable_local_whisper": True,
    "local_whisper_model": WHISPER_MODEL,
    "device_type": DEVICE_TYPE,
    "whisper_model_dir": WHISPER_MODEL_DIR,
    "whisper_vad_filter": WHISPER_VAD_FILTER,
    "default_service": "local_whisper"
}

service_manager = create_speech_service_manager(config)

def convert_to_simplified_chinese(text):
    """
    将文本中的繁体中文转换为简体中文
    
    Args:
        text: 包含繁体中文的文本
        
    Returns:
        转换后的简体中文文本
    """
    try:
        # 尝试导入zhconv库
        import zhconv
        
        # 使用zhconv将文本转换为简体中文
        simplified_text = zhconv.convert(text, 'zh-cn')
        
        # 如果转换前后文本不同，记录日志
        if simplified_text != text:
            log.info(f"已将繁体中文转换为简体中文")
            
        return simplified_text
    except ImportError:
        log.warning("未安装zhconv库，无法进行繁简转换，请使用pip install zhconv安装")
        return text
    except Exception as e:
        log.warning(f"繁简转换出错: {str(e)}")
        return text

def preprocess_audio(audio_path, silence_threshold=-50, min_silence_len=500):
    """
    预处理音频：裁剪前后的静音段落，并转换为WAV格式
    
    Args:
        audio_path: 原始音频文件路径
        silence_threshold: 静音检测阈值(dB)
        min_silence_len: 最小静音长度(ms)
    
    Returns:
        处理后的WAV文件路径
    """
    log.info(f"开始预处理音频文件: {audio_path}")
    
    try:
        # 导入pydub库
        from pydub import AudioSegment
        from pydub.silence import detect_leading_silence
        
        # 加载音频文件
        sound = AudioSegment.from_file(audio_path)
        
        # 获取原始音频信息
        original_channels = sound.channels
        original_sample_rate = sound.frame_rate
        log.info(f"原始音频: {original_channels}声道, {original_sample_rate}Hz, 时长: {len(sound)/1000.0:.2f}秒")
        
        # 检测并裁剪前端静音
        start_trim = detect_leading_silence(sound, silence_threshold=silence_threshold)
        
        # 反转音频检测尾部静音
        end_trim = detect_leading_silence(sound.reverse(), silence_threshold=silence_threshold)
        
        # 裁剪音频
        trimmed_sound = sound[start_trim:len(sound)-end_trim]
        
        # 如果裁剪后音频太短，则使用原始音频
        if len(trimmed_sound) < 1000:  # 小于1秒
            log.warning("裁剪后音频太短，使用原始音频")
            trimmed_sound = sound
            
        # 统一转换为单声道16kHz格式（Whisper模型的最佳输入格式）
        if trimmed_sound.channels > 1:
            trimmed_sound = trimmed_sound.set_channels(1)
            log.info(f"音频已转换为单声道")
            
        if trimmed_sound.frame_rate != 16000:
            trimmed_sound = trimmed_sound.set_frame_rate(16000)
            log.info(f"音频已转换为16kHz采样率")
        
        # 保存为WAV格式
        output_path = f"{os.path.dirname(audio_path)}/{str(uuid.uuid4())}.wav"
        trimmed_sound.export(output_path, format="wav")
        
        duration_before = len(sound) / 1000.0  # 秒
        duration_after = len(trimmed_sound) / 1000.0  # 秒
        log.info(f"音频预处理完成: 从 {duration_before:.2f}秒 裁剪到 {duration_after:.2f}秒, mono+16kHz格式")
        
        return output_path
    except Exception as e:
        log.exception(f"音频预处理失败: {str(e)}")
        return audio_path  # 如果处理失败，返回原始文件路径

@app.on_event("startup")
async def startup_event():
    """应用启动时执行的操作"""
    log.info("Whisper API 服务正在启动...")
    
    # 预加载模型（可选，如果希望启动时就加载模型）
    try:
        service = service_manager.get_service("local_whisper")
        service.initialize()
        log.info(f"Whisper模型 {WHISPER_MODEL} 已预加载")
    except Exception as e:
        log.warning(f"预加载Whisper模型失败: {str(e)}")


@app.get("/")
async def read_root():
    """API根路径，返回欢迎信息"""
    return {
        "status": "ok",
        "message": "Whisper API 服务运行中",
        "model": WHISPER_MODEL,
        "device": DEVICE_TYPE,
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok"}


@app.get("/services")
async def list_services():
    """列出所有可用的语音服务"""
    services = service_manager.list_services()
    current_service = service_manager.current_service
    return {
        "services": services,
        "current": current_service
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    service_id: Optional[str] = Form(None),
):
    """
    转写音频文件
    
    - **file**: 音频文件
    - **language**: 语言代码，如 'zh', 'en'（可选）
    - **service_id**: 使用的服务ID（可选）
    
    返回转写结果
    """
    # 检查文件内容类型
    content_type = file.content_type
    if not (content_type.startswith('audio/') or content_type.startswith('video/')):
        raise HTTPException(
            status_code=415, 
            detail=f"不支持的文件类型: {content_type}，请上传音频文件"
        )
    
    # 保存上传的文件
    try:
        file_content = await file.read()
        
        # 检查文件大小和内容
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="音频文件内容为空")
        
        # 检查文件格式是否有效
        if len(file_content) < 44:  # 至少要包含基本音频头信息
            raise HTTPException(status_code=400, detail="无效的音频文件格式")
        
        # 保存临时文件
        file_ext = os.path.splitext(file.filename)[1] or ".wav"
        if file_ext.startswith('.'):
            file_ext = file_ext[1:]
            
        temp_file_id = str(uuid.uuid4())
        original_file_path = f"{CACHE_DIR}/audio/transcriptions/{temp_file_id}.{file_ext}"
        
        with open(original_file_path, "wb") as f:
            f.write(file_content)
        
        # 预处理音频：裁剪静音并转为WAV
        processed_file_path = preprocess_audio(original_file_path)
        
        # 使用语音服务转写预处理后的音频
        try:
            actual_language = language or WHISPER_LANGUAGE
            log.info(f"开始转写预处理后的音频: {processed_file_path}, 语言: {actual_language}")
            
            result = service_manager.transcribe(
                processed_file_path, 
                service_id=service_id,
                language=actual_language
            )
            
            # 对中文结果进行繁体到简体的转换
            if result.get("text") and (actual_language.startswith("zh") or actual_language == "auto"):
                original_text = result["text"]
                result["text"] = convert_to_simplified_chinese(original_text)
            
            return result
            
        except Exception as e:
            log.exception(f"转写音频时出错: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"转写音频时出错: {str(e)}"
            )
        finally:
            # 清理临时文件
            try:
                if original_file_path != processed_file_path:
                    os.unlink(processed_file_path)
                os.unlink(original_file_path)
            except Exception as e:
                log.warning(f"清理临时文件时出错: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"处理音频文件时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"处理音频文件时出错: {str(e)}"
        )


@app.post("/change-service")
async def change_service(service_id: str):
    """切换当前使用的语音服务"""
    try:
        if service_manager.set_current_service(service_id):
            return {"status": "ok", "message": f"当前服务已切换为: {service_id}"}
        else:
            raise HTTPException(status_code=404, detail=f"未找到服务: {service_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 启动服务器
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    log.info(f"启动Whisper API服务，端口: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 