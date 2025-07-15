#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import requests
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod
from pathlib import Path
import torch

# 配置日志
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("speech_service")

class SpeechTranscriptionService(ABC):
    """语音转写服务的抽象基类"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化模型和资源"""
        pass
    
    @abstractmethod
    def transcribe(self, audio_file: str, language: Optional[str] = None) -> Dict[str, Any]:
        """转写音频文件"""
        pass
    
    def cleanup(self) -> None:
        """清理资源（如果需要）"""
        pass
    
    @property
    def name(self) -> str:
        """获取服务名称"""
        return self.__class__.__name__


class LocalWhisperService(SpeechTranscriptionService):
    """本地Faster-Whisper模型服务"""
    
    def __init__(self, model_name: str = "base", device: str = None, 
                 compute_type: str = "int8", download_root: str = "./models",
                 vad_filter: bool = True):
        """
        初始化本地Whisper服务
        
        Args:
            model_name: 模型名称 ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3")
            device: 设备类型 ("cuda", "cpu", "auto")
            compute_type: 计算类型 ("float16", "int8", "int4")
            download_root: 模型下载和缓存目录
            vad_filter: 是否使用语音活动检测过滤
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type
        self.download_root = download_root
        self.vad_filter = vad_filter
        self.model = None
        
        # 创建模型目录
        os.makedirs(download_root, exist_ok=True)
    
    def initialize(self) -> bool:
        """初始化Faster-Whisper模型"""
        try:
            from faster_whisper import WhisperModel
            
            log.info(f"正在加载Whisper模型 {self.model_name}，设备: {self.device}")
            self.model = WhisperModel(
                model_size_or_path=self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root
            )
            log.info(f"Whisper模型 {self.model_name} 加载完成")
            return True
        except Exception as e:
            log.error(f"加载Whisper模型失败: {str(e)}")
            return False
    
    def transcribe(self, audio_file: str, language: Optional[str] = None) -> Dict[str, Any]:
        """使用本地Whisper模型转写音频"""
        if not self.model:
            if not self.initialize():
                raise RuntimeError("模型初始化失败")
        
        try:
            segments, info = self.model.transcribe(
                audio_file,
                beam_size=5,
                vad_filter=self.vad_filter,
                language=language,
            )
            
            log.info(
                f"检测到语言 '{info.language}' 可信度 {info.language_probability:.2f}"
            )
            
            transcript = "".join([segment.text for segment in list(segments)])
            return {"text": transcript.strip(), "language": info.language}
        
        except Exception as e:
            log.exception(f"转写音频文件时出错: {str(e)}")
            raise


class RemoteWhisperService(SpeechTranscriptionService):
    """远程Whisper API服务"""
    
    def __init__(self, api_url: str, timeout: int = 30):
        """
        初始化远程Whisper服务
        
        Args:
            api_url: Whisper API的URL
            timeout: 请求超时时间（秒）
        """
        self.api_url = api_url
        self.timeout = timeout
    
    def initialize(self) -> bool:
        """检查远程服务是否可用"""
        if not self.api_url:
            log.error("未配置API URL")
            return False
        
        try:
            # 尝试检查服务是否可用
            response = requests.get(
                f"{self.api_url}/health", 
                timeout=self.timeout
            )
            if response.status_code == 200:
                log.info(f"远程Whisper服务可用: {self.api_url}")
                return True
            else:
                log.warning(f"远程Whisper服务可能不可用, 状态码: {response.status_code}")
                return True  # 仍然返回True，因为有些服务可能没有健康检查端点
        except requests.exceptions.RequestException as e:
            log.warning(f"无法连接到远程Whisper服务: {str(e)}")
            return True  # 仍然返回True，等到实际转写时再处理错误
    
    def transcribe(self, audio_file: str, language: Optional[str] = None) -> Dict[str, Any]:
        """使用远程Whisper API转写音频"""
        try:
            with open(audio_file, "rb") as f:
                files = {"file": (os.path.basename(audio_file), f)}
                data = {}
                
                if language:
                    data["language"] = language
                    log.info(f"使用语言设置: {language}")
                
                log.info(f"发送请求到远程Whisper服务: {self.api_url}/transcribe")
                response = requests.post(
                    f"{self.api_url}/transcribe",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.ConnectionError as e:
            log.error(f"连接到远程Whisper服务失败: {str(e)}")
            raise RuntimeError(f"无法连接到远程Whisper服务 {self.api_url}: {str(e)}")
        
        except requests.exceptions.Timeout as e:
            log.error(f"远程Whisper服务请求超时: {str(e)}")
            raise RuntimeError(f"远程Whisper服务请求超时: {str(e)}")
        
        except requests.exceptions.HTTPError as e:
            log.error(f"远程Whisper服务返回HTTP错误: {str(e)}")
            error_detail = "未知错误"
            try:
                error_json = response.json()
                if "error" in error_json:
                    error_detail = error_json["error"]
            except:
                error_detail = response.text[:100] if response.text else str(e)
                
            if response.status_code == 404:
                raise RuntimeError(f"远程Whisper服务API端点不存在 (/transcribe)")
            elif response.status_code == 415:
                raise RuntimeError(f"不支持的媒体类型，请检查音频格式")
            else:
                raise RuntimeError(f"远程Whisper服务错误: 状态码 {response.status_code}, {error_detail}")
        
        except Exception as e:
            log.exception(f"转写请求处理出错: {str(e)}")
            raise RuntimeError(f"转写请求处理出错: {str(e)}")


class OpenAIWhisperService(SpeechTranscriptionService):
    """OpenAI Whisper API服务"""
    
    def __init__(self, api_key: str, api_base_url: str = "https://api.openai.com", model: str = "whisper-1"):
        """
        初始化OpenAI Whisper服务
        
        Args:
            api_key: OpenAI API密钥
            api_base_url: OpenAI API基础URL
            model: Whisper模型名称
        """
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.model = model
    
    def initialize(self) -> bool:
        """验证API密钥和基础URL"""
        if not self.api_key:
            log.error("未设置OpenAI API密钥")
            return False
        
        if not self.api_base_url:
            log.error("未设置OpenAI API基础URL")
            return False
        
        return True
    
    def transcribe(self, audio_file: str, language: Optional[str] = None) -> Dict[str, Any]:
        """使用OpenAI Whisper API转写音频"""
        if not self.initialize():
            raise RuntimeError("OpenAI Whisper服务初始化失败")
        
        try:
            with open(audio_file, "rb") as f:
                response = requests.post(
                    f"{self.api_base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": (os.path.basename(audio_file), f)},
                    data={
                        "model": self.model,
                        **({"language": language} if language else {})
                    }
                )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            log.exception(f"OpenAI Whisper转写失败: {str(e)}")
            error_detail = None
            try:
                if hasattr(response, "json"):
                    res = response.json()
                    if "error" in res:
                        error_detail = f"OpenAI错误: {res['error'].get('message', '')}"
            except:
                error_detail = f"OpenAI错误: {str(e)}"
            
            raise RuntimeError(error_detail or "OpenAI Whisper服务连接错误")


class DeepgramService(SpeechTranscriptionService):
    """Deepgram API服务"""
    
    def __init__(self, api_key: str, model: Optional[str] = None):
        """
        初始化Deepgram服务
        
        Args:
            api_key: Deepgram API密钥
            model: Deepgram模型名称
        """
        self.api_key = api_key
        self.model = model
    
    def initialize(self) -> bool:
        """验证API密钥"""
        if not self.api_key:
            log.error("未设置Deepgram API密钥")
            return False
        
        return True
    
    def transcribe(self, audio_file: str, language: Optional[str] = None) -> Dict[str, Any]:
        """使用Deepgram API转写音频"""
        if not self.initialize():
            raise RuntimeError("Deepgram服务初始化失败")
        
        try:
            import mimetypes
            mime, _ = mimetypes.guess_type(audio_file)
            if not mime:
                mime = "audio/wav"  # 无法检测时默认为wav
                
            with open(audio_file, "rb") as f:
                file_data = f.read()
                
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": mime,
            }
            
            params = {}
            if self.model:
                params["model"] = self.model
                
            response = requests.post(
                "https://api.deepgram.com/v1/listen?smart_format=true",
                headers=headers,
                params=params,
                data=file_data,
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            try:
                transcript = response_data["results"]["channels"][0]["alternatives"][0].get("transcript", "")
                return {"text": transcript.strip()}
            except (KeyError, IndexError) as e:
                log.error(f"Deepgram响应格式错误: {str(e)}")
                raise RuntimeError("解析Deepgram响应失败 - 意外的响应格式")
                
        except Exception as e:
            log.exception(f"Deepgram转写失败: {str(e)}")
            raise RuntimeError(f"Deepgram服务错误: {str(e)}")


class SpeechServiceManager:
    """语音服务管理器，管理多种语音服务"""
    
    def __init__(self):
        self.services = {}
        self.current_service = None
    
    def register_service(self, service_id: str, service: SpeechTranscriptionService) -> None:
        """注册一个语音服务"""
        self.services[service_id] = service
        log.info(f"注册了语音服务: {service_id} ({service.name})")
    
    def set_current_service(self, service_id: str) -> bool:
        """设置当前使用的语音服务"""
        if service_id not in self.services:
            log.error(f"未找到服务ID: {service_id}")
            return False
        
        self.current_service = service_id
        log.info(f"当前语音服务设置为: {service_id}")
        return True
    
    def get_service(self, service_id: Optional[str] = None) -> SpeechTranscriptionService:
        """获取指定的语音服务，如果未指定则返回当前服务"""
        sid = service_id or self.current_service
        if not sid or sid not in self.services:
            available = list(self.services.keys())
            raise ValueError(f"未找到有效的语音服务，可用服务: {available}")
        
        return self.services[sid]
    
    def transcribe(self, 
                  audio_file: str, 
                  service_id: Optional[str] = None, 
                  language: Optional[str] = None) -> Dict[str, Any]:
        """使用指定服务或当前服务转写音频"""
        service = self.get_service(service_id)
        return service.transcribe(audio_file, language)
    
    def list_services(self) -> List[Dict[str, str]]:
        """列出所有注册的服务"""
        return [{"id": sid, "name": service.name} for sid, service in self.services.items()]


# 创建并配置全局服务管理器实例
def create_speech_service_manager(config: Dict[str, Any]) -> SpeechServiceManager:
    """
    根据配置创建并配置语音服务管理器
    
    Args:
        config: 配置字典，包含各个服务的配置
    
    Returns:
        配置好的SpeechServiceManager实例
    """
    manager = SpeechServiceManager()
    
    # 注册本地Whisper服务（如果配置了）
    if config.get("enable_local_whisper", False):
        local_service = LocalWhisperService(
            model_name=config.get("local_whisper_model", "base"),
            device=config.get("device_type"),
            download_root=config.get("whisper_model_dir", "./models"),
            vad_filter=config.get("whisper_vad_filter", True)
        )
        manager.register_service("local_whisper", local_service)
    
    # 注册远程Whisper服务（如果配置了）
    if config.get("external_whisper_url"):
        remote_service = RemoteWhisperService(
            api_url=config.get("external_whisper_url"),
            timeout=config.get("request_timeout", 30)
        )
        manager.register_service("remote_whisper", remote_service)
    
    # 注册OpenAI Whisper服务（如果配置了）
    if config.get("openai_api_key"):
        openai_service = OpenAIWhisperService(
            api_key=config.get("openai_api_key"),
            api_base_url=config.get("openai_api_base_url", "https://api.openai.com"),
            model=config.get("openai_whisper_model", "whisper-1")
        )
        manager.register_service("openai_whisper", openai_service)
    
    # 注册Deepgram服务（如果配置了）
    if config.get("deepgram_api_key"):
        deepgram_service = DeepgramService(
            api_key=config.get("deepgram_api_key"),
            model=config.get("deepgram_model")
        )
        manager.register_service("deepgram", deepgram_service)
    
    # 设置默认服务
    default_service = config.get("default_service")
    if default_service and default_service in manager.services:
        manager.set_current_service(default_service)
    elif manager.services:
        # 如果没有指定默认服务但有可用服务，使用第一个
        first_service = next(iter(manager.services))
        manager.set_current_service(first_service)
        log.info(f"未指定默认语音服务，使用: {first_service}")
    
    return manager


# 使用示例
if __name__ == "__main__":
    # 示例配置
    config = {
        "enable_local_whisper": True,
        "local_whisper_model": "base",
        "device_type": "cpu",
        "whisper_model_dir": "./models",
        "external_whisper_url": "http://localhost:8000",
        "default_service": "local_whisper"
    }
    
    # 创建服务管理器
    manager = create_speech_service_manager(config)
    
    # 列出可用服务
    services = manager.list_services()
    print(f"可用服务: {services}")
    
    # 使用指定服务转写音频
    try:
        result = manager.transcribe("sample.wav", service_id="local_whisper")
        print(f"转写结果: {result}")
    except Exception as e:
        print(f"转写失败: {str(e)}") 