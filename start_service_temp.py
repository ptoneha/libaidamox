#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import subprocess
import time
import signal
import shutil
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("start_service")

def check_dependencies():
    """检查依赖是否已安装"""
    try:
        import faster_whisper
        import torch
        import fastapi
        import uvicorn
    except ImportError as e:
        log.error(f"缺少依赖: {e}")
        log.info("正在尝试安装依赖...")
        
        pip_cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        try:
            subprocess.run(pip_cmd, check=True)
            log.info("依赖安装完成")
        except subprocess.CalledProcessError:
            log.error("依赖安装失败，请手动运行: pip install -r requirements.txt")
            return False
    
    return True

def check_ffmpeg():
    """检查ffmpeg是否已安装"""
    if shutil.which('ffmpeg') is None:
        log.error("未找到ffmpeg，音频处理可能会受影响")
        log.info("请安装ffmpeg: https://ffmpeg.org/download.html")
        return False
    return True

def create_directories():
    """创建必要的目录"""
    Path("cache/audio/transcriptions").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    log.info("已创建必要的目录")

def detect_gpu():
    """检测GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            log.info(f"检测到GPU: {device_name} (共 {device_count} 个设备)")
            return True
        else:
            log.warning("未检测到GPU，将使用CPU模式")
            return False
    except:
        log.warning("检测GPU时出错，将使用CPU模式")
        return False

def load_environment_variables(env_file=".env"):
    """加载.env文件中的环境变量"""
    try:
        from dotenv import load_dotenv
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
            log.info(f"已加载环境变量文件: {env_file}")
        else:
            log.info(f"未找到环境变量文件: {env_file}，使用默认设置")
    except ImportError:
        log.warning("python-dotenv未安装，无法加载.env文件")

def start_service(port=8000, host="0.0.0.0", reload=False):
    """启动Whisper API服务"""
    if not os.environ.get("DEVICE_TYPE"):
        os.environ["DEVICE_TYPE"] = "cuda" if detect_gpu() else "cpu"
    
    log.info(f"启动Whisper API服务，监听 {host}:{port}，设备: {os.environ.get('DEVICE_TYPE')}")
    
    cmd = [
        sys.executable, 
        "whisper_api.py"
    ]
    
    if port != 8000:
        os.environ["PORT"] = str(port)
    
    try:
        process = subprocess.Popen(cmd)
        
        def signal_handler(sig, frame):
            log.info("正在停止服务...")
            process.terminate()
            process.wait(timeout=5)
            log.info("服务已停止")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        process.wait()
    except KeyboardInterrupt:
        log.info("接收到中断信号，正在停止服务...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        log.info("服务已停止")
    except Exception as e:
        log.error(f"启动服务时出错: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="启动李白语音转写服务")
    parser.add_argument(
        "--port", "-p", 
        type=int, 
        default=8000, 
        help="API服务端口 (默认: 8000)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="API服务主机 (默认: 0.0.0.0，监听所有接口)"
    )
    parser.add_argument(
        "--env", "-e", 
        type=str, 
        default=".env", 
        help="环境变量文件路径 (默认: .env)"
    )
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        help="Whisper模型名称 (tiny, base, small, medium, large, large-v2, large-v3)"
    )
    parser.add_argument(
        "--device", "-d", 
        type=str, 
        choices=["cuda", "cpu"], 
        help="运行设备 (cuda 或 cpu)"
    )
    parser.add_argument(
        "--reload", "-r", 
        action="store_true", 
        help="启用热重载 (开发模式)"
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 检查ffmpeg
    check_ffmpeg()
    
    # 创建必要的目录
    create_directories()
    
    # 加载环境变量
    load_environment_variables(args.env)
    
    # 设置命令行参数指定的环境变量
    if args.model:
        os.environ["WHISPER_MODEL"] = args.model
        log.info(f"使用模型: {args.model}")
    
    if args.device:
        os.environ["DEVICE_TYPE"] = args.device
        log.info(f"使用设备: {args.device}")
    
    # 启动服务
    success = start_service(port=args.port, host=args.host, reload=args.reload)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 