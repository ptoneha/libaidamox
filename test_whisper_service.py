#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Whisper语音识别服务的脚本
"""

import os
import sys
import argparse
import requests
from pydub import AudioSegment
import tempfile
import time


def record_audio(output_file, duration=5, rate=16000):
    """
    录制音频（需要PyAudio库）
    """
    try:
        import pyaudio
        import wave
        
        print(f"录制 {duration} 秒的音频...")
        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = rate
        
        p = pyaudio.PyAudio()
        
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
            # 显示录制进度
            sys.stdout.write(f"\r录制中: {i*CHUNK/RATE:.1f}/{duration}秒")
            sys.stdout.flush()
        
        print("\n录制完成!")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return output_file
    except ImportError:
        print("错误: 需要安装PyAudio库才能录制音频")
        print("请运行: pip install pyaudio")
        sys.exit(1)


def convert_audio_to_mp3(input_file, output_file=None):
    """
    将音频文件转换为MP3格式
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".mp3"
    
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="mp3")
    
    return output_file


def transcribe_audio(audio_file, service_url, language=None):
    """
    使用Whisper服务转录音频文件
    """
    print(f"发送音频文件到 {service_url} 进行转录...")
    
    files = {'file': open(audio_file, 'rb')}
    data = {}
    
    if language:
        data['language'] = language
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{service_url}/transcribe",
            files=files,
            data=data
        )
        
        if response.status_code != 200:
            print(f"错误: 服务返回状态码 {response.status_code}")
            print(response.text)
            return None
        
        result = response.json()
        elapsed_time = time.time() - start_time
        
        return {
            'text': result['text'],
            'elapsed_time': elapsed_time
        }
    except Exception as e:
        print(f"错误: {str(e)}")
        return None
    finally:
        files['file'].close()


def test_service_health(service_url):
    """
    测试Whisper服务的健康状态
    """
    try:
        response = requests.get(f"{service_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get('status') == 'ok':
                print("服务状态: 正常")
                print(f"模型加载状态: {'已加载' if health_data.get('model_loaded') else '未加载'}")
                return True
            else:
                print(f"服务状态: {health_data.get('status', '未知')}")
                return False
        else:
            print(f"错误: 服务返回状态码 {response.status_code}")
            return False
    except Exception as e:
        print(f"错误: 无法连接到服务 - {str(e)}")
        return False


def list_available_models(service_url):
    """
    列出可用的Whisper模型
    """
    try:
        response = requests.get(f"{service_url}/models")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("可用的Whisper模型:")
            for model in models:
                print(f"- {model['name']}: {model['description']}")
            return models
        else:
            print(f"错误: 服务返回状态码 {response.status_code}")
            return []
    except Exception as e:
        print(f"错误: 无法连接到服务 - {str(e)}")
        return []


def main():
    parser = argparse.ArgumentParser(description="测试Whisper语音识别服务")
    
    # 命令行参数
    parser.add_argument("--url", default="http://localhost:8000", help="Whisper服务URL")
    parser.add_argument("--file", help="要转录的音频文件路径")
    parser.add_argument("--record", action="store_true", help="录制新的音频进行测试")
    parser.add_argument("--duration", type=int, default=5, help="录制音频的时长(秒)")
    parser.add_argument("--language", help="指定语言代码(如'zh'表示中文)")
    parser.add_argument("--health", action="store_true", help="检查服务健康状态")
    parser.add_argument("--models", action="store_true", help="列出可用的模型")
    
    args = parser.parse_args()
    
    # 检查服务健康状态
    if args.health or (not args.models and not args.file and not args.record):
        if not test_service_health(args.url):
            print("服务健康检查失败，请确保服务正在运行")
            sys.exit(1)
    
    # 列出可用模型
    if args.models:
        list_available_models(args.url)
        return
    
    # 准备音频文件
    audio_file = None
    
    if args.record:
        # 录制新的音频
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        audio_file = record_audio(temp_wav, duration=args.duration)
        
        # 转换为MP3格式
        temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        audio_file = convert_audio_to_mp3(audio_file, temp_mp3)
    elif args.file:
        # 使用提供的音频文件
        if not os.path.exists(args.file):
            print(f"错误: 文件 '{args.file}' 不存在")
            sys.exit(1)
        audio_file = args.file
    else:
        # 如果没有指定文件也没有录制，则退出
        print("请指定音频文件(--file)或使用--record录制新的音频")
        sys.exit(1)
    
    # 转录音频
    result = transcribe_audio(audio_file, args.url, args.language)
    
    if result:
        print("\n转录结果:")
        print("-" * 40)
        print(result['text'])
        print("-" * 40)
        print(f"处理时间: {result['elapsed_time']:.2f} 秒")
    
    # 清理临时文件
    if args.record:
        try:
            os.unlink(temp_wav)
            os.unlink(temp_mp3)
        except:
            pass


if __name__ == "__main__":
    main() 