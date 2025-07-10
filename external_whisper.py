
# 高精度Whisper语音识别代理
# 将此文件保存为 open-webui/backend/open_webui/routers/external_whisper.py

import logging
import os
import uuid
import aiohttp
import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from open_webui.utils.auth import get_verified_user
from open_webui.config import CACHE_DIR

router = APIRouter()

log = logging.getLogger(__name__)

# Whisper服务URL
WHISPER_SERVICE_URL = "http://localhost:8000"

@router.post("/transcriptions")
async def transcription(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    user=Depends(get_verified_user),
):
    """
    将音频转录请求代理到外部Whisper服务
    """
    try:
        # 保存上传的文件
        ext = file.filename.split(".")[-1]
        id = uuid.uuid4()
        filename = f"{id}.{ext}"
        
        file_dir = f"{CACHE_DIR}/audio/transcriptions"
        os.makedirs(file_dir, exist_ok=True)
        file_path = f"{file_dir}/{filename}"
        
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # 准备发送到Whisper服务的请求
        form_data = aiohttp.FormData()
        form_data.add_field(
            'file',
            open(file_path, 'rb'),
            filename=filename,
            content_type=file.content_type
        )
        
        if language:
            form_data.add_field('language', language)
        
        # 发送请求到Whisper服务
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WHISPER_SERVICE_URL}/transcribe",
                data=form_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    log.error(f"Whisper服务返回错误: {error_text}")
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Whisper服务错误: {error_text}"
                    )
                
                result = await response.json()
                return {
                    **result,
                    "filename": os.path.basename(file_path),
                }
    
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理转录请求时出错: {str(e)}"
        )
